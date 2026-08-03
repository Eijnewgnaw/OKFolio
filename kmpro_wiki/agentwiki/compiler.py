from __future__ import annotations

import filecmp
import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from .assets import (
    SourceAsset,
    apply_asset_placements,
    inventory_assets,
    strip_missing_image_references,
    validate_asset_preservation,
)
from .config import Settings
from .contracts import (
    COMPILE_SCHEMA_VERSION,
    DISCOVER_SCHEMA_VERSION,
    ENRICH_SCHEMA_VERSION,
    PRESERVE_SCHEMA_VERSION,
    AssetPlacement,
    ConceptRef,
    DraftConcept,
    LinkSuggestion,
    RelationAudit,
)
from .okf import (
    ConceptDocument,
    OKFValidationError,
    parse_concept_markdown,
    rewrite_image_paths,
    validate_concept,
)
from .relations import apply_relation_audit
from .stages import (
    audit_relations,
    compile_one_concept,
    discover_concepts,
    plan_asset_placements,
)
from .state import (
    CompileFingerprint,
    Manifest,
    SourceState,
    StageCache,
    md5_file,
    stable_hash,
)


class CompletionClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        json_schema_name: str | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class CompileSummary:
    compiled: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: tuple[str, ...]


class CompilationBatchError(RuntimeError):
    def __init__(self, summary: CompileSummary):
        super().__init__(f"compilation failed for: {', '.join(summary.failed)}")
        self.summary = summary


@dataclass
class _PreparedSource:
    path: Path
    content: str
    assets: tuple[SourceAsset, ...]
    fingerprint: CompileFingerprint
    cache: StageCache


@dataclass
class _StagedSource:
    prepared: _PreparedSource
    concepts: tuple[ConceptDocument, ...]
    preservation_key: str
    discovery_status: str = "success"
    concept_status: str = "success"
    preservation_status: str = "success"


class _CapturingLLM:
    def __init__(self, delegate: CompletionClient):
        self.delegate = delegate
        self.responses: list[str] = []

    @property
    def last_response(self) -> str | None:
        return self.responses[-1] if self.responses else None

    def complete(
        self,
        prompt: str,
        *,
        json_schema_name: str | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> str:
        response = self.delegate.complete(
            prompt,
            json_schema_name=json_schema_name,
            json_schema=json_schema,
        )
        self.responses.append(response)
        return response


class Compiler:
    def __init__(
        self,
        settings: Settings,
        llm: CompletionClient,
        *,
        on_event: Callable[[str], None] | None = None,
    ):
        self.settings = settings
        self.llm = llm
        self.on_event = on_event or (lambda _message: None)
        self.sources_dir = settings.sources_dir
        self.source_images_dir = self.sources_dir / "images"
        self.wiki_dir = settings.data_dir / "wiki"
        self.concepts_dir = self.wiki_dir / "concepts"
        self.wiki_images_dir = self.wiki_dir / "images"
        self.state_path = settings.data_dir / ".state" / "manifest.json"
        self.staging_root = settings.data_dir / ".staging"

    def run(self) -> CompileSummary:
        self.concepts_dir.mkdir(parents=True, exist_ok=True)
        self.source_images_dir.mkdir(parents=True, exist_ok=True)
        manifest = Manifest.load(self.state_path)
        templates = self._load_templates()
        prepared: list[_PreparedSource] = []
        skipped: list[str] = []
        failed: list[str] = []

        for source in sorted(self.sources_dir.glob("*.md"), key=lambda path: path.name):
            content = strip_missing_image_references(
                source.read_text(encoding="utf-8"), self.source_images_dir
            )
            try:
                assets = inventory_assets(content, self.source_images_dir)
                fingerprint = self._fingerprint(source, assets, templates)
            except Exception as error:
                fingerprint = self._fingerprint(source, (), templates)
                self._mark_incomplete(
                    manifest,
                    source.name,
                    fingerprint,
                    discovery="failed",
                    concept="failed",
                    preservation="failed",
                    relation="failed",
                )
                self._write_failure_trace(source.name, "preflight", error, None)
                self.on_event(
                    f"preflight.failed source={source.name} "
                    f"error={type(error).__name__}: {error}"
                )
                failed.append(source.name)
                continue

            current = manifest.sources.get(source.name)
            outputs_exist = current is not None and all(
                (self.wiki_dir / output).is_file() for output in current.outputs
            )
            if not manifest.needs_compile(source.name, fingerprint) and outputs_exist:
                skipped.append(source.name)
                continue
            cache_path = (
                self.staging_root
                / "sources"
                / _md5_text(source.name)
                / "cache.json"
            )
            prepared.append(
                _PreparedSource(
                    path=source,
                    content=content,
                    assets=assets,
                    fingerprint=fingerprint,
                    cache=StageCache.load(cache_path),
                )
            )

        if not prepared and not failed:
            return CompileSummary((), tuple(skipped), ())

        old_selected_outputs = {
            Path(output).name
            for item in prepared
            for output in (
                manifest.sources.get(item.path.name).outputs
                if manifest.sources.get(item.path.name) is not None
                else ()
            )
        }
        reserved = {
            path.stem
            for path in self.concepts_dir.glob("*.md")
            if path.name not in old_selected_outputs
        }

        staged: dict[str, _StagedSource] = {}
        for item in prepared:
            try:
                staged[item.path.name] = self._run_first_three_stages(
                    item, templates, reserved
                )
            except Exception as error:
                self._mark_incomplete(
                    manifest,
                    item.path.name,
                    item.fingerprint,
                    discovery=getattr(error, "discovery_status", "failed"),
                    concept=getattr(error, "concept_status", "failed"),
                    preservation=getattr(error, "preservation_status", "failed"),
                    relation="failed",
                )
                self.on_event(
                    f"compile.failed source={item.path.name} "
                    f"error={type(error).__name__}: {error}"
                )
                failed.append(item.path.name)

        candidates = self._candidate_bundle(staged, old_selected_outputs)
        finalized: dict[str, tuple[tuple[ConceptDocument, ...], str]] = {}
        for source_name, item in staged.items():
            try:
                finalized[source_name] = self._run_relation_stage(
                    item, templates["enrich"], candidates
                )
            except Exception as error:
                self._mark_incomplete(
                    manifest,
                    source_name,
                    item.prepared.fingerprint,
                    discovery="success",
                    concept="success",
                    preservation="success",
                    relation="failed",
                )
                self.on_event(
                    f"relation.failed source={source_name} "
                    f"error={type(error).__name__}: {error}"
                )
                failed.append(source_name)

        compiled: list[str] = []
        if finalized:
            self._sync_images()
        for source_name, (concepts, relation_status) in finalized.items():
            item = staged[source_name]
            try:
                self._publish_source(source_name, list(concepts), manifest)
            except Exception as error:
                self._mark_incomplete(
                    manifest,
                    source_name,
                    item.prepared.fingerprint,
                    discovery="success",
                    concept="success",
                    preservation="success",
                    relation=relation_status,
                )
                self.on_event(
                    f"publish.failed source={source_name} "
                    f"error={type(error).__name__}: {error}"
                )
                failed.append(source_name)
                continue
            manifest.sources[source_name] = SourceState(
                fingerprint=item.prepared.fingerprint,
                outputs=tuple(f"concepts/{concept.filename}" for concept in concepts),
                discovery_status="success",
                concept_status="success",
                preservation_status="success",
                relation_status=relation_status,
                status="complete",
            )
            compiled.append(source_name)
            self.on_event(
                f"publish.done source={source_name} concepts={len(concepts)}"
            )

        manifest.save(self.state_path)
        summary = CompileSummary(
            tuple(sorted(compiled)),
            tuple(skipped),
            tuple(sorted(set(failed))),
        )
        if summary.failed:
            raise CompilationBatchError(summary)
        return summary

    def _run_first_three_stages(
        self,
        item: _PreparedSource,
        templates: dict[str, str],
        reserved: set[str],
    ) -> _StagedSource:
        fingerprint = item.fingerprint
        discovery_key = stable_hash(
            {
                "source_md5": fingerprint.source_md5,
                "asset_sha256": fingerprint.asset_sha256,
                "prompt": fingerprint.discover_prompt_md5,
                "schema": fingerprint.discover_schema_version,
                "model": fingerprint.model,
                "enable_thinking": fingerprint.enable_thinking,
                "max_tokens": fingerprint.max_tokens,
            }
        )
        refs_payload = item.cache.get("discovery", discovery_key)
        if refs_payload is None:
            capture = _CapturingLLM(self.llm)
            self.on_event(f"discovery.start source={item.path.name}")
            try:
                refs = discover_concepts(
                    capture,
                    templates["discover"],
                    title=_extract_title(item.path, item.content),
                    source_name=item.path.name,
                    source_content=item.content,
                    assets=item.assets,
                )
            except Exception as error:
                self._write_raw_responses(item.path.name, "discovery", capture.responses)
                self._write_failure_trace(
                    item.path.name, "discovery", error, capture.last_response
                )
                raise
            self._write_raw_responses(item.path.name, "discovery", capture.responses)
            item.cache.put("discovery", discovery_key, _refs_payload(refs))
            self.on_event(
                f"discovery.done source={item.path.name} concepts={len(refs)}"
            )
        else:
            refs = _refs_from_payload(refs_payload)
            self.on_event(f"cache.hit source={item.path.name} stage=discovery")

        refs = self._assign_ref_ids(refs, item.fingerprint, reserved)
        drafts_list: list[DraftConcept] = []
        concept_item_keys: list[str] = []
        for position, ref in enumerate(refs, start=1):
            concept_item_key = stable_hash(
                {
                    "discovery_key": discovery_key,
                    "ref": _refs_payload((ref,))[0],
                    "prompt": fingerprint.compile_prompt_md5,
                    "schema": fingerprint.compile_schema_version,
                    "model": fingerprint.model,
                    "enable_thinking": fingerprint.enable_thinking,
                    "max_tokens": fingerprint.max_tokens,
                }
            )
            concept_item_keys.append(concept_item_key)
            cache_stage = f"concept:{ref.concept_id}"
            draft_payload = item.cache.get(cache_stage, concept_item_key)
            if draft_payload is not None:
                drafts_list.append(_drafts_from_payload([draft_payload])[0])
                self.on_event(
                    f"cache.hit source={item.path.name} stage=concept "
                    f"concept={ref.concept_id}"
                )
                continue
            capture = _CapturingLLM(self.llm)
            self.on_event(
                f"concept.start concept={ref.concept_id} "
                f"position={position}/{len(refs)} source={item.path.name}"
            )
            try:
                draft = compile_one_concept(
                    capture, templates["compile"], ref
                )
            except Exception as error:
                self._write_raw_responses(
                    item.path.name,
                    f"concept/{position:03d}-{ref.concept_id}",
                    capture.responses,
                )
                self._write_failure_trace(
                    item.path.name, "concept", error, capture.last_response
                )
                setattr(error, "discovery_status", "success")
                raise
            self._write_raw_responses(
                item.path.name,
                f"concept/{position:03d}-{ref.concept_id}",
                capture.responses,
            )
            item.cache.put(
                cache_stage, concept_item_key, _drafts_payload((draft,))[0]
            )
            drafts_list.append(draft)
            self.on_event(
                f"concept.done concept={ref.concept_id} "
                f"position={position}/{len(refs)} source={item.path.name}"
            )
        drafts = tuple(drafts_list)
        concept_key = stable_hash(concept_item_keys)

        preservation_key = stable_hash(
            {
                "concept_key": concept_key,
                "assets": _assets_payload(item.assets),
                "prompt": fingerprint.preserve_prompt_md5,
                "schema": fingerprint.preserve_schema_version,
                "model": fingerprint.model,
                "enable_thinking": fingerprint.enable_thinking,
                "max_tokens": fingerprint.max_tokens,
            }
        )
        concepts_payload = item.cache.get("preservation", preservation_key)
        if concepts_payload is None:
            capture = _CapturingLLM(self.llm)
            try:
                placements = plan_asset_placements(
                    capture,
                    templates["preserve"],
                    assets=item.assets,
                    drafts=drafts,
                )
                concepts = apply_asset_placements(drafts, item.assets, placements)
                for concept in concepts:
                    validate_concept(concept, item.path.name)
                validate_asset_preservation(
                    item.assets, concepts, self.source_images_dir
                )
            except Exception as error:
                self._write_raw_responses(
                    item.path.name, "preservation", capture.responses
                )
                self._write_failure_trace(
                    item.path.name, "preservation", error, capture.last_response
                )
                setattr(error, "discovery_status", "success")
                setattr(error, "concept_status", "success")
                raise
            self._write_raw_responses(item.path.name, "preservation", capture.responses)
            item.cache.put(
                "preservation",
                preservation_key,
                {
                    "placements": _placements_payload(placements),
                    "concepts": _concepts_payload(concepts),
                },
            )
        else:
            concepts = _concepts_from_payload(concepts_payload["concepts"])
            validate_asset_preservation(item.assets, concepts, self.source_images_dir)
            self.on_event(f"cache.hit source={item.path.name} stage=preservation")
        return _StagedSource(item, concepts, preservation_key)

    def _run_relation_stage(
        self,
        item: _StagedSource,
        template: str,
        candidates: dict[str, ConceptDocument],
    ) -> tuple[tuple[ConceptDocument, ...], str]:
        current_ids = tuple(concept.filename[:-3] for concept in item.concepts)
        candidate_fingerprint = stable_hash(
            {
                concept_id: concept.render()
                for concept_id, concept in sorted(candidates.items())
            }
        )
        fingerprint = item.prepared.fingerprint
        relation_key = stable_hash(
            {
                "preservation_key": item.preservation_key,
                "candidates": candidate_fingerprint,
                "prompt": fingerprint.enrich_prompt_md5,
                "schema": fingerprint.enrich_schema_version,
                "model": fingerprint.model,
                "enable_thinking": fingerprint.enable_thinking,
                "max_tokens": fingerprint.max_tokens,
            }
        )
        cached = item.prepared.cache.get("relation", relation_key)
        if cached is None:
            capture = _CapturingLLM(self.llm)
            try:
                audits = audit_relations(
                    capture,
                    template,
                    candidates,
                    current_ids=current_ids,
                    on_event=lambda message: self.on_event(
                        f"{message} source={item.prepared.path.name}"
                    ),
                )
                enriched = tuple(
                    apply_relation_audit(concept, audits[concept.filename[:-3]], candidates)
                    for concept in item.concepts
                )
                for concept in enriched:
                    validate_concept(concept, item.prepared.path.name)
                validate_asset_preservation(
                    item.prepared.assets, enriched, self.source_images_dir
                )
                relation_status = (
                    "success"
                    if any(audit.status == "linked" for audit in audits.values())
                    else "no_links"
                )
            except Exception as error:
                self._write_raw_responses(
                    item.prepared.path.name, "relation", capture.responses
                )
                self._write_failure_trace(
                    item.prepared.path.name,
                    "relation",
                    error,
                    capture.last_response,
                )
                raise
            self._write_raw_responses(
                item.prepared.path.name, "relation", capture.responses
            )
            item.prepared.cache.put(
                "relation",
                relation_key,
                {
                    "audits": _audits_payload(audits),
                    "concepts": _concepts_payload(enriched),
                    "status": relation_status,
                },
            )
        else:
            enriched = _concepts_from_payload(cached["concepts"])
            relation_status = str(cached["status"])
            validate_asset_preservation(
                item.prepared.assets, enriched, self.source_images_dir
            )
            self.on_event(
                f"cache.hit source={item.prepared.path.name} stage=relation"
            )

        final = tuple(
            parse_concept_markdown(
                concept.filename, rewrite_image_paths(concept.render())
            )
            for concept in enriched
        )
        return final, relation_status

    def _load_templates(self) -> dict[str, str]:
        return {
            name: (self.settings.prompts_dir / f"{name}.md").read_text(
                encoding="utf-8"
            )
            for name in ("discover", "compile", "preserve", "enrich")
        }

    def _fingerprint(
        self,
        source: Path,
        assets: tuple[SourceAsset, ...],
        templates: dict[str, str],
    ) -> CompileFingerprint:
        return CompileFingerprint(
            source_md5=md5_file(source),
            asset_sha256=stable_hash(_assets_payload(assets)),
            discover_prompt_md5=_md5_text(templates["discover"]),
            compile_prompt_md5=_md5_text(templates["compile"]),
            preserve_prompt_md5=_md5_text(templates["preserve"]),
            enrich_prompt_md5=_md5_text(templates["enrich"]),
            discover_schema_version=DISCOVER_SCHEMA_VERSION,
            compile_schema_version=COMPILE_SCHEMA_VERSION,
            preserve_schema_version=PRESERVE_SCHEMA_VERSION,
            enrich_schema_version=ENRICH_SCHEMA_VERSION,
            model=self.settings.openai_model,
            enable_thinking=self.settings.openai_enable_thinking,
            max_tokens=self.settings.openai_max_tokens,
        )

    def _assign_ref_ids(
        self,
        refs: tuple[ConceptRef, ...],
        fingerprint: CompileFingerprint,
        reserved: set[str],
    ) -> tuple[ConceptRef, ...]:
        assigned: list[ConceptRef] = []
        local: set[str] = set()
        for ref in refs:
            concept_id = ref.concept_id
            if concept_id in local:
                raise OKFValidationError(f"duplicate concept id: {concept_id}")
            local.add(concept_id)
            if concept_id in reserved:
                concept_id = f"{concept_id}--{fingerprint.source_md5[:8]}"
            if concept_id in reserved:
                raise OKFValidationError(
                    f"unresolved concept id collision: {concept_id}"
                )
            reserved.add(concept_id)
            assigned.append(replace(ref, concept_id=concept_id))
        return tuple(assigned)

    def _candidate_bundle(
        self,
        staged: dict[str, _StagedSource],
        excluded_existing: set[str],
    ) -> dict[str, ConceptDocument]:
        concepts: dict[str, ConceptDocument] = {}
        for path in sorted(self.concepts_dir.glob("*.md"), key=lambda item: item.name):
            if path.name in excluded_existing:
                continue
            try:
                concepts[path.stem] = parse_concept_markdown(
                    path.name, path.read_text(encoding="utf-8")
                )
            except OKFValidationError as error:
                self.on_event(
                    f"candidate.skipped concept={path.name} error={type(error).__name__}: {error}"
                )
        for item in staged.values():
            for concept in item.concepts:
                concepts[concept.filename[:-3]] = concept
        return dict(sorted(concepts.items()))

    def _mark_incomplete(
        self,
        manifest: Manifest,
        source_name: str,
        fingerprint: CompileFingerprint,
        *,
        discovery: str,
        concept: str,
        preservation: str,
        relation: str,
    ) -> None:
        previous = manifest.sources.get(source_name)
        manifest.sources[source_name] = SourceState(
            fingerprint=fingerprint,
            outputs=() if previous is None else previous.outputs,
            discovery_status=discovery,
            concept_status=concept,
            preservation_status=preservation,
            relation_status=relation,
            status="incomplete",
        )

    def _write_failure_trace(
        self,
        source_name: str,
        stage: str,
        error: Exception,
        response: str | None,
    ) -> None:
        failures_dir = self.staging_root / "failures"
        failures_dir.mkdir(parents=True, exist_ok=True)
        trace_id = hashlib.md5(
            f"{source_name}:{stage}".encode("utf-8")
        ).hexdigest()[:12]
        trace = (
            f"stage: {stage}\n"
            f"source: {source_name}\n"
            f"error: {type(error).__name__}: {error}\n"
        )
        if response is not None:
            trace += f"\n{response.rstrip()}\n"
        (failures_dir / f"{trace_id}.txt").write_text(trace, encoding="utf-8")

    def _write_raw_responses(
        self, source_name: str, stage: str, responses: list[str]
    ) -> None:
        if not responses:
            return
        directory = (
            self.staging_root / "sources" / _md5_text(source_name) / "raw" / stage
        )
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        for index, response in enumerate(responses, start=1):
            (directory / f"{index:03d}.txt").write_text(
                response.rstrip() + "\n", encoding="utf-8"
            )

    def _sync_images(self) -> None:
        self.wiki_images_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(self.source_images_dir.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(self.source_images_dir)
            destination = self.wiki_images_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and filecmp.cmp(source, destination, shallow=False):
                continue
            shutil.copy2(source, destination)

    def _publish_source(
        self,
        source_name: str,
        concepts: list[ConceptDocument],
        manifest: Manifest,
    ) -> None:
        transaction = self.staging_root / "publish" / _md5_text(source_name)
        if transaction.exists():
            shutil.rmtree(transaction)
        new_dir = transaction / "new"
        backup_dir = transaction / "backup"
        new_dir.mkdir(parents=True)
        backup_dir.mkdir(parents=True)
        for concept in concepts:
            (new_dir / concept.filename).write_text(
                concept.render(), encoding="utf-8"
            )

        previous = manifest.sources.get(source_name)
        previous_names = (
            [] if previous is None else [Path(path).name for path in previous.outputs]
        )
        published: list[Path] = []
        backed_up: list[tuple[Path, Path]] = []
        try:
            for name in previous_names:
                live = self.concepts_dir / name
                if live.exists():
                    backup = backup_dir / name
                    os.replace(live, backup)
                    backed_up.append((live, backup))
            for concept in concepts:
                staged = new_dir / concept.filename
                live = self.concepts_dir / concept.filename
                os.replace(staged, live)
                published.append(live)
        except Exception:
            for live in published:
                live.unlink(missing_ok=True)
            for live, backup in reversed(backed_up):
                if backup.exists():
                    os.replace(backup, live)
            raise
        finally:
            shutil.rmtree(transaction, ignore_errors=True)


def _refs_payload(refs: tuple[ConceptRef, ...]) -> list[dict[str, Any]]:
    return [
        {
            "concept_id": ref.concept_id,
            "type": ref.type,
            "title": ref.title,
            "description": ref.description,
            "source": ref.source,
            "evidence": list(ref.evidence),
            "asset_hints": list(ref.asset_hints),
            "section_path": list(ref.section_path),
            "page_start": ref.page_start,
            "page_end": ref.page_end,
            "evidence_block_ids": list(ref.evidence_block_ids),
        }
        for ref in refs
    ]


def _refs_from_payload(payload: Any) -> tuple[ConceptRef, ...]:
    if not isinstance(payload, list):
        raise ValueError("invalid cached ConceptRef payload")
    return tuple(
        ConceptRef(
            concept_id=str(item["concept_id"]),
            type=str(item["type"]),
            title=str(item["title"]),
            description=str(item["description"]),
            source=str(item["source"]),
            evidence=tuple(item["evidence"]),
            asset_hints=tuple(item["asset_hints"]),
            section_path=tuple(item.get("section_path") or ()),
            page_start=item.get("page_start"),
            page_end=item.get("page_end"),
            evidence_block_ids=tuple(item.get("evidence_block_ids") or ()),
        )
        for item in payload
    )


def _drafts_payload(drafts: tuple[DraftConcept, ...]) -> list[dict[str, Any]]:
    return [
        {
            "ref": _refs_payload((draft.ref,))[0],
            "title": draft.title,
            "description": draft.description,
            "body": draft.body,
        }
        for draft in drafts
    ]


def _drafts_from_payload(payload: Any) -> tuple[DraftConcept, ...]:
    if not isinstance(payload, list):
        raise ValueError("invalid cached Draft payload")
    return tuple(
        DraftConcept(
            ref=_refs_from_payload([item["ref"]])[0],
            title=str(item["title"]),
            description=str(item["description"]),
            body=str(item["body"]),
        )
        for item in payload
    )


def _assets_payload(assets: tuple[SourceAsset, ...]) -> list[dict[str, Any]]:
    return [
        {
            "asset_id": asset.asset_id,
            "kind": asset.kind,
            "raw": asset.raw,
            "target": asset.target,
            "ordinal": asset.ordinal,
            "sha256": asset.sha256,
        }
        for asset in assets
    ]


def _placements_payload(
    placements: tuple[AssetPlacement, ...],
) -> list[dict[str, str]]:
    return [
        {
            "asset_id": item.asset_id,
            "concept_id": item.concept_id,
            "anchor": item.anchor,
            "position": item.position,
            "reason": item.reason,
        }
        for item in placements
    ]


def _concepts_payload(
    concepts: tuple[ConceptDocument, ...],
) -> list[dict[str, str]]:
    return [
        {"filename": concept.filename, "content": concept.render()}
        for concept in concepts
    ]


def _concepts_from_payload(payload: Any) -> tuple[ConceptDocument, ...]:
    if not isinstance(payload, list):
        raise ValueError("invalid cached Concept payload")
    return tuple(
        parse_concept_markdown(str(item["filename"]), str(item["content"]))
        for item in payload
    )


def _audits_payload(audits: dict[str, RelationAudit]) -> dict[str, Any]:
    return {
        concept_id: {
            "status": audit.status,
            "links": [
                {
                    "target_id": link.target_id,
                    "anchor": link.anchor,
                    "occurrence": link.occurrence,
                    "reason": link.reason,
                }
                for link in audit.links
            ],
        }
        for concept_id, audit in audits.items()
    }


def _md5_text(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def _extract_title(source: Path, content: str | None = None) -> str:
    value = source.read_text(encoding="utf-8") if content is None else content
    match = re.search(r"^#\s+(.+?)\s*$", value, flags=re.MULTILINE)
    return match.group(1) if match else source.stem
