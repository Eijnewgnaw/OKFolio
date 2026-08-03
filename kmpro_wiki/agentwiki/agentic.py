"""Auditable AgentWiki orchestration for corpus-wide knowledge compilation.

The module writes only to ``agent-runs/<run-id>``.  It combines discovery,
one-Concept compilation and asset preservation with bounded model decisions
for routing, Ref refinement, compile grouping and quality-triggered
recompilation.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any, Protocol, TypeVar

from .agent_contracts import (
    AgentPolicy,
    CompileGroup,
    GroupDecision,
    QualityAudit,
    SourcePlan,
    group_plan_schema,
    parse_group_plan,
    parse_quality_audit,
    parse_source_plan,
    quality_schema,
    recover_group_plan,
    source_plan_schema,
)
from .assets import (
    SourceAsset,
    apply_asset_placements,
    inventory_assets,
    strip_missing_image_references,
    validate_asset_preservation,
)
from .config import Settings
from .contracts import (
    ContractError,
    ConceptRef,
    DraftConcept,
    discovery_json_schema,
    draft_json_schema,
    parse_discovery,
    parse_draft,
)
from .global_cluster import CandidateEdge, candidate_edges, kind_for
from .okf import ConceptDocument, normalize_slug, rewrite_image_paths
from .stages import (
    build_evidence_catalog,
    compile_one_concept,
    infer_discovery_constraints,
    plan_asset_placements,
    render_prompt,
)
from .state import _write_json_atomic, stable_hash
from .versioning import semantic_key


class CompletionClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        json_schema_name: str | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> str: ...


class AgentRunError(RuntimeError):
    pass


DISCOVERY_CHUNK_CHARS = 48_000
ASSET_PLACEMENT_BATCH_SIZE = 6
ASSET_CANDIDATE_GROUP_LIMIT = 16


@dataclass(frozen=True)
class DocumentProfile:
    source: str
    title: str
    character_count: int
    evidence_count: int
    heading_count: int
    structured_section_count: int
    heading_outline: tuple[str, ...]
    section_previews: tuple[str, ...]
    asset_count: int
    asset_kinds: tuple[str, ...]
    asset_contexts: tuple[str, ...]
    document_family_id: str = ""
    document_version_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentRefRecord:
    ref_id: str
    article_id: str
    local_id: str
    type: str
    title: str
    description: str
    evidence: tuple[str, ...]
    asset_hints: tuple[str, ...]
    source: str
    section_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    evidence_block_ids: tuple[str, ...] = ()
    semantic_signature: Mapping[str, Any] = field(default_factory=dict)
    scope: Mapping[str, Any] = field(default_factory=dict)
    ref_family_hint: str = ""
    ref_version_id: str = ""
    document_family_id: str = ""
    document_version_id: str = ""

    def candidate_payload(self) -> dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "article_id": self.article_id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "section_path": list(self.section_path),
            "page_start": self.page_start,
            "page_end": self.page_end,
            "semantic_signature": dict(self.semantic_signature),
            "scope": dict(self.scope),
            "ref_family_hint": self.ref_family_hint,
            "ref_version_id": self.ref_version_id,
            "document_family_id": self.document_family_id,
            "document_version_id": self.document_version_id,
        }


@dataclass(frozen=True)
class AgentRunSummary:
    status: str
    output_dir: Path
    articles: int
    refs: int
    groups: int
    concepts: int
    reviews: int
    recompiles: int


@dataclass(frozen=True)
class _SourceRecord:
    path: Path
    content: str
    assets: tuple[SourceAsset, ...]
    profile: DocumentProfile
    plan: SourcePlan
    refs: tuple[AgentRefRecord, ...]
    structure: Mapping[str, Any] | None = None


class AgentCompiler:
    """Run an isolated AgentWiki compilation without touching releases."""

    def __init__(
        self,
        settings: Settings,
        llm: CompletionClient,
        *,
        policy: AgentPolicy | None = None,
        on_event: Callable[[str], None] | None = None,
        compile_workers: int = 1,
    ):
        if not 1 <= compile_workers <= 4:
            raise ValueError("compile_workers must be between 1 and 4")
        self.settings = settings
        self.llm = llm
        self.policy = policy or AgentPolicy()
        self.on_event = on_event or (lambda _message: None)
        self.compile_workers = compile_workers
        self.trace: list[dict[str, Any]] = []
        self.ref_rejections: list[dict[str, Any]] = []
        self._trace_lock = Lock()

    def run(self, output_dir: Path, *, resume: bool = False) -> AgentRunSummary:
        output = self._guard_output(output_dir, resume=resume)
        if resume:
            trace_path = output / "agent_trace.json"
            if trace_path.exists():
                self.trace = list(
                    json.loads(trace_path.read_text(encoding="utf-8")).get(
                        "events", []
                    )
                )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("status") not in {"running", "failed"}:
                raise AgentRunError(
                    "only a running or failed Agent run can be resumed"
                )
            if manifest.get("policy") != asdict(self.policy):
                raise AgentRunError(
                    "resume policy differs from the original Agent run"
                )
        else:
            self.trace = []
            output.mkdir(parents=True, exist_ok=False)
        self.ref_rejections = []
        _write_json_atomic(
            output / "manifest.json",
            {
                "version": 1,
                "status": "running",
                "schema": "kmpro.agent-run.v2",
                "model": self.settings.openai_model,
                "resumed": resume,
                "policy": asdict(self.policy),
            },
        )
        try:
            templates = self._load_templates()
            sources = self._plan_and_discover(output, templates, resume=resume)
            _write_json_atomic(
                output / "ref_validation.json",
                {
                    "raw_refs": (
                        sum(len(source.refs) for source in sources)
                        + len(self.ref_rejections)
                    ),
                    "accepted_refs": sum(len(source.refs) for source in sources),
                    "rejected_refs": self.ref_rejections,
                },
            )
            refs = tuple(ref for source in sources for ref in source.refs)
            candidates, states = candidate_edges(
                [ref.candidate_payload() for ref in refs]
            )
            _write_json_atomic(
                output / "candidates.json",
                {
                    "algorithm": "lexical-v2",
                    "edges": [edge.as_dict() for edge in candidates],
                    "states": states,
                },
            )
            group_input_hash = stable_hash(
                {
                    "refs": [asdict(ref) for ref in refs],
                    "candidates": [edge.as_dict() for edge in candidates],
                    "max_component_refs": self.policy.max_component_refs,
                }
            )
            groups_path = output / "groups.json"
            cached_groups = (
                json.loads(groups_path.read_text(encoding="utf-8"))
                if resume and groups_path.exists()
                else {}
            )
            if cached_groups.get("input_hash") == group_input_hash:
                groups = tuple(
                    CompileGroup(
                        group_id=item["group_id"],
                        ref_ids=tuple(item["ref_ids"]),
                        title=item["title"],
                        description=item["description"],
                        reason=item["reason"],
                    )
                    for item in cached_groups.get("groups", [])
                )
                self._record(
                    {
                        "stage": "resume",
                        "reused": "compile_groups",
                        "count": len(groups),
                    }
                )
            else:
                groups = plan_compile_groups(
                    self.llm,
                    templates["agent_group"],
                    refs,
                    tuple(candidates),
                    max_component_refs=self.policy.max_component_refs,
                    on_decision=self._record,
                )
            _write_json_atomic(
                groups_path,
                {
                    "input_hash": group_input_hash,
                    "groups": [asdict(group) for group in groups],
                },
            )
            drafts, quality, reviews, recompile_count = self._compile_and_audit(
                output, templates, refs, groups, resume=resume
            )
            drafts, asset_reviews, withheld = self._place_assets(
                output,
                templates,
                sources,
                groups,
                drafts,
                resume=resume,
            )
            reviews.extend(asset_reviews)
            self._publish(
                output,
                refs,
                groups,
                drafts,
                quality,
                reviews,
                withheld,
            )
            status = "complete" if not reviews else "needs_review"
            summary = AgentRunSummary(
                status=status,
                output_dir=output,
                articles=len(sources),
                refs=len(refs),
                groups=len(groups),
                concepts=len(drafts) - len(withheld),
                reviews=len(reviews),
                recompiles=recompile_count,
            )
            self._write_trace(output)
            _write_json_atomic(
                output / "manifest.json",
                {
                    "version": 1,
                    "status": status,
                    "schema": "kmpro.agent-run.v2",
                    "model": self.settings.openai_model,
                    "policy": asdict(self.policy),
                    **{
                        key: value
                        for key, value in asdict(summary).items()
                        if key != "output_dir"
                    },
                },
            )
            self.on_event(
                "agent.done "
                f"articles={summary.articles} refs={summary.refs} "
                f"groups={summary.groups} concepts={summary.concepts} "
                f"reviews={summary.reviews} recompiles={summary.recompiles}"
            )
            return summary
        except Exception as error:
            self._write_trace(output)
            _write_json_atomic(
                output / "manifest.json",
                {
                    "version": 1,
                    "status": "failed",
                    "schema": "kmpro.agent-run.v2",
                    "model": self.settings.openai_model,
                    "policy": asdict(self.policy),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise

    def _plan_and_discover(
        self,
        output: Path,
        templates: Mapping[str, str],
        *,
        resume: bool,
    ) -> tuple[_SourceRecord, ...]:
        records: list[_SourceRecord] = []
        progress_path = output / "source_progress.json"
        cached_sources = {}
        if resume and progress_path.exists():
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            cached_sources = {
                item["source"]: item for item in payload.get("sources", [])
            }
        sources_dir = self.settings.sources_dir
        images_dir = sources_dir / "images"
        source_paths = sorted(sources_dir.glob("*.md"), key=lambda item: item.name)
        if not source_paths:
            raise AgentRunError("no Markdown sources found")
        for position, path in enumerate(source_paths, start=1):
            content = strip_missing_image_references(
                path.read_text(encoding="utf-8"), images_dir
            )
            structure = _load_source_structure(path)
            assets = inventory_assets(content, images_dir)
            metadata = _load_article_metadata(path, content)
            if structure is not None:
                metadata["toc_entries"] = [
                    {
                        "title": item.get("title"),
                        "level": item.get("level"),
                        "page_number": item.get("page_number"),
                    }
                    for item in structure.get("toc_entries", [])
                    if isinstance(item, dict) and item.get("title")
                ]
                metadata["outline_entries"] = [
                    item.get("title")
                    for item in structure.get("outline", [])
                    if isinstance(item, dict) and item.get("title")
                ]
            profile = profile_document(path.name, content, assets, metadata=metadata)
            source_hash = stable_hash(
                {
                    "content": content,
                    "structure": structure,
                    "assets": [asdict(item) for item in assets],
                    "agent_prompts": {
                        name: templates[name]
                        for name in ("agent_plan", "agent_refine", "discover")
                    },
                    "model": self.settings.openai_model,
                }
            )
            cached = cached_sources.get(path.name)
            if cached is not None and cached.get("source_hash") == source_hash:
                record = _source_record_from_payload(
                    path,
                    content,
                    assets,
                    cached,
                    structure=structure,
                )
                record = replace(
                    record,
                    refs=tuple(
                        _attach_agent_ref_provenance(ref, structure)
                        for ref in record.refs
                    ),
                )
                record = replace(
                    record,
                    refs=self._validate_source_refs(
                        path.name, content, record.refs
                    ),
                )
                records.append(record)
                self._record(
                    {
                        "stage": "resume",
                        "source": path.name,
                        "reused": "plan_and_discovery",
                        "ref_count": len(record.refs),
                    }
                )
                continue
            self.on_event(
                f"agent.plan.start source={path.name} position={position}/{len(source_paths)}"
            )
            plan = plan_source(
                self.llm,
                templates["agent_plan"],
                profile,
                on_decision=self._record,
            )
            if plan.discovery_mode == "llm":
                refs = discover_agent_concepts(
                    self.llm,
                    templates["discover"],
                    title=profile.title,
                    source_name=path.name,
                    source_content=content,
                    assets=assets,
                    on_decision=self._record,
                )
            else:
                refs = discover_from_headings(path.name, content, assets)
            if plan.refine_discovery:
                refs = refine_discovery(
                    self.llm,
                    templates["agent_refine"],
                    source_name=path.name,
                    source_content=content,
                    assets=assets,
                    refs=refs,
                    on_decision=self._record,
                )
            article_id = _stable_id("article", path.name)
            agent_refs = tuple(
                _attach_agent_ref_provenance(
                    AgentRefRecord(
                        ref_id=_stable_id(
                            "ref", f"{path.name}\0{ref.concept_id}"
                        ),
                        article_id=article_id,
                        local_id=ref.concept_id,
                        type=ref.type,
                        title=ref.title,
                        description=ref.description,
                        evidence=ref.evidence,
                        asset_hints=ref.asset_hints,
                        source=path.name,
                        semantic_signature=dict(ref.semantic_signature),
                        scope=dict(ref.scope),
                        ref_family_hint=ref.ref_family_hint or semantic_key(ref),
                        ref_version_id=profile.document_version_id,
                        document_family_id=profile.document_family_id,
                        document_version_id=profile.document_version_id,
                    ),
                    structure,
                )
                for ref in refs
            )
            agent_refs = self._validate_source_refs(
                path.name, content, agent_refs
            )
            record = _SourceRecord(
                path,
                content,
                assets,
                profile,
                plan,
                agent_refs,
                structure,
            )
            records.append(record)
            self._record(
                {
                    "stage": "source_complete",
                    "source": path.name,
                    "plan": asdict(plan),
                    "ref_count": len(agent_refs),
                }
            )
            _write_json_atomic(
                output / "source_progress.json",
                {
                    "sources": [
                        {
                            "source": item.path.name,
                            "source_hash": stable_hash(
                                {
                                    "content": item.content,
                                    "structure": item.structure,
                                    "assets": [
                                        asdict(asset) for asset in item.assets
                                    ],
                                    "agent_prompts": {
                                        name: templates[name]
                                        for name in (
                                            "agent_plan",
                                            "agent_refine",
                                            "discover",
                                        )
                                    },
                                    "model": self.settings.openai_model,
                                }
                            ),
                            "profile": asdict(item.profile),
                            "plan": asdict(item.plan),
                            "refs": [asdict(ref) for ref in item.refs],
                        }
                        for item in records
                    ]
                },
            )
            self.on_event(
                f"agent.discovery.done source={path.name} refs={len(agent_refs)} "
                f"mode={plan.discovery_mode} refine={str(plan.refine_discovery).lower()}"
            )
        return tuple(records)

    def _validate_source_refs(
        self,
        source_name: str,
        source_content: str,
        refs: tuple[AgentRefRecord, ...],
    ) -> tuple[AgentRefRecord, ...]:
        container_titles = _container_heading_titles(source_content)
        accepted = tuple(
            ref
            for ref in refs
            if _has_substantive_evidence(ref.evidence)
            and ref.title not in container_titles
        )
        rejected = tuple(ref for ref in refs if ref not in accepted)
        if rejected:
            for ref in rejected:
                self.ref_rejections.append(
                    {
                        "source": source_name,
                        "ref_id": ref.ref_id,
                        "local_id": ref.local_id,
                        "title": ref.title,
                        "reason": (
                            "logical_container_heading"
                            if ref.title in container_titles
                            else "heading_container_without_substantive_evidence"
                        ),
                    }
                )
            self._record(
                {
                    "stage": "ref_validation",
                    "source": source_name,
                    "accepted": len(accepted),
                    "rejected": len(rejected),
                    "rejected_ref_ids": [ref.ref_id for ref in rejected],
                }
            )
        if not accepted:
            raise AgentRunError(
                f"source has no substantive ConceptRef after validation: {source_name}"
            )
        return accepted

    def _compile_and_audit(
        self,
        output: Path,
        templates: Mapping[str, str],
        refs: tuple[AgentRefRecord, ...],
        groups: tuple[CompileGroup, ...],
        *,
        resume: bool,
    ) -> tuple[
        dict[str, DraftConcept],
        dict[str, list[QualityAudit]],
        list[dict[str, Any]],
        int,
    ]:
        by_id = {ref.ref_id: ref for ref in refs}
        progress_path = output / "compile_progress.json"
        compile_input_hash = stable_hash(
            {
                "refs": [asdict(ref) for ref in refs],
                "groups": [asdict(group) for group in groups],
                "prompts": {
                    name: templates[name]
                    for name in (
                        "compile",
                        "agent_quality",
                        "agent_recompile",
                    )
                },
                "model": self.settings.openai_model,
                "policy": asdict(self.policy),
            }
        )
        loaded_progress = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if resume and progress_path.exists()
            else {}
        )
        group_hashes = {
            group.group_id: _compile_group_cache_hash(
                group,
                by_id,
                templates,
                self.settings.openai_model,
                self.policy,
            )
            for group in groups
        }
        if loaded_progress.get("input_hash") == compile_input_hash:
            progress = loaded_progress
        else:
            loaded_hashes = loaded_progress.get("group_hashes", {})
            reusable = {
                group_id
                for group_id, group_hash in group_hashes.items()
                if loaded_hashes.get(group_id) == group_hash
                and group_id in loaded_progress.get("drafts", {})
                and group_id in loaded_progress.get("quality", {})
            }
            reusable_quality = {
                group_id: loaded_progress["quality"][group_id]
                for group_id in reusable
            }
            progress = {
                "drafts": {
                    group_id: loaded_progress["drafts"][group_id]
                    for group_id in reusable
                },
                "quality": reusable_quality,
                "reviews": [
                    item
                    for item in loaded_progress.get("reviews", [])
                    if item.get("group_id") in reusable
                ],
                "recompiles": sum(
                    max(0, len(values) - 1)
                    for values in reusable_quality.values()
                ),
            }
        drafts = {
            group_id: _draft_from_payload(payload)
            for group_id, payload in progress.get("drafts", {}).items()
        }
        audits = {
            group_id: [
                QualityAudit(
                    score=float(item["score"]),
                    decision=item["decision"],
                    issues=tuple(item["issues"]),
                    recompile_instructions=item["recompile_instructions"],
                )
                for item in values
            ]
            for group_id, values in progress.get("quality", {}).items()
        }
        reviews = list(progress.get("reviews", []))
        recompiles = int(progress.get("recompiles", 0))
        pending: list[tuple[int, CompileGroup]] = []
        for position, group in enumerate(groups, start=1):
            if group.group_id in drafts and group.group_id in audits:
                self._record(
                    {
                        "stage": "resume",
                        "group_id": group.group_id,
                        "reused": "compile_and_quality",
                    }
                )
                continue
            pending.append((position, group))

        def save_progress() -> None:
            _write_json_atomic(
                progress_path,
                {
                    "input_hash": compile_input_hash,
                    "completed_groups": sorted(drafts),
                    "drafts": {
                        group_id: _draft_payload(item)
                        for group_id, item in drafts.items()
                    },
                    "quality": {
                        group_id: [asdict(item) for item in values]
                        for group_id, values in audits.items()
                    },
                    "reviews": reviews,
                    "recompiles": recompiles,
                    "group_hashes": {
                        group_id: group_hashes[group_id]
                        for group_id in drafts
                    },
                },
            )

        def accept_result(
            group: CompileGroup,
            result: tuple[
                DraftConcept,
                list[QualityAudit],
                list[dict[str, Any]],
                int,
            ],
        ) -> None:
            nonlocal recompiles
            draft, group_audits, group_reviews, group_recompiles = result
            drafts[group.group_id] = draft
            audits[group.group_id] = group_audits
            reviews.extend(group_reviews)
            recompiles += group_recompiles
            save_progress()

        if self.compile_workers == 1:
            for position, group in pending:
                accept_result(
                    group,
                    self._compile_one_group(
                        templates, by_id, group, position, len(groups)
                    ),
                )
        else:
            with ThreadPoolExecutor(
                max_workers=self.compile_workers,
                thread_name_prefix="agent-compile",
            ) as pool:
                futures = {
                    pool.submit(
                        self._compile_one_group,
                        templates,
                        by_id,
                        group,
                        position,
                        len(groups),
                    ): group
                    for position, group in pending
                }
                for future in as_completed(futures):
                    accept_result(futures[future], future.result())
        return drafts, audits, reviews, recompiles

    def _compile_one_group(
        self,
        templates: Mapping[str, str],
        by_id: Mapping[str, AgentRefRecord],
        group: CompileGroup,
        position: int,
        total: int,
    ) -> tuple[
        DraftConcept,
        list[QualityAudit],
        list[dict[str, Any]],
        int,
    ]:
        synthetic = _group_ref(group, by_id)
        self.on_event(
            f"agent.compile.start group={group.group_id} "
            f"position={position}/{total} refs={len(group.ref_ids)}"
        )
        draft = compile_one_concept(self.llm, templates["compile"], synthetic)
        group_audits: list[QualityAudit] = []
        reviews: list[dict[str, Any]] = []
        recompiles = 0
        for attempt in range(self.policy.max_recompile_attempts + 1):
            audit = audit_concept_quality(
                self.llm,
                templates["agent_quality"],
                draft=draft,
                refs=tuple(by_id[ref_id] for ref_id in group.ref_ids),
                threshold=self.policy.quality_threshold,
                on_decision=self._record,
            )
            group_audits.append(audit)
            if audit.decision == "pass":
                break
            if audit.decision == "human_review":
                reviews.append(
                    {
                        "kind": "concept_quality",
                        "group_id": group.group_id,
                        "score": audit.score,
                        "issues": list(audit.issues),
                        "reason": "quality_agent_requested_human_review",
                    }
                )
                break
            if attempt >= self.policy.max_recompile_attempts:
                reviews.append(
                    {
                        "kind": "concept_quality",
                        "group_id": group.group_id,
                        "score": audit.score,
                        "issues": list(audit.issues),
                        "reason": "recompile_budget_exhausted",
                    }
                )
                break
            draft = recompile_concept(
                self.llm,
                templates["agent_recompile"],
                ref=synthetic,
                previous=draft,
                audit=audit,
            )
            recompiles += 1
            self._record(
                {
                    "stage": "recompile",
                    "group_id": group.group_id,
                    "attempt": attempt + 1,
                    "issues": list(audit.issues),
                    "instructions": audit.recompile_instructions,
                }
            )
        return draft, group_audits, reviews, recompiles

    def _place_assets(
        self,
        output: Path,
        templates: Mapping[str, str],
        sources: tuple[_SourceRecord, ...],
        groups: tuple[CompileGroup, ...],
        drafts: dict[str, DraftConcept],
        *,
        resume: bool,
    ) -> tuple[dict[str, DraftConcept], list[dict[str, Any]], set[str]]:
        group_by_ref = {
            ref_id: group.group_id
            for group in groups
            for ref_id in group.ref_ids
        }
        progress_path = output / "asset_progress.json"
        input_hash = stable_hash(
            {
                "sources": [
                    {
                        "source": source.path.name,
                        "assets": [asdict(item) for item in source.assets],
                        "plan": asdict(source.plan),
                        "ref_ids": [ref.ref_id for ref in source.refs],
                    }
                    for source in sources
                ],
                "groups": [asdict(group) for group in groups],
                "drafts": {
                    group_id: _draft_payload(item)
                    for group_id, item in drafts.items()
                },
                "prompt": templates["preserve"],
                "model": self.settings.openai_model,
            }
        )
        loaded = (
            json.loads(progress_path.read_text(encoding="utf-8"))
            if resume and progress_path.exists()
            else {}
        )
        progress = loaded if loaded.get("input_hash") == input_hash else {}
        if progress:
            drafts = {
                group_id: _draft_from_payload(payload)
                for group_id, payload in progress.get("drafts", {}).items()
            }
        reviews: list[dict[str, Any]] = list(progress.get("reviews", []))
        withheld: set[str] = set(progress.get("withheld", []))
        processed: set[str] = set(progress.get("processed_sources", []))
        processed_asset_batches: dict[str, int] = {
            str(source): int(count)
            for source, count in progress.get(
                "processed_asset_batches", {}
            ).items()
        }

        def save_progress() -> None:
            _write_json_atomic(
                progress_path,
                {
                    "input_hash": input_hash,
                    "processed_sources": sorted(processed),
                    "processed_asset_batches": processed_asset_batches,
                    "drafts": {
                        group_id: _draft_payload(item)
                        for group_id, item in drafts.items()
                    },
                    "reviews": reviews,
                    "withheld": sorted(withheld),
                },
            )

        for source in sources:
            if not source.assets:
                continue
            if source.path.name in processed:
                self._record(
                    {
                        "stage": "resume",
                        "source": source.path.name,
                        "reused": "asset_placement",
                    }
                )
                continue
            target_groups = sorted(
                {
                    group_by_ref[ref.ref_id]
                    for ref in source.refs
                    if ref.ref_id in group_by_ref
                }
            )
            if source.plan.asset_policy == "human_review":
                withheld.update(target_groups)
                for asset in source.assets:
                    reviews.append(
                        {
                            "kind": "asset_placement",
                            "source": source.path.name,
                            "asset_id": asset.asset_id,
                            "asset_kind": asset.kind,
                            "candidate_group_ids": target_groups,
                            "before": asset.before,
                            "after": asset.after,
                            "reason": source.plan.reason,
                        }
                    )
                processed.add(source.path.name)
                save_progress()
                continue
            missing_groups = [
                group_id for group_id in target_groups if group_id not in drafts
            ]
            if missing_groups:
                withheld.update(target_groups)
                reviews.append(
                    {
                        "kind": "asset_placement",
                        "source": source.path.name,
                        "candidate_group_ids": target_groups,
                        "reason": "a candidate Concept is unavailable for automatic placement",
                    }
                )
                processed.add(source.path.name)
                save_progress()
                continue
            asset_batches = tuple(
                source.assets[start : start + ASSET_PLACEMENT_BATCH_SIZE]
                for start in range(
                    0, len(source.assets), ASSET_PLACEMENT_BATCH_SIZE
                )
            )
            for batch_no, asset_batch in enumerate(asset_batches, start=1):
                if batch_no <= processed_asset_batches.get(
                    source.path.name, 0
                ):
                    continue
                candidate_ids = _asset_candidate_group_ids(
                    asset_batch,
                    source.refs,
                    group_by_ref,
                    drafts,
                    tuple(target_groups),
                    limit=ASSET_CANDIDATE_GROUP_LIMIT,
                )
                candidate_drafts = tuple(
                    drafts[group_id] for group_id in candidate_ids
                )
                placements = plan_asset_placements(
                    self.llm,
                    templates["preserve"],
                    assets=asset_batch,
                    drafts=candidate_drafts,
                )
                documents = apply_asset_placements(
                    candidate_drafts, asset_batch, placements
                )
                validate_asset_preservation(
                    asset_batch,
                    documents,
                    self.settings.sources_dir / "images",
                    baseline=candidate_drafts,
                )
                for document in documents:
                    group_id = document.filename[:-3]
                    previous = drafts[group_id]
                    drafts[group_id] = replace(previous, body=document.body)
                self._record(
                    {
                        "stage": "asset_placement",
                        "source": source.path.name,
                        "policy": "auto",
                        "batch": batch_no,
                        "batches": len(asset_batches),
                        "candidate_group_ids": list(candidate_ids),
                        "placements": [
                            {
                                "asset_id": item.asset_id,
                                "concept_id": item.concept_id,
                                "anchor": item.anchor,
                                "position": item.position,
                                "reason": item.reason,
                            }
                            for item in placements
                        ],
                    }
                )
                processed_asset_batches[source.path.name] = batch_no
                save_progress()
            processed.add(source.path.name)
            save_progress()
        self._copy_referenced_images(output, sources)
        return drafts, reviews, withheld

    def _copy_referenced_images(
        self, output: Path, sources: tuple[_SourceRecord, ...]
    ) -> None:
        source_root = self.settings.sources_dir / "images"
        target_root = output / "images"
        copied: set[Path] = set()
        for source in sources:
            for asset in source.assets:
                if asset.kind != "image" or asset.target is None:
                    continue
                if not asset.target.startswith("images/"):
                    continue
                relative = PurePosixPath(asset.target).relative_to("images")
                if ".." in relative.parts:
                    raise AgentRunError(
                        f"unsafe image path in Agent run: {asset.target}"
                    )
                relative_path = Path(*relative.parts)
                if relative_path in copied:
                    continue
                source_path = source_root / relative_path
                target_path = target_root / relative_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                copied.add(relative_path)

    def _publish(
        self,
        output: Path,
        refs: tuple[AgentRefRecord, ...],
        groups: tuple[CompileGroup, ...],
        drafts: Mapping[str, DraftConcept],
        quality: Mapping[str, list[QualityAudit]],
        reviews: list[dict[str, Any]],
        withheld: set[str],
    ) -> None:
        by_ref = {ref.ref_id: ref for ref in refs}
        review_groups = {
            str(item["group_id"])
            for item in reviews
            if item.get("kind") == "concept_quality" and "group_id" in item
        }
        withheld.update(review_groups)
        concepts_dir = output / "concepts"
        drafts_dir = output / "drafts"
        for directory in (concepts_dir, drafts_dir):
            if directory.exists():
                shutil.rmtree(directory)
        concepts_dir.mkdir(parents=True, exist_ok=True)
        drafts_dir.mkdir(parents=True, exist_ok=True)
        concept_payloads: list[dict[str, Any]] = []
        for group in groups:
            draft = drafts[group.group_id]
            articles = sorted(
                {by_ref[ref_id].article_id for ref_id in group.ref_ids}
            )
            sources = sorted({by_ref[ref_id].source for ref_id in group.ref_ids})
            audit = quality[group.group_id][-1]
            frontmatter = {
                "type": draft.ref.type,
                "title": draft.title,
                "description": draft.description,
                "source": (
                    sources[0] if len(sources) == 1 else "多来源联合编译"
                ),
                "concept_refs": list(group.ref_ids),
                "articles": articles,
                "ref_families": sorted(
                    {
                        by_ref[ref_id].ref_family_hint
                        for ref_id in group.ref_ids
                        if by_ref[ref_id].ref_family_hint
                    }
                ),
                "scopes": [
                    dict(by_ref[ref_id].scope)
                    for ref_id in group.ref_ids
                    if by_ref[ref_id].scope
                ],
                "source_locations": [
                    {
                        "ref_id": ref_id,
                        "article_id": by_ref[ref_id].article_id,
                        "source": by_ref[ref_id].source,
                        "section_path": list(by_ref[ref_id].section_path),
                        "page_start": by_ref[ref_id].page_start,
                        "page_end": by_ref[ref_id].page_end,
                        "document_family_id": by_ref[ref_id].document_family_id,
                        "document_version_id": by_ref[ref_id].document_version_id,
                        "ref_family_hint": by_ref[ref_id].ref_family_hint,
                        "semantic_signature": dict(by_ref[ref_id].semantic_signature),
                        "scope": dict(by_ref[ref_id].scope),
                    }
                    for ref_id in group.ref_ids
                ],
                "agent_quality_score": audit.score,
            }
            document = ConceptDocument(
                filename=f"{group.group_id}.md",
                frontmatter=frontmatter,
                body=draft.body,
            )
            target = (
                drafts_dir / document.filename
                if group.group_id in withheld
                else concepts_dir / document.filename
            )
            target.write_text(
                rewrite_image_paths(document.render()), encoding="utf-8"
            )
            concept_payloads.append(
                {
                    **asdict(group),
                    "articles": articles,
                    "sources": sources,
                    "quality": [asdict(item) for item in quality[group.group_id]],
                    "status": (
                        "needs_review"
                        if group.group_id in withheld
                        else "publishable"
                    ),
                }
            )
        _write_json_atomic(
            output / "refs.json", {"refs": [asdict(ref) for ref in refs]}
        )
        _write_json_atomic(output / "concepts.json", {"concepts": concept_payloads})
        _write_json_atomic(output / "review_queue.json", {"reviews": reviews})

    def _guard_output(self, output_dir: Path, *, resume: bool) -> Path:
        root = (self.settings.data_dir / "agent-runs").resolve()
        output = output_dir.resolve()
        try:
            output.relative_to(root)
        except ValueError as error:
            raise AgentRunError(
                f"Agent output must stay under {root}; got {output}"
            ) from error
        if output == root:
            raise AgentRunError("Agent output requires a run-specific subdirectory")
        if output.exists() and not resume:
            raise AgentRunError(f"Agent output already exists: {output}")
        if resume and not (output / "manifest.json").is_file():
            raise AgentRunError(
                f"Agent run cannot resume without manifest.json: {output}"
            )
        return output

    def _load_templates(self) -> dict[str, str]:
        names = (
            "discover",
            "compile",
            "preserve",
            "agent_plan",
            "agent_refine",
            "agent_group",
            "agent_quality",
            "agent_recompile",
        )
        return {
            name: (self.settings.prompts_dir / f"{name}.md").read_text(
                encoding="utf-8"
            )
            for name in names
        }

    def _record(self, item: dict[str, Any]) -> None:
        with self._trace_lock:
            self.trace.append(item)

    def _write_trace(self, output: Path) -> None:
        _write_json_atomic(output / "agent_trace.json", {"events": self.trace})


def profile_document(
    source_name: str,
    source_content: str,
    assets: tuple[SourceAsset, ...],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> DocumentProfile:
    headings = _heading_sections(source_content)
    structured = _select_heading_level(headings)
    metadata_payload = dict(metadata or {})
    family_id = str(metadata_payload.get("document_family_id") or "").strip()
    if not family_id:
        family_id = stable_hash(
            {"source_file": source_name, "title": _extract_title(source_name, source_content)}
        )[:20]
    version_id = str(metadata_payload.get("document_version_id") or "").strip()
    if not version_id:
        version_id = stable_hash(
            {"source_file": source_name, "content": source_content}
        )[:20]
    metadata_payload.update(
        {
            "document_family_id": family_id,
            "document_version_id": version_id,
        }
    )
    return DocumentProfile(
        source=source_name,
        title=_extract_title(source_name, source_content),
        character_count=len(source_content),
        evidence_count=len(build_evidence_catalog(source_content)),
        heading_count=len(headings),
        structured_section_count=len(structured),
        heading_outline=tuple(item[1] for item in headings[:120]),
        section_previews=tuple(
            f"{title}: {_section_description(source_content[start:end], title)}"
            for _depth, title, start, end in structured[:80]
        ),
        asset_count=len(assets),
        asset_kinds=tuple(sorted({item.kind for item in assets})),
        asset_contexts=tuple(
            f"{item.asset_id} ({item.kind}) 前文：{item.before[:160]} "
            f"后文：{item.after[:160]}"
            for item in assets[:80]
        ),
        document_family_id=family_id,
        document_version_id=version_id,
        metadata=metadata_payload,
    )


def _load_article_metadata(path: Path, content: str) -> dict[str, Any]:
    """Load optional YAML frontmatter without making it a hard requirement."""
    match = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|$)", content, flags=re.DOTALL)
    if not match:
        return {}
    try:
        import yaml

        value = yaml.safe_load(match.group(1))
    except Exception:
        value = None
    if not isinstance(value, dict):
        return {}
    metadata = {str(key): item for key, item in value.items()}
    metadata.setdefault("source_file", path.name)
    return metadata


def plan_source(
    llm: CompletionClient,
    template: str,
    profile: DocumentProfile,
    *,
    on_decision: Callable[[dict[str, Any]], None] | None = None,
) -> SourcePlan:
    prompt = render_prompt(
        template,
        document_profile=_json(asdict(profile)),
    )
    plan = _complete_agent(
        llm,
        prompt,
        schema_name="agent_source_plan",
        schema=source_plan_schema(),
        parser=lambda response: parse_source_plan(
            response,
            structured_section_count=profile.structured_section_count,
            asset_count=profile.asset_count,
        ),
    )
    if on_decision is not None:
        on_decision(
            {
                "stage": "source_plan",
                "source": profile.source,
                "decision": asdict(plan),
            }
        )
    return plan


def discover_from_headings(
    source_name: str,
    source_content: str,
    assets: tuple[SourceAsset, ...],
) -> tuple[ConceptRef, ...]:
    headings = _heading_sections(source_content)
    selected = _select_heading_level(headings)
    if len(selected) < 2:
        raise ContractError(
            "heading discovery requires at least two substantive same-level sections"
        )
    catalog = build_evidence_catalog(source_content)
    container_titles = _container_heading_titles(source_content)
    refs: list[ConceptRef] = []
    used_ids: set[str] = set()
    for position, (depth, title, start, end) in enumerate(selected, start=1):
        block = source_content[start:end].strip()
        cleaned_title = _clean_heading(title)
        if (
            not _has_substantive_markdown(block)
            or cleaned_title in container_titles
        ):
            continue
        evidence = tuple(value for value in catalog.values() if value in block)
        if not evidence:
            evidence = (block,)
        concept_id = normalize_slug(f"{cleaned_title}.md")[:-3]
        if concept_id in used_ids:
            concept_id = f"{concept_id}-{position}"
        used_ids.add(concept_id)
        refs.append(
            ConceptRef(
                concept_id=concept_id,
                type=_infer_type(title),
                title=cleaned_title,
                description=_section_description(block, title),
                source=source_name,
                evidence=evidence,
                asset_hints=tuple(
                    item.asset_id for item in assets if item.raw in block
                ),
            )
        )
    if len(refs) < 2:
        raise ContractError(
            "heading discovery requires at least two substantive same-level sections"
        )
    return tuple(refs)


def discover_agent_concepts(
    llm: CompletionClient,
    template: str,
    *,
    title: str,
    source_name: str,
    source_content: str,
    assets: tuple[SourceAsset, ...],
    on_decision: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[ConceptRef, ...]:
    evidence_catalog = build_evidence_catalog(source_content)
    corrections: list[dict[str, str]] = []
    chunks = _partition_evidence_catalog(
        evidence_catalog,
        maximum_chars=DISCOVERY_CHUNK_CHARS,
    )
    discovered: list[ConceptRef] = []
    for chunk_no, chunk in enumerate(chunks, start=1):
        chunk_content = "\n\n".join(chunk.values())
        constraints = infer_discovery_constraints(chunk_content)
        min_concepts = max(
            constraints.min_concepts,
            2 if len(chunk) >= 8 else 1,
        )
        chunk_assets = _assets_for_evidence(assets, chunk)
        discovered.extend(
            _discover_catalog(
                llm,
                template,
                title=title,
                source_name=source_name,
                catalog=chunk,
                assets=chunk_assets,
                min_concepts=min_concepts,
                required_types=set(constraints.required_types),
                outline=constraints.outline,
                corrections=corrections,
            )
        )
        if on_decision is not None and len(chunks) > 1:
            on_decision(
                {
                    "stage": "llm_discovery_chunk",
                    "source": source_name,
                    "chunk": chunk_no,
                    "chunks": len(chunks),
                    "evidence_count": len(chunk),
                    "ref_count": len(discovered),
                }
            )
    refs = _merge_discovered_refs(discovered)
    if not refs:
        raise ContractError("chunked discovery returned no ConceptRef")
    if on_decision is not None:
        on_decision(
            {
                "stage": "llm_discovery",
                "source": source_name,
                "chunks": len(chunks),
                "ref_count": len(refs),
                "id_corrections": corrections,
            }
        )
    return refs


def _discover_catalog(
    llm: CompletionClient,
    template: str,
    *,
    title: str,
    source_name: str,
    catalog: dict[str, str],
    assets: tuple[SourceAsset, ...],
    min_concepts: int,
    required_types: set[str],
    outline: tuple[str, ...],
    corrections: list[dict[str, str]],
) -> tuple[ConceptRef, ...]:
    prompt = render_prompt(
        template,
        title=title,
        source_file=source_name,
        evidence_catalog=_json(
            [
                {"evidence_id": evidence_id, "text": text}
                for evidence_id, text in catalog.items()
            ]
        ),
        minimum_concepts=str(min_concepts),
        required_types=_json(sorted(required_types)),
        source_outline=_json(list(outline)),
        asset_inventory=_json([asdict(item) for item in assets]),
    )
    return _complete_agent(
        llm,
        prompt,
        schema_name="concept_discovery",
        schema=discovery_json_schema(min_concepts),
        parser=lambda response: _parse_agent_discovery(
            response,
            source_name=source_name,
            evidence_catalog=catalog,
            asset_ids={item.asset_id for item in assets},
            min_concepts=min_concepts,
            required_types=required_types,
            corrections=corrections,
        ),
    )


def refine_discovery(
    llm: CompletionClient,
    template: str,
    *,
    source_name: str,
    source_content: str,
    assets: tuple[SourceAsset, ...],
    refs: tuple[ConceptRef, ...],
    on_decision: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[ConceptRef, ...]:
    catalog = build_evidence_catalog(source_content)
    values = set(catalog.values())
    synthetic_no = 1
    for ref in refs:
        for evidence in ref.evidence:
            if evidence in values:
                continue
            catalog[f"agent-evidence-{synthetic_no:04d}"] = evidence
            values.add(evidence)
            synthetic_no += 1
    corrections: list[dict[str, str]] = []
    chunks = _partition_evidence_catalog(
        catalog,
        maximum_chars=DISCOVERY_CHUNK_CHARS,
    )
    refined_refs: list[ConceptRef] = []
    for chunk_no, chunk in enumerate(chunks, start=1):
        chunk_values = set(chunk.values())
        current_refs = tuple(
            ref
            for ref in refs
            if any(evidence in chunk_values for evidence in ref.evidence)
        )
        if not current_refs:
            continue
        refined_refs.extend(
            _refine_catalog(
                llm,
                template,
                source_name=source_name,
                catalog=chunk,
                assets=_assets_for_evidence(assets, chunk),
                refs=current_refs,
                corrections=corrections,
            )
        )
        if on_decision is not None and len(chunks) > 1:
            on_decision(
                {
                    "stage": "discovery_refine_chunk",
                    "source": source_name,
                    "chunk": chunk_no,
                    "chunks": len(chunks),
                    "before": len(current_refs),
                    "after": len(refined_refs),
                }
            )
    refined = _merge_discovered_refs(refined_refs)
    if not refined:
        raise ContractError("chunked discovery refinement returned no ConceptRef")
    if on_decision is not None:
        on_decision(
            {
                "stage": "discovery_refine",
                "source": source_name,
                "chunks": len(chunks),
                "before": len(refs),
                "after": len(refined),
                "ref_ids": [item.concept_id for item in refined],
                "id_corrections": corrections,
            }
        )
    return refined


def _refine_catalog(
    llm: CompletionClient,
    template: str,
    *,
    source_name: str,
    catalog: dict[str, str],
    assets: tuple[SourceAsset, ...],
    refs: tuple[ConceptRef, ...],
    corrections: list[dict[str, str]],
) -> tuple[ConceptRef, ...]:
    evidence_ids = {text: evidence_id for evidence_id, text in catalog.items()}
    current = [
        {
            "id": ref.concept_id,
            "type": ref.type,
            "title": ref.title,
            "description": ref.description,
            "evidence": [
                evidence_ids[item]
                for item in ref.evidence
                if item in evidence_ids
            ],
            "asset_hints": [
                item
                for item in ref.asset_hints
                if item in {asset.asset_id for asset in assets}
            ],
        }
        for ref in refs
    ]
    prompt = render_prompt(
        template,
        current_refs=_json(current),
        evidence_catalog=_json(
            [
                {"evidence_id": evidence_id, "text": text}
                for evidence_id, text in catalog.items()
            ]
        ),
        asset_inventory=_json([asdict(item) for item in assets]),
    )
    return _complete_agent(
        llm,
        prompt,
        schema_name="agent_refined_discovery",
        schema=discovery_json_schema(1),
        parser=lambda response: _parse_agent_discovery(
            response,
            source_name=source_name,
            evidence_catalog=catalog,
            asset_ids={item.asset_id for item in assets},
            min_concepts=1,
            required_types=set(),
            corrections=corrections,
        ),
    )


def _partition_evidence_catalog(
    catalog: dict[str, str],
    *,
    maximum_chars: int,
) -> tuple[dict[str, str], ...]:
    if maximum_chars < 1:
        raise ValueError("maximum_chars must be positive")
    chunks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    size = 0
    for evidence_id, text in catalog.items():
        cost = len(text) + len(evidence_id) + 32
        if current and size + cost > maximum_chars:
            chunks.append(current)
            current = {}
            size = 0
        current[evidence_id] = text
        size += cost
    if current:
        chunks.append(current)
    return tuple(chunks)


def _assets_for_evidence(
    assets: tuple[SourceAsset, ...],
    catalog: dict[str, str],
) -> tuple[SourceAsset, ...]:
    combined = "\n\n".join(catalog.values())
    return tuple(asset for asset in assets if asset.raw in combined)


def _merge_discovered_refs(refs: list[ConceptRef]) -> tuple[ConceptRef, ...]:
    merged: list[ConceptRef] = []
    by_key: dict[tuple[str, str], int] = {}
    for ref in refs:
        key = (ref.type, re.sub(r"\s+", "", ref.title).casefold())
        position = by_key.get(key)
        if position is None:
            by_key[key] = len(merged)
            merged.append(ref)
            continue
        previous = merged[position]
        merged[position] = replace(
            previous,
            evidence=tuple(dict.fromkeys(previous.evidence + ref.evidence)),
            asset_hints=tuple(
                dict.fromkeys(previous.asset_hints + ref.asset_hints)
            ),
        )
    used: set[str] = set()
    unique: list[ConceptRef] = []
    for ref in merged:
        concept_id = ref.concept_id
        base = concept_id
        suffix = 2
        while concept_id in used:
            concept_id = f"{base}-{suffix}"
            suffix += 1
        used.add(concept_id)
        unique.append(
            ref if concept_id == ref.concept_id else replace(ref, concept_id=concept_id)
        )
    return tuple(unique)


def _asset_candidate_group_ids(
    assets: tuple[SourceAsset, ...],
    refs: tuple[AgentRefRecord, ...],
    group_by_ref: Mapping[str, str],
    drafts: Mapping[str, DraftConcept],
    target_groups: tuple[str, ...],
    *,
    limit: int,
) -> tuple[str, ...]:
    if limit < 1:
        raise ValueError("asset candidate group limit must be positive")
    asset_ids = {asset.asset_id for asset in assets}
    hinted = {
        group_by_ref[ref.ref_id]
        for ref in refs
        if ref.ref_id in group_by_ref
        and asset_ids.intersection(ref.asset_hints)
    }
    pool = hinted or set(target_groups)
    context = " ".join(
        f"{asset.before} {asset.after}" for asset in assets
    )
    context_terms = _placement_terms(context)
    ranked = sorted(
        pool,
        key=lambda group_id: (
            -len(
                context_terms
                & _placement_terms(
                    f"{drafts[group_id].title} "
                    f"{drafts[group_id].description} "
                    f"{drafts[group_id].body[:1200]}"
                )
            ),
            group_id,
        ),
    )
    return tuple(ranked[:limit])


def _placement_terms(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    words = {
        item.casefold()
        for item in re.findall(r"[A-Za-z0-9]{2,}|[\u3400-\u9fff]{2,}", compact)
    }
    for item in tuple(words):
        if re.fullmatch(r"[\u3400-\u9fff]{2,}", item):
            words.update(
                item[index : index + 2] for index in range(len(item) - 1)
            )
    return words


def _parse_agent_discovery(
    response: str,
    *,
    source_name: str,
    evidence_catalog: dict[str, str],
    asset_ids: set[str],
    min_concepts: int,
    required_types: set[str],
    corrections: list[dict[str, str]],
) -> tuple[ConceptRef, ...]:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return parse_discovery(
            response,
            source_name=source_name,
            evidence_catalog=evidence_catalog,
            asset_ids=asset_ids,
            min_concepts=min_concepts,
            required_types=required_types,
        )
    if isinstance(payload, dict) and isinstance(payload.get("concepts"), list):
        seen: set[str] = set()
        for position, item in enumerate(payload["concepts"], start=1):
            if not isinstance(item, dict):
                continue
            raw_id = item.get("id")
            if not isinstance(raw_id, str):
                continue
            candidate = raw_id.removesuffix(".md").strip()
            candidate = (
                candidate.replace("/", "-")
                .replace("\\", "-")
                .replace("..", "-")
            )
            if not candidate:
                candidate = f"concept-{position}"
            normalized = normalize_slug(f"{candidate}.md")[:-3]
            base = normalized
            suffix = 2
            while normalized in seen:
                normalized = f"{base}-{suffix}"
                suffix += 1
            seen.add(normalized)
            if normalized != raw_id:
                corrections.append({"from": raw_id, "to": normalized})
                item["id"] = normalized
        response = json.dumps(payload, ensure_ascii=False)
    return parse_discovery(
        response,
        source_name=source_name,
        evidence_catalog=evidence_catalog,
        asset_ids=asset_ids,
        min_concepts=min_concepts,
        required_types=required_types,
    )


def plan_compile_groups(
    llm: CompletionClient,
    template: str,
    refs: tuple[AgentRefRecord, ...],
    candidates: tuple[CandidateEdge, ...],
    *,
    max_component_refs: int = 24,
    on_decision: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[CompileGroup, ...]:
    by_id = {ref.ref_id: ref for ref in refs}
    components = _candidate_components(tuple(by_id), candidates)
    groups: list[CompileGroup] = []
    for component in components:
        for chunk in _partition_component(
            component, candidates, max_component_refs
        ):
            chunk_edges = tuple(
                edge
                for edge in candidates
                if edge.left_ref_id in chunk and edge.right_ref_id in chunk
            )
            if len(chunk) == 1 or not chunk_edges:
                for ref_id in chunk:
                    ref = by_id[ref_id]
                    groups.append(
                        _compile_group(
                            GroupDecision(
                                ref_ids=(ref_id,),
                                title=ref.title,
                                description=ref.description,
                                reason="没有可进入联合编译审查的跨文档候选边。",
                            ),
                            by_id,
                        )
                    )
                continue
            cards = {
                ref_id: {
                    "ref_id": ref_id,
                    "article_id": by_id[ref_id].article_id,
                    "source": by_id[ref_id].source,
                    "type": by_id[ref_id].type,
                    "title": by_id[ref_id].title,
                    "description": by_id[ref_id].description,
                    "section_path": list(by_id[ref_id].section_path),
                    "page_start": by_id[ref_id].page_start,
                    "page_end": by_id[ref_id].page_end,
                    "document_family_id": by_id[ref_id].document_family_id,
                    "document_version_id": by_id[ref_id].document_version_id,
                    "ref_family_hint": by_id[ref_id].ref_family_hint,
                    "semantic_signature": dict(by_id[ref_id].semantic_signature),
                    "scope": dict(by_id[ref_id].scope),
                    "evidence_preview": [
                        item[:500] for item in by_id[ref_id].evidence[:2]
                    ],
                }
                for ref_id in chunk
            }
            pairs = {
                tuple(sorted((edge.left_ref_id, edge.right_ref_id)))
                for edge in chunk_edges
            }
            prompt = render_prompt(
                template,
                ref_cards=_json(list(cards.values())),
                candidate_edges=_json([edge.as_dict() for edge in chunk_edges]),
            )
            recovery: dict[str, Any] = {}

            def recover_decisions(
                response: str,
                error: ContractError,
                *,
                cards: Mapping[str, Mapping[str, Any]] = cards,
                pairs: set[tuple[str, str]] = pairs,
            ) -> tuple[GroupDecision, ...]:
                recovered, ref_ids = recover_group_plan(
                    response,
                    refs=cards,
                    candidate_pairs=pairs,
                )
                recovery.update(
                    {
                        "mode": "demote_invalid_joint_groups",
                        "ref_ids": list(ref_ids),
                        "contract_error": str(error),
                    }
                )
                return recovered

            decisions = _complete_agent(
                llm,
                prompt,
                schema_name="agent_compile_groups",
                schema=group_plan_schema(chunk),
                parser=lambda response, cards=cards, pairs=pairs: parse_group_plan(
                    response, refs=cards, candidate_pairs=pairs
                ),
                recovery_parser=recover_decisions,
            )
            planned = tuple(_compile_group(item, by_id) for item in decisions)
            groups.extend(planned)
            if on_decision is not None:
                event = {
                    "stage": "compile_group_plan",
                    "component_ref_ids": list(chunk),
                    "groups": [asdict(item) for item in planned],
                }
                if recovery:
                    event["contract_recovery"] = {
                        "mode": recovery["mode"],
                        "ref_ids": recovery["ref_ids"],
                    }
                    event["contract_error"] = recovery["contract_error"]
                on_decision(event)
    expected = {ref.ref_id for ref in refs}
    actual = [ref_id for group in groups for ref_id in group.ref_ids]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ContractError("final compile groups must cover every Ref exactly once")
    return tuple(sorted(groups, key=lambda item: item.group_id))


def audit_concept_quality(
    llm: CompletionClient,
    template: str,
    *,
    draft: DraftConcept,
    refs: tuple[AgentRefRecord, ...],
    threshold: float,
    on_decision: Callable[[dict[str, Any]], None] | None = None,
) -> QualityAudit:
    prompt = render_prompt(
        template,
        quality_threshold=f"{threshold:.2f}",
        concept_refs=_json([asdict(item) for item in refs]),
        draft=_json(
            {
                "title": draft.title,
                "description": draft.description,
                "body": draft.body,
            }
        ),
    )
    audit = _complete_agent(
        llm,
        prompt,
        schema_name="agent_quality_audit",
        schema=quality_schema(),
        parser=lambda response: parse_quality_audit(response, threshold),
    )
    if on_decision is not None:
        on_decision(
            {
                "stage": "quality_audit",
                "concept_id": draft.ref.concept_id,
                "decision": asdict(audit),
            }
        )
    return audit


def recompile_concept(
    llm: CompletionClient,
    template: str,
    *,
    ref: ConceptRef,
    previous: DraftConcept,
    audit: QualityAudit,
) -> DraftConcept:
    prompt = render_prompt(
        template,
        concept_ref=_json(
            {
                "concept_id": ref.concept_id,
                "type": ref.type,
                "title": ref.title,
                "description": ref.description,
                "source": ref.source,
            }
        ),
        evidence=_json(list(ref.evidence)),
        previous_draft=_json(
            {
                "title": previous.title,
                "description": previous.description,
                "body": previous.body,
            }
        ),
        quality_issues=_json(list(audit.issues)),
        recompile_instructions=audit.recompile_instructions,
    )
    return _complete_agent(
        llm,
        prompt,
        schema_name="agent_recompiled_concept",
        schema=draft_json_schema(),
        parser=lambda response: parse_draft(response, ref),
    )


def _source_record_from_payload(
    path: Path,
    content: str,
    assets: tuple[SourceAsset, ...],
    payload: Mapping[str, Any],
    *,
    structure: Mapping[str, Any] | None = None,
) -> _SourceRecord:
    raw_profile = payload["profile"]
    profile = DocumentProfile(
        source=raw_profile["source"],
        title=raw_profile["title"],
        character_count=int(raw_profile["character_count"]),
        evidence_count=int(raw_profile["evidence_count"]),
        heading_count=int(raw_profile["heading_count"]),
        structured_section_count=int(raw_profile["structured_section_count"]),
        heading_outline=tuple(raw_profile["heading_outline"]),
        section_previews=tuple(raw_profile["section_previews"]),
        asset_count=int(raw_profile["asset_count"]),
        asset_kinds=tuple(raw_profile["asset_kinds"]),
        asset_contexts=tuple(raw_profile["asset_contexts"]),
        document_family_id=str(raw_profile.get("document_family_id") or ""),
        document_version_id=str(raw_profile.get("document_version_id") or ""),
        metadata=dict(raw_profile.get("metadata") or {}),
    )
    raw_plan = payload["plan"]
    plan = SourcePlan(
        discovery_mode=raw_plan["discovery_mode"],
        refine_discovery=bool(raw_plan["refine_discovery"]),
        asset_policy=raw_plan["asset_policy"],
        reason=raw_plan["reason"],
    )
    refs = tuple(
        AgentRefRecord(
            ref_id=item["ref_id"],
            article_id=item["article_id"],
            local_id=item["local_id"],
            type=item["type"],
            title=item["title"],
            description=item["description"],
            evidence=tuple(item["evidence"]),
            asset_hints=tuple(item["asset_hints"]),
            source=item["source"],
            section_path=tuple(item.get("section_path") or ()),
            page_start=_optional_page(item.get("page_start")),
            page_end=_optional_page(item.get("page_end")),
            evidence_block_ids=tuple(item.get("evidence_block_ids") or ()),
            semantic_signature=dict(item.get("semantic_signature") or {}),
            scope=dict(item.get("scope") or {}),
            ref_family_hint=str(item.get("ref_family_hint") or ""),
            ref_version_id=str(item.get("ref_version_id") or ""),
            document_family_id=str(item.get("document_family_id") or ""),
            document_version_id=str(item.get("document_version_id") or ""),
        )
        for item in payload["refs"]
    )
    return _SourceRecord(
        path,
        content,
        assets,
        profile,
        plan,
        refs,
        structure,
    )


def _load_source_structure(path: Path) -> Mapping[str, Any] | None:
    structure_path = path.with_suffix(".structure.json")
    if not structure_path.is_file():
        return None
    try:
        value = json.loads(structure_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AgentRunError(
            f"invalid source structure sidecar: {structure_path.name}"
        ) from error
    if not isinstance(value, dict):
        raise AgentRunError(
            f"source structure sidecar must be an object: {structure_path.name}"
        )
    if value.get("schema_version") != "kmpro.document-structure.v1":
        raise AgentRunError(
            f"unsupported source structure schema: {structure_path.name}"
        )
    if value.get("status") != "complete":
        unresolved = [
            int(item.get("page_number") or 0)
            for item in value.get("pages", [])
            if isinstance(item, dict) and item.get("role") == "content_retry"
        ]
        raise AgentRunError(
            f"source structure requires page-role review: {path.name}; "
            f"pages={unresolved[:20]}"
        )
    return value


def _attach_agent_ref_provenance(
    ref: AgentRefRecord,
    structure: Mapping[str, Any] | None,
) -> AgentRefRecord:
    if structure is None:
        return ref
    matches: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    blocks = [
        item
        for item in structure.get("blocks", [])
        if isinstance(item, dict)
        and bool(item.get("evidence_eligible"))
        and isinstance(item.get("block_id"), str)
    ]
    for evidence in ref.evidence:
        normalized_evidence = _provenance_text(evidence)
        for block in blocks:
            block_id = str(block["block_id"])
            if block_id in seen:
                continue
            content = str(block.get("content") or "").strip()
            asset_uri = str(block.get("asset_uri") or "").strip()
            if (
                content
                and _provenance_text(content)
                and _provenance_text(content) in normalized_evidence
            ) or (asset_uri and asset_uri in evidence):
                matches.append(block)
                seen.add(block_id)
    if not matches:
        return ref
    pages = [
        int(item["page_idx"])
        for item in matches
        if isinstance(item.get("page_idx"), int)
        and int(item["page_idx"]) >= 0
    ]
    paths = [
        tuple(str(part) for part in item.get("heading_path") or ())
        for item in matches
        if item.get("heading_path")
    ]
    return replace(
        ref,
        section_path=_common_heading_path(paths),
        page_start=(min(pages) + 1 if pages else None),
        page_end=(max(pages) + 1 if pages else None),
        evidence_block_ids=tuple(str(item["block_id"]) for item in matches),
    )


def _provenance_text(value: str) -> str:
    text = re.sub(r"(?m)^\s*#{1,6}\s+", "", value)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _common_heading_path(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    if not paths:
        return ()
    common = list(paths[0])
    for path in paths[1:]:
        limit = min(len(common), len(path))
        position = 0
        while position < limit and common[position] == path[position]:
            position += 1
        common = common[:position]
        if not common:
            break
    return tuple(common)


def _merge_scope(refs: list[AgentRefRecord]) -> dict[str, Any]:
    """Keep scope dimensions explicit when a Concept has multiple Refs."""
    merged: dict[str, Any] = {}
    for ref in refs:
        for key, value in ref.scope.items():
            if value in (None, "", [], {}):
                continue
            existing = merged.get(key)
            values = []
            for item in (existing, value):
                if item in (None, "", [], {}):
                    continue
                if isinstance(item, (list, tuple, set)):
                    values.extend(item)
                else:
                    values.append(item)
            deduped = list(dict.fromkeys(str(item) for item in values))
            merged[key] = deduped if len(deduped) > 1 else deduped[0]
    return merged


def _optional_page(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 1 else None


def _draft_payload(draft: DraftConcept) -> dict[str, Any]:
    return {
        "ref": asdict(draft.ref),
        "title": draft.title,
        "description": draft.description,
        "body": draft.body,
    }


def _draft_from_payload(payload: Mapping[str, Any]) -> DraftConcept:
    raw_ref = payload["ref"]
    ref = ConceptRef(
        concept_id=raw_ref["concept_id"],
        type=raw_ref["type"],
        title=raw_ref["title"],
        description=raw_ref["description"],
        source=raw_ref["source"],
        evidence=tuple(raw_ref["evidence"]),
        asset_hints=tuple(raw_ref["asset_hints"]),
        section_path=tuple(raw_ref.get("section_path") or ()),
        page_start=_optional_page(raw_ref.get("page_start")),
        page_end=_optional_page(raw_ref.get("page_end")),
        evidence_block_ids=tuple(raw_ref.get("evidence_block_ids") or ()),
        semantic_signature=dict(raw_ref.get("semantic_signature") or {}),
        scope=dict(raw_ref.get("scope") or {}),
        ref_family_hint=str(raw_ref.get("ref_family_hint") or ""),
        ref_version_id=str(raw_ref.get("ref_version_id") or ""),
        document_family_id=str(raw_ref.get("document_family_id") or ""),
        document_version_id=str(raw_ref.get("document_version_id") or ""),
    )
    return DraftConcept(
        ref=ref,
        title=payload["title"],
        description=payload["description"],
        body=payload["body"],
    )


def _group_ref(
    group: CompileGroup, by_id: Mapping[str, AgentRefRecord]
) -> ConceptRef:
    members = [by_id[ref_id] for ref_id in group.ref_ids]
    evidence: list[str] = []
    for member in members:
        for item in member.evidence:
            if item not in evidence:
                evidence.append(item)
    pages_start = [item.page_start for item in members if item.page_start is not None]
    pages_end = [item.page_end for item in members if item.page_end is not None]
    section_path = (
        _common_heading_path([item.section_path for item in members])
        if len({item.source for item in members}) == 1
        else ()
    )
    return ConceptRef(
        concept_id=group.group_id,
        type=members[0].type,
        title=group.title,
        description=group.description,
        source=(
            members[0].source
            if len({item.source for item in members}) == 1
            else "多来源联合编译"
        ),
        evidence=tuple(evidence),
        asset_hints=tuple(
            sorted({hint for member in members for hint in member.asset_hints})
        ),
        section_path=section_path,
        page_start=min(pages_start) if pages_start else None,
        page_end=max(pages_end) if pages_end else None,
        evidence_block_ids=tuple(
            dict.fromkeys(
                block_id
                for member in members
                for block_id in member.evidence_block_ids
            )
        ),
        semantic_signature=dict(
            next(
                (
                    member.semantic_signature
                    for member in members
                    if member.semantic_signature
                ),
                {},
            )
        ),
        scope=_merge_scope(members),
        ref_family_hint=(
            members[0].ref_family_hint
            if len({member.ref_family_hint for member in members}) == 1
            else ""
        ),
        ref_version_id=(
            members[0].ref_version_id
            if len({member.ref_version_id for member in members}) == 1
            else ""
        ),
        document_family_id=(
            members[0].document_family_id
            if len({member.document_family_id for member in members}) == 1
            else ""
        ),
        document_version_id=(
            members[0].document_version_id
            if len({member.document_version_id for member in members}) == 1
            else ""
        ),
    )


def _compile_group(
    decision: GroupDecision, refs: Mapping[str, AgentRefRecord]
) -> CompileGroup:
    first = refs[decision.ref_ids[0]]
    digest = hashlib.sha256("\0".join(decision.ref_ids).encode()).hexdigest()[:12]
    group_id = f"{kind_for(first.candidate_payload()).lower()}-{digest}"
    return CompileGroup(
        group_id=group_id,
        ref_ids=decision.ref_ids,
        title=decision.title,
        description=decision.description,
        reason=decision.reason,
    )


def _compile_group_cache_hash(
    group: CompileGroup,
    refs: Mapping[str, AgentRefRecord],
    templates: Mapping[str, str],
    model: str,
    policy: AgentPolicy,
) -> str:
    return stable_hash(
        {
            "group": asdict(group),
            "synthetic_ref": asdict(_group_ref(group, refs)),
            "prompts": {
                name: templates[name]
                for name in ("compile", "agent_quality", "agent_recompile")
            },
            "model": model,
            "policy": asdict(policy),
        }
    )


def _candidate_components(
    ref_ids: tuple[str, ...], candidates: tuple[CandidateEdge, ...]
) -> tuple[tuple[str, ...], ...]:
    adjacency: dict[str, set[str]] = {ref_id: set() for ref_id in ref_ids}
    for edge in candidates:
        adjacency[edge.left_ref_id].add(edge.right_ref_id)
        adjacency[edge.right_ref_id].add(edge.left_ref_id)
    unseen = set(ref_ids)
    components: list[tuple[str, ...]] = []
    while unseen:
        start = min(unseen)
        queue = deque([start])
        reached: set[str] = set()
        while queue:
            current = queue.popleft()
            if current in reached:
                continue
            reached.add(current)
            queue.extend(sorted(adjacency[current] - reached))
        unseen -= reached
        components.append(tuple(sorted(reached)))
    return tuple(components)


def _partition_component(
    component: tuple[str, ...],
    candidates: tuple[CandidateEdge, ...],
    maximum: int,
) -> tuple[tuple[str, ...], ...]:
    if len(component) <= maximum:
        return (component,)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in candidates:
        if edge.left_ref_id in component and edge.right_ref_id in component:
            adjacency[edge.left_ref_id].add(edge.right_ref_id)
            adjacency[edge.right_ref_id].add(edge.left_ref_id)
    remaining = set(component)
    chunks: list[tuple[str, ...]] = []
    while remaining:
        queue = deque([min(remaining)])
        chunk: list[str] = []
        queued = set(queue)
        while queue and len(chunk) < maximum:
            current = queue.popleft()
            if current not in remaining:
                continue
            remaining.remove(current)
            chunk.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in remaining and neighbor not in queued:
                    queue.append(neighbor)
                    queued.add(neighbor)
        chunks.append(tuple(sorted(chunk)))
    return tuple(chunks)


def _heading_sections(content: str) -> tuple[tuple[int, str, int, int], ...]:
    matches = list(
        re.finditer(
            r"(?m)^(?P<marks>#{2,4})[ \t]+(?P<title>.+?)[ \t]*$",
            content,
        )
    )
    sections: list[tuple[int, str, int, int]] = []
    for index, match in enumerate(matches):
        depth = len(match.group("marks"))
        end = len(content)
        for following in matches[index + 1 :]:
            if len(following.group("marks")) <= depth:
                end = following.start()
                break
        sections.append((depth, match.group("title").strip(), match.start(), end))
    return tuple(sections)


def _select_heading_level(
    headings: tuple[tuple[int, str, int, int], ...],
) -> tuple[tuple[int, str, int, int], ...]:
    for depth in (2, 3, 4):
        selected = tuple(
            item
            for item in headings
            if item[0] == depth and item[3] - item[2] >= 24
        )
        if len(selected) >= 2:
            return selected
    return ()


def _markdown_body_text(value: str) -> str:
    text = re.sub(r"(?m)^\s*#{1,6}\s+.*$", "", value)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"<(?:img|br)\b[^>]*>", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", "", text)


def _has_substantive_markdown(value: str) -> bool:
    return bool(_markdown_body_text(value))


def _has_substantive_evidence(values: tuple[str, ...]) -> bool:
    return _has_substantive_markdown("\n".join(values))


def _logical_heading_rank(title: str) -> int:
    if re.match(r"^\s*[一二三四五六七八九十]+[、.．]", title):
        return 1
    if re.match(r"^\s*[（(][一二三四五六七八九十0-9]+[）)]", title):
        return 2
    if re.match(r"^\s*\d+[、.．]", title):
        return 3
    return 99


def _container_heading_titles(content: str) -> set[str]:
    headings = _heading_sections(content)
    containers: set[str] = set()
    for current, following in zip(headings, headings[1:]):
        depth, title, start, end = current
        next_depth, next_title, _next_start, _next_end = following
        if (
            depth == next_depth
            and _logical_heading_rank(title) < _logical_heading_rank(next_title)
            and len(_markdown_body_text(content[start:end])) < 200
        ):
            containers.add(_clean_heading(title))
    return containers


def _infer_type(title: str) -> str:
    if re.search(r"口径|指标|计算|数据来源|定义", title):
        return "数据口径"
    if re.search(r"国际|国外|境外|全球|对标|比较", title):
        return "国际比较"
    if re.search(r"建议|对策|行动|举措|路径|措施", title):
        return "政策建议"
    if re.search(r"术语|释义|概念", title):
        return "术语解释"
    return "分析框架"


def _section_description(block: str, title: str) -> str:
    text = re.sub(r"(?m)^#{1,6}\s+.*$", "", block)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return f"{_clean_heading(title)}的结构化知识单元。"
    sentence = re.split(r"(?<=[。！？；])", text)[0].strip()
    return sentence[:160]


def _clean_heading(title: str) -> str:
    cleaned = re.sub(
        r"^\s*(?:[一二三四五六七八九十]+[、.．]|"
        r"[（(][一二三四五六七八九十0-9]+[）)]|"
        r"\d+[、.．])\s*",
        "",
        title,
    ).strip()
    return cleaned or title.strip()


def _extract_title(source_name: str, content: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", content)
    return match.group(1).strip() if match else Path(source_name).stem


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{stable_hash(value)[:12]}"


_T = TypeVar("_T")


def _complete_agent(
    llm: CompletionClient,
    prompt: str,
    *,
    schema_name: str,
    schema: dict[str, Any],
    parser: Callable[[str], _T],
    recovery_parser: Callable[[str, ContractError], _T] | None = None,
) -> _T:
    current_prompt = prompt
    for attempt in range(2):
        response = llm.complete(
            current_prompt,
            json_schema_name=schema_name,
            json_schema=schema,
        )
        try:
            return parser(response)
        except ContractError as error:
            if attempt == 1:
                if recovery_parser is not None:
                    return recovery_parser(response, error)
                raise
            current_prompt = (
                f"{prompt}\n\n## 上次决策未通过代码合同\n\n"
                f"错误：{error}\n\n上次输出：{response}\n\n"
                "重新执行原任务，只输出满足 Schema 和代码合同的完整 JSON。"
            )
    raise AssertionError("Agent structured completion attempts exhausted")


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)
