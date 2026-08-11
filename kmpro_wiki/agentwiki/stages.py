from __future__ import annotations

import json
import re
from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol, TypeVar

from .assets import SourceAsset
from .contracts import (
    AssetPlacement,
    ContractError,
    ConceptRef,
    DraftConcept,
    RelationAudit,
    discovery_json_schema,
    draft_json_schema,
    parse_discovery,
    parse_draft,
    parse_placements,
    parse_relation_audit,
    placements_json_schema,
    relation_json_schema,
)
from .llm import LLMOutputTruncated
from .okf import ConceptDocument
from .relations import (
    RelationError,
    build_relation_anchor_catalog,
    validate_relation_anchor_selection,
)


class CompletionClient(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        json_schema_name: str | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> str: ...


class PromptRenderError(ValueError):
    pass


@dataclass(frozen=True)
class DiscoveryConstraints:
    min_concepts: int
    required_types: tuple[str, ...]
    outline: tuple[str, ...]


_PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")


def render_prompt(template: str, **values: str) -> str:
    placeholders = set(_PLACEHOLDER_RE.findall(template))
    expected = {f"{{{name}}}" for name in values}
    unknown = placeholders - expected
    if unknown:
        raise PromptRenderError(
            f"unknown placeholders: {', '.join(sorted(unknown))}"
        )
    rendered = template
    for name, value in values.items():
        rendered = rendered.replace(f"{{{name}}}", value)
    return rendered


def discover_concepts(
    llm: CompletionClient,
    template: str,
    *,
    title: str,
    source_name: str,
    source_content: str,
    assets: tuple[SourceAsset, ...],
) -> tuple[ConceptRef, ...]:
    evidence_catalog = build_evidence_catalog(source_content)
    constraints = infer_discovery_constraints(source_content)
    min_concepts = max(
        constraints.min_concepts,
        2 if len(evidence_catalog) >= 8 else 1,
    )
    prompt = render_prompt(
        template,
        title=title,
        source_file=source_name,
        evidence_catalog=_json(
            [
                {"evidence_id": evidence_id, "text": text}
                for evidence_id, text in evidence_catalog.items()
            ]
        ),
        minimum_concepts=str(min_concepts),
        required_types=_json(list(constraints.required_types)),
        source_outline=_json(list(constraints.outline)),
        asset_inventory=_json([_asset_payload(item) for item in assets]),
    )
    return _complete_structured(
        llm,
        prompt,
        schema_name="concept_discovery",
        schema=discovery_json_schema(min_concepts),
        parser=lambda response: parse_discovery(
            response,
            source_name=source_name,
            evidence_catalog=evidence_catalog,
            asset_ids={item.asset_id for item in assets},
            min_concepts=min_concepts,
            required_types=set(constraints.required_types),
        ),
    )


def infer_discovery_constraints(source_content: str) -> DiscoveryConstraints:
    outline = tuple(
        match.group("title").strip()
        for match in re.finditer(
            r"(?m)^#{1,6}\s+(?P<title>.+?)\s*$", source_content
        )
    )
    joined = "\n".join(outline)
    required: list[str] = []
    if re.search(r"构建说明|指标设置|计算方式|数据来源|统计口径", joined):
        required.append("数据口径")
    if re.search(r"运行特征|现状|趋势|风险|问题|分析|评估", joined):
        required.append("分析框架")
    if re.search(r"建议|对策|行动|实施路径|工作措施", joined):
        required.append("政策建议")
    reusable_sections = len(
        re.findall(r"(?m)^##\s+（[一二三四五六七八九十]+）", source_content)
    )
    minimum = max(1, len(required), min(8, reusable_sections))
    return DiscoveryConstraints(minimum, tuple(required), outline)


def build_evidence_catalog(source_content: str) -> dict[str, str]:
    raw_blocks = [
        block.strip()
        for block in re.split(r"\n[ \t]*\n+", source_content)
        if block.strip()
    ]
    blocks: list[str] = []
    for block in raw_blocks:
        if blocks and _should_merge_evidence_blocks(blocks[-1], block):
            blocks[-1] = f"{blocks[-1]}\n\n{block}"
        else:
            blocks.append(block)
    return {
        f"evidence-{index:04d}": block
        for index, block in enumerate(blocks, start=1)
    }


def _should_merge_evidence_blocks(previous: str, current: str) -> bool:
    if previous.lstrip().startswith(("#", "!", "|", "<table")):
        return False
    if current.lstrip().startswith(("#", "!", "|", "<table")):
        return False
    return previous.rstrip()[-1] not in "。！？；：.!?;)>）】]"


def compile_concepts(
    llm: CompletionClient,
    template: str,
    refs: tuple[ConceptRef, ...],
    *,
    on_event: Callable[[str], None] | None = None,
) -> tuple[DraftConcept, ...]:
    emit = on_event or (lambda _message: None)
    drafts: list[DraftConcept] = []
    for position, ref in enumerate(refs, start=1):
        emit(
            f"concept.start concept={ref.concept_id} position={position}/{len(refs)}"
        )
        drafts.append(compile_one_concept(llm, template, ref))
        emit(
            f"concept.done concept={ref.concept_id} position={position}/{len(refs)}"
        )
    return tuple(drafts)


def compile_one_concept(
    llm: CompletionClient,
    template: str,
    ref: ConceptRef,
) -> DraftConcept:
    prompt = render_prompt(
        template,
        concept_ref=_json(_ref_payload(ref)),
        evidence=_json(list(ref.evidence)),
    )
    return _complete_structured(
        llm,
        prompt,
        schema_name="concept_draft",
        schema=draft_json_schema(),
        parser=lambda response: parse_draft(response, ref),
    )


def plan_asset_placements(
    llm: CompletionClient,
    template: str,
    *,
    assets: tuple[SourceAsset, ...],
    drafts: tuple[DraftConcept, ...],
) -> tuple[AssetPlacement, ...]:
    if not assets:
        return ()
    anchor_catalog = build_anchor_catalog(drafts)
    prompt = render_prompt(
        template,
        asset_inventory=_json([_asset_payload(item) for item in assets]),
        concepts=_json([_draft_payload(item) for item in drafts]),
        anchor_catalog=_json(
            [
                {
                    "concept_id": concept_id,
                    "anchor_id": anchor_id,
                    "text": text,
                }
                for (concept_id, anchor_id), text in anchor_catalog.items()
            ]
        ),
    )
    return _complete_structured(
        llm,
        prompt,
        schema_name="asset_placements",
        schema=placements_json_schema(len(assets)),
        parser=lambda response: parse_placements(
            response,
            asset_ids={item.asset_id for item in assets},
            concept_ids={item.ref.concept_id for item in drafts},
            anchor_catalog=anchor_catalog,
        ),
    )


def build_anchor_catalog(
    drafts: tuple[DraftConcept, ...],
) -> dict[tuple[str, str], str]:
    catalog: dict[tuple[str, str], str] = {}
    for draft in drafts:
        candidates: list[str] = []
        for line in draft.body.splitlines() or [draft.body]:
            candidates.extend(
                value.strip()
                for value in re.findall(r"[^。！？；]+[。！？；]?", line)
                if value.strip()
            )
        unique = [value for value in candidates if draft.body.count(value) == 1]
        if not unique:
            raise ContractError(
                f"concept has no unique asset anchor: {draft.ref.concept_id}"
            )
        for position, value in enumerate(unique, start=1):
            catalog[(draft.ref.concept_id, f"anchor-{position:03d}")] = value
    return catalog


def audit_relations(
    llm: CompletionClient,
    template: str,
    concepts: dict[str, ConceptDocument],
    *,
    current_ids: tuple[str, ...] | None = None,
    on_event: Callable[[str], None] | None = None,
) -> dict[str, RelationAudit]:
    emit = on_event or (lambda _message: None)
    audits: dict[str, RelationAudit] = {}
    selected_ids = tuple(concepts) if current_ids is None else current_ids
    for concept_id in selected_ids:
        if concept_id not in concepts:
            raise ValueError(f"unknown current concept: {concept_id}")
    for position, concept_id in enumerate(selected_ids, start=1):
        current = concepts[concept_id]
        candidates = {
            candidate_id: candidate
            for candidate_id, candidate in concepts.items()
            if candidate_id != concept_id
        }
        emit(
            f"relation.start concept={concept_id} position={position}/{len(selected_ids)}"
        )
        anchors = build_relation_anchor_catalog(current, candidates)
        anchor_catalog = {
            item.anchor_id: (item.target_id, item.text, item.occurrence)
            for item in anchors
        }
        linkable_target_ids = {item.target_id for item in anchors}
        eligible_candidates = {
            candidate_id: candidate
            for candidate_id, candidate in candidates.items()
            if candidate_id in linkable_target_ids
        }
        if not eligible_candidates:
            audits[concept_id] = RelationAudit("no_links", ())
            emit(
                f"relation.done concept={concept_id} status=no_links "
                f"position={position}/{len(selected_ids)}"
            )
            continue
        prompt = render_prompt(
            template,
            current_concept=_json(_concept_payload(concept_id, current, body=True)),
            candidate_index=_json(
                [
                    _concept_payload(candidate_id, candidate, body=False)
                    for candidate_id, candidate in eligible_candidates.items()
                ]
            ),
            anchor_catalog=_json(
                [
                    {
                        "anchor_id": item.anchor_id,
                        "target_id": item.target_id,
                        "text": item.text,
                        "context": item.context,
                    }
                    for item in anchors
                ]
            ),
        )
        audits[concept_id] = _complete_structured(
            llm,
            prompt,
            schema_name="relation_audit",
            schema=relation_json_schema(set(anchor_catalog)),
            parser=lambda response: _parse_relation_response(
                response,
                anchor_catalog,
                current,
                lambda dropped: emit(
                    f"relation.pruned concept={concept_id} dropped={dropped}"
                ),
            ),
        )
        emit(
            f"relation.done concept={concept_id} "
            f"status={audits[concept_id].status} position={position}/{len(selected_ids)}"
        )
    return audits


def _parse_relation_response(
    response: str,
    anchor_catalog: dict[str, tuple[str, str, int]],
    current: ConceptDocument,
    on_pruned: Callable[[int], None],
) -> RelationAudit:
    audit = parse_relation_audit(
        response,
        anchor_catalog=anchor_catalog,
        current_body=current.body,
    )
    raw_links = json.loads(response).get("links", [])
    dropped = max(0, len(raw_links) - len(audit.links))
    if dropped:
        on_pruned(dropped)
    try:
        validate_relation_anchor_selection(current, audit)
    except RelationError as error:
        raise ContractError(str(error)) from error
    return audit


def _asset_payload(asset: SourceAsset) -> dict[str, object]:
    return {
        "asset_id": asset.asset_id,
        "kind": asset.kind,
        "raw": asset.raw,
        "target": asset.target,
        "before": asset.before,
        "after": asset.after,
        "ordinal": asset.ordinal,
    }


def _ref_payload(ref: ConceptRef) -> dict[str, object]:
    return {
        "concept_id": ref.concept_id,
        "type": ref.type,
        "title": ref.title,
        "description": ref.description,
        "source": ref.source,
        "asset_hints": list(ref.asset_hints),
    }


def _draft_payload(draft: DraftConcept) -> dict[str, object]:
    return {
        **_ref_payload(draft.ref),
        "title": draft.title,
        "description": draft.description,
        "body": draft.body,
    }


def _concept_payload(
    concept_id: str, concept: ConceptDocument, *, body: bool
) -> dict[str, object]:
    payload: dict[str, object] = {
        "concept_id": concept_id,
        "filename": concept.filename,
        "type": concept.frontmatter.get("type", ""),
        "title": concept.frontmatter.get("title", ""),
        "description": concept.frontmatter.get("description", ""),
        "source": concept.frontmatter.get("source", ""),
    }
    if body:
        payload["body"] = concept.body
    return payload


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


_T = TypeVar("_T")


def _complete_structured(
    llm: CompletionClient,
    prompt: str,
    *,
    schema_name: str,
    schema: dict[str, object],
    parser: Callable[[str], _T],
    max_attempts: int = 2,
) -> _T:
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    current_prompt = prompt
    for attempt in range(max_attempts):
        try:
            response = llm.complete(
                current_prompt,
                json_schema_name=schema_name,
                json_schema=schema,
            )
        except LLMOutputTruncated:
            if attempt == max_attempts - 1:
                raise
            # A truncated response has no usable payload: its partial output
            # must never be appended as guidance.  Retry the original prompt
            # verbatim so the next draw starts from a clean slate.
            current_prompt = prompt
            continue
        try:
            return parser(response)
        except ContractError as error:
            if attempt == max_attempts - 1:
                raise
            current_prompt = (
                f"{prompt}\n\n"
                "## 上次结构化输出验证失败\n\n"
                f"验证错误：{error}\n\n"
                f"上次输出：{response}\n\n"
                "请重新执行原任务，只输出满足 JSON Schema 和上述语义约束的完整 JSON。"
            )
    raise AssertionError("structured completion attempts exhausted")
