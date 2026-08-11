"""Materialize a fully reviewed Claim Review run as an immutable C1 corpus.

The adapter never writes to either input run.  It accepts only a formal,
complete Claim Review whose source snapshot still matches the AgentWiki run,
then writes the minimal directory contract consumed by ``build_c1_audited_concepts``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from kmpro_wiki.agentwiki.claim_review import build_draft_sentences
from kmpro_wiki.agentwiki.contracts import ContractError
from kmpro_wiki.agentwiki.okf import (
    OKFValidationError,
    ConceptDocument,
    normalize_slug,
    parse_concept_markdown,
)
from kmpro_wiki.agentwiki.state import stable_hash


C1_MATERIALIZATION_SCHEMA = "okfolio.c1-materialized-run.v1"
REVIEWED_PROGRESS_SCHEMA = "okfolio.reviewed-compile-progress.v1"
CLAIM_REVIEW_SCHEMA = "okfolio.claim-review-run.v1"
CLAIM_COVERAGE_SCHEMA = "okfolio.claim-coverage.v2"
SOURCE_INPUTS = (
    "source_progress.json",
    "groups.json",
    "compile_progress.json",
)


class C1MaterializationError(RuntimeError):
    """An input run is not safe to publish as a formal C1 corpus."""


def materialize_c1_run(
    *,
    source_run: Path,
    review_run: Path,
    output_dir: Path,
    expected_groups: int | None = None,
) -> dict[str, Any]:
    """Create a standard C1 directory from frozen AgentWiki and review runs.

    The frozen ``groups.json`` defines the complete dataset by default.  An
    explicit ``expected_groups`` is an additional dataset-specific assertion,
    never a library-wide corpus-size policy.
    """
    source_run = source_run.expanduser().resolve()
    review_run = review_run.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _validate_paths(source_run, review_run, output_dir)

    source_paths = {name: source_run / name for name in SOURCE_INPUTS}
    review_paths = {
        "manifest.json": review_run / "manifest.json",
        "source_snapshot.json": review_run / "source_snapshot.json",
        "reviewed_compile_progress.json": (
            review_run / "reviewed_compile_progress.json"
        ),
        "review_queue.json": review_run / "review_queue.json",
    }
    missing = [
        str(path)
        for path in (*source_paths.values(), *review_paths.values())
        if not path.is_file()
    ]
    if missing:
        raise C1MaterializationError(
            "missing C1 materialization input: " + min(missing)
        )

    source_progress = _read_object(source_paths["source_progress.json"])
    groups_payload = _read_object(source_paths["groups.json"])
    review_manifest = _read_object(review_paths["manifest.json"])
    source_snapshot = _read_object(review_paths["source_snapshot.json"])
    reviewed = _read_object(review_paths["reviewed_compile_progress.json"])
    review_queue = _read_object(review_paths["review_queue.json"])

    _validate_source_snapshot(
        source_snapshot,
        review_manifest=review_manifest,
        source_paths=source_paths,
        source_progress=source_progress,
    )
    group_count = _resolve_group_count(
        groups_payload,
        expected_groups=expected_groups,
    )
    _validate_review_manifest(
        review_manifest,
        source_run=source_run,
        expected_groups=group_count,
    )
    refs, by_ref = _source_refs(source_progress)
    groups = _source_groups(
        groups_payload,
        by_ref=by_ref,
        expected_groups=group_count,
    )
    group_ids = tuple(str(item["group_id"]) for item in groups)
    _validate_reviewed_progress(
        reviewed,
        group_ids=group_ids,
        review_queue=review_queue,
    )

    drafts = _mapping(reviewed["drafts"], "reviewed drafts")
    reviews = _mapping(reviewed["claim_reviews"], "claim reviews")
    concept_documents: dict[str, str] = {}
    concepts: list[dict[str, Any]] = []
    seen_claim_ids: set[str] = set()
    article_ids: set[str] = set()
    for group in groups:
        group_id = str(group["group_id"])
        ref_ids = tuple(str(item) for item in group["ref_ids"])
        group_refs = tuple(by_ref[ref_id] for ref_id in ref_ids)
        draft = _mapping(drafts[group_id], f"draft {group_id}")
        title, description, body, concept_type = _draft_content(
            group_id,
            draft,
            group_refs,
        )
        claim_review = _mapping(reviews[group_id], f"claim review {group_id}")
        contract, claims, sentence_attributions = _validated_claim_review(
            group_id,
            ref_ids,
            group_refs,
            claim_review,
            draft={"description": description, "body": body},
            seen_claim_ids=seen_claim_ids,
        )
        articles = sorted({str(item["article_id"]) for item in group_refs})
        sources = sorted({str(item["source"]) for item in group_refs})
        article_ids.update(articles)
        locations = [_source_location(item) for item in group_refs]
        frontmatter = {
            "type": concept_type,
            "title": title,
            "description": description,
            "source": sources[0] if len(sources) == 1 else "多来源联合编译",
            "concept_refs": list(ref_ids),
            "articles": articles,
            "canonical_question": contract["canonical_question"],
            "claims": claims,
            "sentence_attributions": sentence_attributions,
            "claim_members": contract["members"],
            "source_locations": locations,
            "claim_review_decision": "pass",
            "claim_contract_schema": contract.get("schema_version"),
            "claim_coverage_schema": CLAIM_COVERAGE_SCHEMA,
        }
        filename = _concept_filename(group_id)
        document = ConceptDocument(
            filename=filename,
            frontmatter=frontmatter,
            body=body,
        )
        rendered = document.render()
        parsed = parse_concept_markdown(document.filename, rendered)
        if not parsed.body or parsed.frontmatter.get("concept_refs") != list(ref_ids):
            raise C1MaterializationError(
                f"rendered Concept contract failed: {group_id}"
            )
        concept_documents[filename] = rendered
        concepts.append(
            {
                **group,
                "articles": articles,
                "sources": sources,
                "status": "publishable",
                "canonical_question": contract["canonical_question"],
                "claims": claims,
                "sentence_attributions": sentence_attributions,
                "claim_members": contract["members"],
                "source_locations": locations,
                "claim_review": {
                    "decision": "pass",
                    "recompile_attempts": int(
                        claim_review.get("recompile_attempts") or 0
                    ),
                    "contract_schema": contract.get("schema_version"),
                    "coverage_schema": _mapping(
                        claim_review["coverage"], f"coverage {group_id}"
                    ).get("schema_version"),
                },
            }
        )

    input_fingerprints = {
        "source_run": {
            name: _file_fingerprint(path) for name, path in source_paths.items()
        },
        "claim_review": {
            name: _file_fingerprint(path) for name, path in review_paths.items()
        },
    }
    materialization_id = stable_hash(input_fingerprints)
    manifest = {
        "schema": C1_MATERIALIZATION_SCHEMA,
        "version": 1,
        "status": "complete",
        "materialization_id": materialization_id,
        "source_run_name": source_run.name,
        "claim_review_run_name": review_run.name,
        "source_snapshot_sha256": stable_hash(source_snapshot),
        "articles": len(article_ids),
        "refs": len(refs),
        "groups": len(groups),
        "concepts": len(concepts),
        "claim_coverage_schema": CLAIM_COVERAGE_SCHEMA,
        "inputs": input_fingerprints,
    }
    acceptance = {
        "schema": "okfolio.c1-materialization-acceptance.v1",
        "status": "pass",
        "materialization_id": materialization_id,
        "expected_groups": group_count,
        "groups": len(groups),
        "refs": len(refs),
        "concepts": len(concepts),
        "completed_groups": len(groups),
        "accepted_groups": len(groups),
        "withheld_groups": 0,
        "missing_refs": 0,
        "claim_review_decision": "pass",
        "claim_coverage_schema": CLAIM_COVERAGE_SCHEMA,
    }
    _write_output(
        output_dir,
        manifest=manifest,
        acceptance=acceptance,
        refs=refs,
        concepts=concepts,
        concept_documents=concept_documents,
    )
    return manifest


def _validate_paths(source_run: Path, review_run: Path, output_dir: Path) -> None:
    for label, path in (("source run", source_run), ("review run", review_run)):
        if not path.is_dir():
            raise C1MaterializationError(f"{label} does not exist: {path}")
        if (
            output_dir == path
            or path in output_dir.parents
            or output_dir in path.parents
        ):
            raise C1MaterializationError(
                "C1 output must be outside both immutable input runs"
            )
    if output_dir.exists():
        raise C1MaterializationError(f"C1 output already exists: {output_dir}")


def _validate_review_manifest(
    manifest: Mapping[str, Any],
    *,
    source_run: Path,
    expected_groups: int,
) -> None:
    if manifest.get("schema") != CLAIM_REVIEW_SCHEMA:
        raise C1MaterializationError("unsupported Claim Review manifest schema")
    if manifest.get("status") != "complete":
        raise C1MaterializationError("Claim Review is partial or incomplete")
    if manifest.get("source_run_name") != source_run.name:
        raise C1MaterializationError("Claim Review points to a different source run")
    configuration = _mapping(manifest.get("configuration"), "review configuration")
    if configuration.get("allow_partial") is not False:
        raise C1MaterializationError("probe/partial Claim Review cannot publish C1")
    if list(configuration.get("selected_group_ids") or []):
        raise C1MaterializationError("selected-group Claim Review cannot publish C1")
    summary = _mapping(manifest.get("summary"), "review summary")
    expected = {
        "groups": expected_groups,
        "completed": expected_groups,
        "accepted": expected_groups,
        "withheld": 0,
        "reviews": 0,
    }
    for key, value in expected.items():
        if _integer(summary.get(key), f"review summary {key}") != value:
            raise C1MaterializationError(
                f"formal Claim Review summary is not publishable: {key}"
            )


def _validate_source_snapshot(
    snapshot: Mapping[str, Any],
    *,
    review_manifest: Mapping[str, Any],
    source_paths: Mapping[str, Path],
    source_progress: Mapping[str, Any],
) -> None:
    if review_manifest.get("source_snapshot_sha256") != stable_hash(snapshot):
        raise C1MaterializationError("Claim Review source snapshot hash mismatch")
    inputs = _mapping(snapshot.get("inputs"), "source snapshot inputs")
    for name, path in source_paths.items():
        recorded = _mapping(inputs.get(name), f"source snapshot {name}")
        current = _file_fingerprint(path)
        if recorded != current:
            raise C1MaterializationError(f"source run changed after review: {name}")
    if list(snapshot.get("provenance_warnings") or []):
        raise C1MaterializationError("Claim Review used provenance fallbacks")
    structures = _mapping(snapshot.get("structures"), "structure snapshot")
    sources = source_progress.get("sources")
    if not isinstance(sources, list) or not sources:
        raise C1MaterializationError("source_progress.sources must be non-empty")
    expected_structures = {
        f"{Path(str(item.get('source') or '')).stem}.structure.json"
        for item in sources
        if isinstance(item, dict) and str(item.get("source") or "").strip()
    }
    if not expected_structures or not expected_structures.issubset(structures):
        raise C1MaterializationError("Claim Review lacks complete structure snapshots")
    for name in expected_structures:
        digest = structures.get(name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise C1MaterializationError(f"invalid structure snapshot: {name}")


def _source_refs(
    source_progress: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    sources = source_progress.get("sources")
    if not isinstance(sources, list) or not sources:
        raise C1MaterializationError("source_progress.sources must be non-empty")
    refs: list[dict[str, Any]] = []
    by_ref: dict[str, dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("refs"), list):
            raise C1MaterializationError("invalid source_progress source entry")
        for raw in source["refs"]:
            if not isinstance(raw, dict):
                raise C1MaterializationError("ConceptRef must be an object")
            ref = dict(raw)
            ref_id = _nonempty(ref.get("ref_id"), "ConceptRef ref_id")
            if ref_id in by_ref:
                raise C1MaterializationError(f"duplicate ConceptRef: {ref_id}")
            _nonempty(ref.get("article_id"), f"ConceptRef article_id: {ref_id}")
            _nonempty(ref.get("source"), f"ConceptRef source: {ref_id}")
            blocks = _string_list(
                ref.get("evidence_block_ids"),
                f"ConceptRef evidence blocks: {ref_id}",
            )
            if not blocks:
                raise C1MaterializationError(
                    f"ConceptRef lacks block provenance: {ref_id}"
                )
            page_start = _positive_int(ref.get("page_start"), f"page_start: {ref_id}")
            page_end = _positive_int(ref.get("page_end"), f"page_end: {ref_id}")
            if page_start > page_end:
                raise C1MaterializationError(f"invalid page range: {ref_id}")
            refs.append(ref)
            by_ref[ref_id] = ref
    return refs, by_ref


def _resolve_group_count(
    payload: Mapping[str, Any],
    *,
    expected_groups: int | None,
) -> int:
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise C1MaterializationError("source run groups must be a non-empty array")
    actual = len(raw_groups)
    if expected_groups is None:
        return actual
    if not isinstance(expected_groups, int) or isinstance(expected_groups, bool):
        raise ValueError("expected_groups must be an integer")
    if expected_groups < 1:
        raise ValueError("expected_groups must be positive")
    if expected_groups != actual:
        raise C1MaterializationError(
            "explicit expected group count mismatch: "
            f"expected={expected_groups}, frozen={actual}"
        )
    return actual


def _source_groups(
    payload: Mapping[str, Any],
    *,
    by_ref: Mapping[str, Mapping[str, Any]],
    expected_groups: int,
) -> list[dict[str, Any]]:
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or len(raw_groups) != expected_groups:
        raise C1MaterializationError(
            f"source run must contain exactly {expected_groups} groups"
        )
    groups: list[dict[str, Any]] = []
    group_ids: set[str] = set()
    assigned: list[str] = []
    for raw in raw_groups:
        if not isinstance(raw, dict):
            raise C1MaterializationError("compile group must be an object")
        group = dict(raw)
        group_id = _nonempty(group.get("group_id"), "group_id")
        if group_id in group_ids:
            raise C1MaterializationError(f"duplicate group_id: {group_id}")
        group_ids.add(group_id)
        ref_ids = _string_list(group.get("ref_ids"), f"group refs: {group_id}")
        if not ref_ids:
            raise C1MaterializationError(f"empty compile group: {group_id}")
        unknown = set(ref_ids) - set(by_ref)
        if unknown:
            raise C1MaterializationError(
                f"compile group contains unknown ConceptRef: {min(unknown)}"
            )
        assigned.extend(ref_ids)
        group["ref_ids"] = ref_ids
        groups.append(group)
    if len(assigned) != len(set(assigned)):
        raise C1MaterializationError("ConceptRef is assigned to multiple groups")
    missing = set(by_ref) - set(assigned)
    if missing:
        raise C1MaterializationError(f"source groups omit ConceptRef: {min(missing)}")
    if set(assigned) - set(by_ref):
        raise C1MaterializationError("source groups contain an unknown ConceptRef")
    return groups


def _validate_reviewed_progress(
    reviewed: Mapping[str, Any],
    *,
    group_ids: Sequence[str],
    review_queue: Mapping[str, Any],
) -> None:
    if reviewed.get("schema") != REVIEWED_PROGRESS_SCHEMA:
        raise C1MaterializationError("unsupported reviewed progress schema")
    expected = set(group_ids)
    completed = _unique_id_set(reviewed.get("completed_groups"), "completed groups")
    accepted = _unique_id_set(reviewed.get("accepted_groups"), "accepted groups")
    withheld = _unique_id_set(reviewed.get("withheld_groups"), "withheld groups")
    if completed != expected:
        raise C1MaterializationError("Claim Review has incomplete groups")
    if accepted != expected:
        raise C1MaterializationError("Claim Review has non-pass groups")
    if withheld:
        raise C1MaterializationError("Claim Review contains withheld groups")
    drafts = _mapping(reviewed.get("drafts"), "reviewed drafts")
    reviews = _mapping(reviewed.get("claim_reviews"), "claim reviews")
    if set(drafts) != expected or set(reviews) != expected:
        raise C1MaterializationError("Claim Review lacks a draft or review for a group")
    queued = review_queue.get("reviews")
    if not isinstance(queued, list) or queued:
        raise C1MaterializationError("Claim Review queue is not empty")


def _validated_claim_review(
    group_id: str,
    ref_ids: Sequence[str],
    refs: Sequence[Mapping[str, Any]],
    review: Mapping[str, Any],
    *,
    draft: Mapping[str, Any],
    seen_claim_ids: set[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if review.get("decision") != "pass":
        raise C1MaterializationError(f"Claim Review did not pass: {group_id}")
    contract = dict(_mapping(review.get("contract"), f"contract {group_id}"))
    coverage = _mapping(review.get("coverage"), f"coverage {group_id}")
    if contract.get("schema_version") != "okfolio.claim-contract.v1":
        raise C1MaterializationError(f"unsupported claim contract: {group_id}")
    if coverage.get("schema_version") != CLAIM_COVERAGE_SCHEMA:
        raise C1MaterializationError(f"unsupported coverage matrix: {group_id}")
    if contract.get("group_id") != group_id:
        raise C1MaterializationError(f"claim contract group mismatch: {group_id}")
    _nonempty(contract.get("canonical_question"), f"canonical question: {group_id}")
    members = contract.get("members")
    if not isinstance(members, list):
        raise C1MaterializationError(f"claim members must be an array: {group_id}")
    member_ids: list[str] = []
    for member in members:
        if not isinstance(member, dict):
            raise C1MaterializationError(f"invalid claim member: {group_id}")
        member_ids.append(_nonempty(member.get("ref_id"), "claim member ref_id"))
        relation = member.get("relation")
        if relation not in {
            "supports",
            "qualifies",
            "contrasts",
            "applies_to",
            "separate",
        }:
            raise C1MaterializationError(
                f"invalid claim member relation: {group_id}"
            )
        if relation == "separate":
            raise C1MaterializationError(
                f"separate member cannot enter a published Concept: {group_id}"
            )
    if len(member_ids) != len(set(member_ids)) or set(member_ids) != set(ref_ids):
        raise C1MaterializationError(f"claim members do not cover group: {group_id}")

    by_ref = {str(item["ref_id"]): item for item in refs}
    raw_claims = contract.get("claims")
    if not isinstance(raw_claims, list) or not raw_claims:
        raise C1MaterializationError(f"claim contract has no claims: {group_id}")
    claims: list[dict[str, Any]] = []
    contributor_refs: set[str] = set()
    claim_ids: list[str] = []
    for raw in raw_claims:
        if not isinstance(raw, dict):
            raise C1MaterializationError(f"invalid claim: {group_id}")
        claim = dict(raw)
        claim_id = _nonempty(claim.get("claim_id"), f"claim_id: {group_id}")
        if claim_id in seen_claim_ids:
            raise C1MaterializationError(f"duplicate claim_id: {claim_id}")
        seen_claim_ids.add(claim_id)
        claim_ids.append(claim_id)
        ref_id = _nonempty(claim.get("ref_id"), f"claim ref_id: {claim_id}")
        if ref_id not in by_ref:
            raise C1MaterializationError(f"claim points outside group: {claim_id}")
        contributor_refs.add(ref_id)
        _nonempty(claim.get("claim"), f"claim text: {claim_id}")
        _nonempty(claim.get("evidence_excerpt"), f"claim evidence: {claim_id}")
        _require_no_warnings(claim, label=f"claim provenance warning: {claim_id}")
        blocks = _string_list(
            claim.get("evidence_block_ids"), f"claim blocks: {claim_id}"
        )
        if not blocks or not set(blocks).issubset(
            set(str(item) for item in by_ref[ref_id]["evidence_block_ids"])
        ):
            raise C1MaterializationError(f"invalid claim block provenance: {claim_id}")
        pages = claim.get("page_numbers")
        if not isinstance(pages, list) or not pages:
            raise C1MaterializationError(f"claim lacks page provenance: {claim_id}")
        page_start = int(by_ref[ref_id]["page_start"])
        page_end = int(by_ref[ref_id]["page_end"])
        if any(
            not isinstance(page, int)
            or isinstance(page, bool)
            or not page_start <= page <= page_end
            for page in pages
        ):
            raise C1MaterializationError(f"invalid claim page provenance: {claim_id}")
        claims.append(claim)
    if contributor_refs != set(ref_ids):
        raise C1MaterializationError(
            f"not every ConceptRef contributes a claim: {group_id}"
        )

    if coverage.get("decision") != "pass":
        raise C1MaterializationError(f"coverage matrix did not pass: {group_id}")
    rows = coverage.get("rows")
    if not isinstance(rows, list):
        raise C1MaterializationError(f"coverage rows must be an array: {group_id}")
    row_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "covered":
            raise C1MaterializationError(f"claim is not covered: {group_id}")
        row_ids.append(_nonempty(row.get("claim_id"), "coverage claim_id"))
        _nonempty(row.get("draft_excerpt"), "coverage draft excerpt")
        _nonempty(row.get("finding"), "coverage finding")
    if len(row_ids) != len(set(row_ids)) or set(row_ids) != set(claim_ids):
        raise C1MaterializationError(f"coverage rows do not match claims: {group_id}")
    _require_empty_array(
        coverage,
        "unsupported_claims",
        label=f"unsupported claims: {group_id}",
    )
    _require_empty_array(
        coverage,
        "scope_violations",
        label=f"scope violations: {group_id}",
    )
    sentence_attributions = _validated_sentence_attributions(
        group_id,
        coverage,
        draft=draft,
        claim_ids=set(claim_ids),
    )
    return contract, claims, sentence_attributions


def _validated_sentence_attributions(
    group_id: str,
    coverage: Mapping[str, Any],
    *,
    draft: Mapping[str, Any],
    claim_ids: set[str],
) -> list[dict[str, Any]]:
    try:
        catalog = build_draft_sentences(draft)
    except ContractError as error:
        raise C1MaterializationError(
            f"invalid draft sentence catalog: {group_id}"
        ) from error
    expected = {item.sentence_id: item.text for item in catalog}
    raw_attributions = coverage.get("sentence_attributions")
    if not isinstance(raw_attributions, list):
        raise C1MaterializationError(
            f"sentence attributions must be an array: {group_id}"
        )
    attributions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_attributions):
        if not isinstance(raw, dict):
            raise C1MaterializationError(
                f"invalid sentence attribution: {group_id}/{index}"
            )
        item = dict(raw)
        sentence_id = _nonempty(
            item.get("sentence_id"),
            f"sentence attribution ID: {group_id}/{index}",
        )
        if sentence_id not in expected:
            raise C1MaterializationError(
                f"unknown sentence attribution: {group_id}/{sentence_id}"
            )
        if sentence_id in seen:
            raise C1MaterializationError(
                f"duplicate sentence attribution: {group_id}/{sentence_id}"
            )
        seen.add(sentence_id)
        if item.get("status") != "supported":
            raise C1MaterializationError(
                f"sentence attribution is not supported: {group_id}/{sentence_id}"
            )
        attributed_claims = _string_list(
            item.get("claim_ids"),
            f"sentence attribution claims: {group_id}/{sentence_id}",
        )
        if not attributed_claims or not set(attributed_claims).issubset(claim_ids):
            raise C1MaterializationError(
                f"invalid sentence attribution claims: {group_id}/{sentence_id}"
            )
        if item.get("draft_excerpt") != expected[sentence_id]:
            raise C1MaterializationError(
                f"sentence attribution does not match catalog: "
                f"{group_id}/{sentence_id}"
            )
        _nonempty(
            item.get("finding"),
            f"sentence attribution finding: {group_id}/{sentence_id}",
        )
        _require_no_warnings(
            item,
            label=f"sentence attribution warning: {group_id}/{sentence_id}",
        )
        attributions.append(item)
    if seen != set(expected):
        raise C1MaterializationError(
            f"sentence attributions do not cover catalog: {group_id}"
        )
    return attributions


def _require_no_warnings(value: Mapping[str, Any], *, label: str) -> None:
    for field in ("source_text_anomalies", "ocr_suspicions"):
        _require_empty_array(value, field, label=label)


def _require_empty_array(
    value: Mapping[str, Any],
    field: str,
    *,
    label: str,
) -> None:
    items = value.get(field)
    if not isinstance(items, list):
        raise C1MaterializationError(f"{label} lacks {field}")
    if items:
        raise C1MaterializationError(f"{label} contains {field}")


def _draft_content(
    group_id: str,
    draft: Mapping[str, Any],
    refs: Sequence[Mapping[str, Any]],
) -> tuple[str, str, str, str]:
    title = _nonempty(draft.get("title"), f"draft title: {group_id}")
    description = _nonempty(
        draft.get("description"), f"draft description: {group_id}"
    )
    body = _nonempty(draft.get("body"), f"draft body: {group_id}")
    raw_ref = _mapping(draft.get("ref"), f"draft ref: {group_id}")
    if raw_ref.get("concept_id") != group_id:
        raise C1MaterializationError(f"draft Concept ID mismatch: {group_id}")
    ref_types = {_nonempty(item.get("type"), "ConceptRef type") for item in refs}
    if len(ref_types) != 1:
        raise C1MaterializationError(f"mixed ConceptRef types: {group_id}")
    concept_type = _nonempty(raw_ref.get("type"), f"draft type: {group_id}")
    if concept_type not in ref_types:
        raise C1MaterializationError(f"draft type mismatch: {group_id}")
    return title, description, body, concept_type


def _source_location(ref: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ref_id": ref["ref_id"],
        "article_id": ref["article_id"],
        "source": ref["source"],
        "section_path": list(ref.get("section_path") or []),
        "page_start": ref["page_start"],
        "page_end": ref["page_end"],
        "evidence_block_ids": list(ref["evidence_block_ids"]),
        "scope": dict(ref.get("scope") or {}),
    }


def _write_output(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    refs: Sequence[Mapping[str, Any]],
    concepts: Sequence[Mapping[str, Any]],
    concept_documents: Mapping[str, str],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent)
    )
    try:
        concept_dir = stage / "concepts"
        concept_dir.mkdir()
        _write_json(stage / "manifest.json", manifest)
        _write_json(stage / "acceptance.json", acceptance)
        _write_json(
            stage / "refs.json",
            {"schema": "okfolio.c1-refs.v1", "refs": list(refs)},
        )
        _write_json(
            stage / "concepts.json",
            {"schema": "okfolio.c1-concepts.v1", "concepts": list(concepts)},
        )
        for filename, content in concept_documents.items():
            (concept_dir / filename).write_text(content, encoding="utf-8")
        if output_dir.exists():
            raise C1MaterializationError(f"C1 output already exists: {output_dir}")
        os.replace(stage, output_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise C1MaterializationError(f"invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise C1MaterializationError(f"JSON input must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size": path.stat().st_size}


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C1MaterializationError(f"{label} must be an object")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise C1MaterializationError(f"{label} must be non-empty")
    return value.strip()


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise C1MaterializationError(f"{label} must be a string array")
    result = [item.strip() for item in value]
    if len(result) != len(set(result)):
        raise C1MaterializationError(f"{label} contains duplicates")
    return result


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise C1MaterializationError(f"{label} must be a positive integer")
    return value


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise C1MaterializationError(f"{label} must be an integer")
    return value


def _concept_filename(group_id: str) -> str:
    candidate = f"{group_id}.md"
    try:
        normalized = normalize_slug(candidate)
    except OKFValidationError as error:
        raise C1MaterializationError(
            f"unsafe Concept filename for group: {group_id}"
        ) from error
    if normalized != candidate:
        raise C1MaterializationError(
            f"group_id is not a normalized Concept slug: {group_id}"
        )
    return candidate


def _unique_id_set(value: Any, label: str) -> set[str]:
    values = _string_list(value, label)
    return set(values)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a complete Claim Review as a standard C1 run."
    )
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--review-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-groups",
        type=int,
        help=(
            "Optional dataset-specific assertion; by default the complete "
            "group count is derived from frozen groups.json."
        ),
    )
    args = parser.parse_args(argv)
    try:
        result = materialize_c1_run(
            source_run=args.source_run,
            review_run=args.review_run,
            output_dir=args.output,
            expected_groups=args.expected_groups,
        )
    except (C1MaterializationError, OSError, ValueError) as error:
        print(f"C1 materialization refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "C1MaterializationError",
    "C1_MATERIALIZATION_SCHEMA",
    "CLAIM_COVERAGE_SCHEMA",
    "materialize_c1_run",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
