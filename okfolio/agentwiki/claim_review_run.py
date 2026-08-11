"""Resumable, source-immutable Claim Review orchestration.

The source AgentWiki run is treated as a frozen input snapshot.  This module
builds a claim contract from ConceptRefs only, reuses or creates one draft,
audits claim coverage, and (when needed) performs at most two targeted
recompiles.  Every group has an atomic checkpoint so a resumed process does
not repeat a completed model stage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .agent_contracts import CompileGroup, QualityAudit
from .agentic import AgentRefRecord, _group_ref, recompile_concept
from .claim_review import (
    ClaimCoverageBatch,
    ClaimCoverageMatrix,
    ConceptClaimContract,
    build_draft_sentences,
    build_evidence_units,
    claim_contract_json_schema,
    claim_coverage_batch_json_schema,
    chunk_draft_sentences,
    merge_claim_coverage_batches,
    normalize_known_source_anomalies,
    parse_claim_contract,
    parse_claim_coverage,
    parse_claim_coverage_batch,
)
from .contracts import ConceptRef, DraftConcept
from .stages import _complete_structured, compile_one_concept, render_prompt
from .state import _write_json_atomic, stable_hash


SOURCE_FILES = (
    "source_progress.json",
    "groups.json",
    "compile_progress.json",
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
class ClaimReviewStageClients:
    """Route Claim Review schemas to independently configured clients.

    Evidence interpretation benefits from deliberate reasoning, while draft
    rendering should remain terse and bounded.  Routing by the schema name
    lets a local runtime use different chat-template thinking policies without
    weakening the frozen run configuration or changing the public client
    protocol.
    """

    contract: CompletionClient
    coverage: CompletionClient
    compile: CompletionClient
    recompile: CompletionClient

    def complete(
        self,
        prompt: str,
        *,
        json_schema_name: str | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> str:
        stage_by_schema = {
            "concept_claim_contract": self.contract,
            "concept_claim_coverage": self.coverage,
            "concept_draft": self.compile,
            "agent_recompiled_concept": self.recompile,
        }
        client = stage_by_schema.get(str(json_schema_name))
        if client is None:
            raise ClaimReviewRunError(
                f"no Claim Review client route for schema: {json_schema_name}"
            )
        return client.complete(
            prompt,
            json_schema_name=json_schema_name,
            json_schema=json_schema,
        )


class ClaimReviewRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClaimReviewTemplates:
    contract: str
    coverage: str
    compile: str
    recompile: str


def _plain_client_configuration(client: CompletionClient) -> dict[str, Any]:
    return {
        "model": str(getattr(client, "model", type(client).__name__)),
        "max_tokens": getattr(client, "max_tokens", None),
        "response_format": getattr(client, "response_format", None),
        "send_chat_template_kwargs": bool(
            getattr(client, "send_chat_template_kwargs", False)
        ),
        "enable_thinking": bool(getattr(client, "enable_thinking", False)),
    }


def _client_configuration(client: CompletionClient) -> dict[str, Any]:
    if isinstance(client, ClaimReviewStageClients):
        return {
            "routing": "schema_stage",
            "stages": {
                "contract": _plain_client_configuration(client.contract),
                "coverage": _plain_client_configuration(client.coverage),
                "compile": _plain_client_configuration(client.compile),
                "recompile": _plain_client_configuration(client.recompile),
            },
        }
    return _plain_client_configuration(client)


def _normalize_client_configuration(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if value.get("routing") == "schema_stage":
        stages = value.get("stages")
        if not isinstance(stages, dict):
            return value
        return {
            **value,
            "stages": {
                stage: _normalize_client_configuration(stages.get(stage))
                for stage in ("contract", "coverage", "compile", "recompile")
            },
        }
    return {
        **value,
        "send_chat_template_kwargs": bool(
            value.get("send_chat_template_kwargs", False)
        ),
        "enable_thinking": bool(value.get("enable_thinking", False)),
    }


def _seed_client_policy_compatible(seed: Any, target: Any) -> tuple[bool, bool, bool]:
    """Return ``(compatible, model_relaxed, thinking_relaxed)`` for reusing audited passes.

    Three mechanisms may legitimately evolve between audited passes and are
    therefore not compared strictly: the served ``model`` id (the same weights
    may be exposed under a different id by a different runtime), the
    ``enable_thinking`` flag of every stage (a different serving stack may
    produce shorter or cleaner traces, so a run may switch thinking on or off
    wholesale), and the Coverage stage's ``max_tokens`` (the coverage
    mechanism may change; target must be >= seed).  Every other stage-client
    field (``response_format``, ``send_chat_template_kwargs``, and
    ``max_tokens`` for contract/compile/recompile) must match exactly.
    ``model_relaxed``/``thinking_relaxed`` report the observed drifts so the
    manifest can record them.
    """
    if not isinstance(seed, dict) or not isinstance(target, dict):
        return seed == target, False, False
    if seed.get("routing") != "schema_stage" or target.get("routing") != "schema_stage":
        seed_flat = dict(seed)
        target_flat = dict(target)
        model_relaxed = seed_flat.pop("model", None) != target_flat.pop(
            "model", None
        )
        thinking_relaxed = seed_flat.pop(
            "enable_thinking", None
        ) != target_flat.pop("enable_thinking", None)
        return seed_flat == target_flat, model_relaxed, thinking_relaxed
    seed_stages = seed.get("stages")
    target_stages = target.get("stages")
    if not isinstance(seed_stages, dict) or not isinstance(target_stages, dict):
        return False, False, False
    model_relaxed = False
    thinking_relaxed = False
    for stage in ("contract", "compile", "recompile"):
        seed_stage = dict(seed_stages.get(stage) or {})
        target_stage = dict(target_stages.get(stage) or {})
        if seed_stage.pop("model", None) != target_stage.pop("model", None):
            model_relaxed = True
        if seed_stage.pop("enable_thinking", None) != target_stage.pop(
            "enable_thinking", None
        ):
            thinking_relaxed = True
        if seed_stage != target_stage:
            return False, model_relaxed, thinking_relaxed
    seed_coverage = dict(seed_stages.get("coverage") or {})
    target_coverage = dict(target_stages.get("coverage") or {})
    if seed_coverage.pop("enable_thinking", None) != target_coverage.pop(
        "enable_thinking", None
    ):
        thinking_relaxed = True
    if seed_coverage.pop("model", None) != target_coverage.pop("model", None):
        model_relaxed = True
    seed_max_tokens = seed_coverage.pop("max_tokens", None)
    target_max_tokens = target_coverage.pop("max_tokens", None)
    if seed_coverage != target_coverage:
        return False, model_relaxed, thinking_relaxed
    if isinstance(seed_max_tokens, int) and isinstance(target_max_tokens, int):
        return target_max_tokens >= seed_max_tokens, model_relaxed, thinking_relaxed
    return seed_max_tokens == target_max_tokens, model_relaxed, thinking_relaxed


def run_claim_review(
    client: CompletionClient,
    *,
    source_run: Path,
    output_dir: Path,
    templates: ClaimReviewTemplates,
    structures_dir: Path | None = None,
    resume: bool = False,
    allow_partial: bool = False,
    max_recompile_attempts: int = 2,
    selected_group_ids: Sequence[str] = (),
    known_source_anomalies: Sequence[str] = (),
    seed_run: Path | None = None,
    coverage_batch_size: int = 12,
    draft_overrides: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Review an AgentWiki run without modifying it.

    ``allow_partial`` is the explicit probe mode.  It permits missing structure
    blocks, but records them in ``review_queue.json``; formal mode refuses
    provenance fallback.  Both modes process every selected group: a group
    whose model calls exhaust their retries (persistent ContractError or
    truncation) is recorded as a failed checkpoint and withheld from
    publication instead of aborting the run.  Systemic errors — configuration
    mismatches, source or structure snapshot drift, missing inputs — are
    raised before the group loop and still abort the run.  A formal run that
    withholds any group finishes with a non-zero exit and a ``needs_review``
    manifest so an operator can inspect ``review_queue.json``.

    Coverage audits the draft in deterministic contiguous sentence batches of
    at most ``coverage_batch_size`` sentences.  Each batch reuses the frozen
    Claim Contract, is persisted atomically, and is skipped on resume when its
    batch index already exists for the same draft hash.

    ``draft_overrides`` maps a group ID to a manually repaired draft payload
    ``{"title", "description", "body"}`` (the same shape ``_draft_payload``
    produces).  Overrides are frozen in the run configuration, never touch the
    source run, and are consumed in ``_review_group`` only when the checkpoint
    has no persisted draft (checkpoint drafts always win on resume).  For a
    seed checkpoint reopened as ``running`` (a budget-exhausted human_review
    group), an override replaces the inherited draft and invalidates the
    inherited Coverage so the repaired draft is audited from scratch.
    """
    if not 0 <= max_recompile_attempts <= 2:
        raise ValueError("max_recompile_attempts must be between 0 and 2")
    if (
        isinstance(coverage_batch_size, bool)
        or not isinstance(coverage_batch_size, int)
        or coverage_batch_size < 1
    ):
        raise ValueError("coverage_batch_size must be a positive integer")
    selection = tuple(dict.fromkeys(str(item).strip() for item in selected_group_ids))
    if any(not item for item in selection):
        raise ValueError("selected group IDs must be non-empty")
    if selection and not allow_partial:
        raise ClaimReviewRunError("group selection is probe-only; add allow_partial")
    anomalies = normalize_known_source_anomalies(known_source_anomalies)
    source_run = source_run.resolve()
    output_dir = output_dir.resolve()
    seed_run = seed_run.resolve() if seed_run is not None else None
    if seed_run is not None:
        if not seed_run.is_dir():
            raise ClaimReviewRunError(f"seed run does not exist: {seed_run}")
        if seed_run == output_dir:
            raise ClaimReviewRunError("seed run and output run must be different")
    _validate_paths(source_run, output_dir, resume=resume)
    structures_dir = (
        structures_dir.resolve()
        if structures_dir is not None
        else (source_run.parent.parent / "normalized-sources").resolve()
    )

    source_paths = {name: source_run / name for name in SOURCE_FILES}
    missing_inputs = [name for name, path in source_paths.items() if not path.is_file()]
    if missing_inputs:
        raise ClaimReviewRunError(
            "source run is missing required inputs: " + ", ".join(missing_inputs)
        )
    snapshot = _source_snapshot(source_run, source_paths, structures_dir)
    if not resume:
        output_dir.mkdir(parents=True)

    source_progress = _read_object(source_paths["source_progress.json"])
    groups_payload = _read_object(source_paths["groups.json"])
    compile_progress = _read_object(source_paths["compile_progress.json"])
    raw_refs = _refs_from_source_progress(source_progress)
    all_groups = _groups_from_payload(groups_payload, raw_refs)
    by_group_id = {item.group_id: item for item in all_groups}
    if draft_overrides is None:
        draft_overrides = {}
    elif not isinstance(draft_overrides, Mapping):
        raise ClaimReviewRunError("draft_overrides must be a mapping")
    else:
        draft_overrides = dict(draft_overrides)
    for override_group_id, payload in draft_overrides.items():
        if not isinstance(override_group_id, str) or not override_group_id.strip():
            raise ClaimReviewRunError(
                "draft override group IDs must be non-empty strings"
            )
        if override_group_id not in by_group_id:
            raise ClaimReviewRunError(
                f"draft override targets an unknown group: {override_group_id}"
            )
        if not isinstance(payload, Mapping):
            raise ClaimReviewRunError(
                f"draft override for {override_group_id} must be an object"
            )
        for field in ("title", "description", "body"):
            value = payload.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ClaimReviewRunError(
                    f"draft override for {override_group_id} must provide "
                    f"a non-empty {field}"
                )
    unknown_selection = set(selection) - set(by_group_id)
    if unknown_selection:
        raise ClaimReviewRunError(
            f"unknown selected group: {min(unknown_selection)}"
        )
    groups = (
        tuple(by_group_id[group_id] for group_id in selection)
        if selection
        else all_groups
    )
    source_drafts = compile_progress.get("drafts")
    if not isinstance(source_drafts, dict):
        raise ClaimReviewRunError("compile_progress.json drafts must be an object")

    refs_with_blocks, provenance_warnings, structure_hashes = _attach_evidence_blocks(
        raw_refs,
        structures_dir=structures_dir,
        allow_partial=allow_partial,
    )
    snapshot_with_structures = {
        **snapshot,
        "structures": structure_hashes,
        "provenance_warnings": provenance_warnings,
    }
    if resume:
        frozen = _read_object(output_dir / "source_snapshot.json")
        if _snapshot_integrity(frozen) != _snapshot_integrity(
            snapshot_with_structures
        ):
            raise ClaimReviewRunError(
                "source or structure snapshot changed; refusing to resume"
            )
    else:
        _write_json_atomic(
            output_dir / "source_snapshot.json", snapshot_with_structures
        )
    snapshot = snapshot_with_structures

    prompt_configuration = {
        "contract": _text_sha256(templates.contract),
        "coverage": _text_sha256(templates.coverage),
        "compile": _text_sha256(templates.compile),
        "recompile": _text_sha256(templates.recompile),
    }
    client_configuration = _client_configuration(client)

    seeded_checkpoints: dict[str, dict[str, Any]] = {}
    seed_configuration: dict[str, Any] | None = None
    contract_prompt_relaxed = False
    compile_prompt_relaxed = False
    recompile_prompt_relaxed = False
    coverage_prompt_relaxed = False
    model_relaxed = False
    thinking_relaxed = False
    if seed_run is not None:
        frozen_seed_snapshot = _read_object(seed_run / "source_snapshot.json")
        if _snapshot_integrity(frozen_seed_snapshot) != _snapshot_integrity(
            snapshot
        ):
            raise ClaimReviewRunError(
                "seed run uses a different source or structure snapshot"
            )
        seed_hashes: dict[str, str] = {}
        seed_skipped: dict[str, str] = {}
        for group in groups:
            seed_checkpoint_path = (
                seed_run
                / "checkpoints"
                / f"{_checkpoint_name(group.group_id)}.json"
            )
            if not seed_checkpoint_path.is_file():
                continue
            try:
                seeded, checkpoint_hash = _prepare_seed_checkpoint(
                    seed_run,
                    group=group,
                    refs=refs_with_blocks,
                    known_source_anomalies=anomalies,
                    draft_overrides=draft_overrides,
                )
            except ClaimReviewRunError as error:
                # A partial seed may contain the checkpoint that was executing
                # when its process stopped.  It is safer to rerun that group
                # than to reject every otherwise reusable checkpoint.
                seed_skipped[group.group_id] = str(error)
                continue
            seeded_checkpoints[group.group_id] = seeded
            seed_hashes[group.group_id] = checkpoint_hash
        if not seeded_checkpoints:
            raise ClaimReviewRunError(
                "seed run has no checkpoint matching the selected groups"
            )
        if any(
            checkpoint.get("status") == "complete"
            for checkpoint in seeded_checkpoints.values()
        ):
            seed_manifest = _read_object(seed_run / "manifest.json")
            seed_runtime_configuration = seed_manifest.get("configuration")
            if not isinstance(seed_runtime_configuration, dict):
                raise ClaimReviewRunError("complete seed has no frozen configuration")
            if list(seed_runtime_configuration.get("known_source_anomalies") or ()) != list(
                anomalies
            ):
                raise ClaimReviewRunError(
                    "complete seed uses a different source anomaly policy"
                )
            # Every prompt stage may evolve between audited passes: a repair
            # run strengthens the Contract/Recompile prompts while inheriting
            # the already audited pass checkpoints.  The seed checkpoints
            # themselves were produced under the frozen seed prompts and stay
            # valid; each observed drift is recorded in the manifest.  The
            # strictness that protects audited passes (source anomaly policy,
            # stage client policy with model/thinking relaxed and max_tokens
            # still strict, frozen source snapshot) is unchanged.
            seed_prompt_hashes = seed_runtime_configuration.get("prompt_sha256")
            if not isinstance(seed_prompt_hashes, dict) or set(
                seed_prompt_hashes
            ) != set(prompt_configuration):
                raise ClaimReviewRunError("complete seed uses different prompts")
            contract_prompt_relaxed = bool(
                seed_prompt_hashes.get("contract")
                != prompt_configuration["contract"]
            )
            compile_prompt_relaxed = bool(
                seed_prompt_hashes.get("compile")
                != prompt_configuration["compile"]
            )
            recompile_prompt_relaxed = bool(
                seed_prompt_hashes.get("recompile")
                != prompt_configuration["recompile"]
            )
            coverage_prompt_relaxed = bool(
                seed_prompt_hashes.get("coverage")
                != prompt_configuration["coverage"]
            )
            compatible, model_relaxed, thinking_relaxed = (
                _seed_client_policy_compatible(
                    _normalize_client_configuration(
                        seed_runtime_configuration.get("client")
                    ),
                    client_configuration,
                )
            )
            if not compatible:
                raise ClaimReviewRunError(
                    "complete seed uses a different stage client policy"
                )
        seed_configuration = {
            "run_name": seed_run.name,
            "source_snapshot_sha256": stable_hash(frozen_seed_snapshot),
            "checkpoint_sha256": seed_hashes,
            "skipped_checkpoints": seed_skipped,
            "contract_prompt_relaxed": contract_prompt_relaxed,
            "compile_prompt_relaxed": compile_prompt_relaxed,
            "recompile_prompt_relaxed": recompile_prompt_relaxed,
            "coverage_prompt_relaxed": coverage_prompt_relaxed,
            "model_relaxed": model_relaxed,
            "thinking_relaxed": thinking_relaxed,
        }

    configuration = {
        "allow_partial": allow_partial,
        "known_source_anomalies": list(anomalies),
        "max_recompile_attempts": max_recompile_attempts,
        "selected_group_ids": list(selection),
        "client": client_configuration,
        "seed": seed_configuration,
        "prompt_sha256": prompt_configuration,
        "coverage_batch_size": coverage_batch_size,
        "draft_overrides": draft_overrides,
    }
    if resume:
        existing_manifest = _read_object(output_dir / "manifest.json")
        existing_configuration = existing_manifest.get("configuration")
        if isinstance(existing_configuration, dict):
            existing_client = _normalize_client_configuration(
                existing_configuration.get("client")
            )
            existing_configuration = {
                **existing_configuration,
                "known_source_anomalies": list(
                    existing_configuration.get("known_source_anomalies") or ()
                ),
                "client": existing_client,
                "seed": existing_configuration.get("seed"),
            }
        if existing_configuration != configuration:
            raise ClaimReviewRunError(
                "review configuration changed; start a new output run"
            )
    manifest = {
        "schema": "okfolio.claim-review-run.v1",
        "status": "running",
        "source_run_name": source_run.name,
        "source_snapshot_sha256": stable_hash(snapshot),
        "configuration": configuration,
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)

    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    if not resume:
        for group_id, checkpoint in seeded_checkpoints.items():
            _write_json_atomic(
                checkpoints / f"{_checkpoint_name(group_id)}.json",
                checkpoint,
            )
    runtime_reviews: list[dict[str, Any]] = [
        {
            "kind": "provenance_fallback",
            **warning,
        }
        for warning in provenance_warnings
    ]
    for position, group in enumerate(groups, start=1):
        checkpoint_path = checkpoints / f"{_checkpoint_name(group.group_id)}.json"
        checkpoint = (
            _read_object(checkpoint_path) if checkpoint_path.is_file() else {}
        )
        if checkpoint.get("status") == "complete":
            continue
        try:
            _review_group(
                client,
                group=group,
                refs=refs_with_blocks,
                source_draft=source_drafts.get(group.group_id),
                templates=templates,
                checkpoint_path=checkpoint_path,
                checkpoint=checkpoint,
                max_recompile_attempts=max_recompile_attempts,
                known_source_anomalies=anomalies,
                coverage_batch_size=coverage_batch_size,
                draft_overrides=draft_overrides,
            )
        except Exception as error:
            latest = (
                _read_object(checkpoint_path) if checkpoint_path.is_file() else checkpoint
            )
            failed = {
                **latest,
                "schema": "okfolio.claim-review-group.v1",
                "group_id": group.group_id,
                "ref_ids": list(group.ref_ids),
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
            _write_json_atomic(checkpoint_path, failed)
            # A group-level failure (e.g. a model that keeps violating the
            # Claim Contract after all retries) is recorded as a failed
            # checkpoint and the run continues with the remaining groups.  The
            # finalize pass reports the group in review_queue.json with
            # decision=failed and withholds it from publication.  Systemic
            # errors (configuration, snapshots, structure provenance) are
            # still raised before the loop and abort the run.
        _write_progress(output_dir, groups, position=position)

    _verify_snapshot(source_paths, snapshot, structures_dir=structures_dir)
    result = _finalize_outputs(
        output_dir,
        groups,
        source_drafts,
        runtime_reviews,
        manifest,
    )
    if result["status"] != "complete" and not allow_partial:
        raise ClaimReviewRunError(
            "formal Claim Review did not pass every group; inspect review_queue.json"
        )
    return result


def _review_group(
    client: CompletionClient,
    *,
    group: CompileGroup,
    refs: Mapping[str, Mapping[str, Any]],
    source_draft: Any,
    templates: ClaimReviewTemplates,
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    max_recompile_attempts: int,
    known_source_anomalies: Sequence[str],
    coverage_batch_size: int = 12,
    draft_overrides: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    group_refs = tuple(refs[ref_id] for ref_id in group.ref_ids)
    checkpoint = {
        **checkpoint,
        "schema": "okfolio.claim-review-group.v1",
        "group_id": group.group_id,
        "ref_ids": list(group.ref_ids),
        "status": "running",
    }
    checkpoint.pop("error", None)

    raw_contract = checkpoint.get("contract")
    if isinstance(raw_contract, dict):
        contract = _contract_from_payload(
            raw_contract,
            known_source_anomalies=known_source_anomalies,
        )
    else:
        units = build_evidence_units(
            group_refs,
            known_source_anomalies=known_source_anomalies,
        )
        prompt = render_prompt(
            templates.contract,
            group=_json(asdict(group)),
            concept_refs=_json([_contract_ref_payload(item) for item in group_refs]),
            evidence_units=_json(
                [
                    {
                        "evidence_id": item.evidence_id,
                        "ref_id": item.ref_id,
                        "text": item.text,
                    }
                    for item in units
                ]
            ),
            known_source_anomalies=_json(list(known_source_anomalies)),
        )
        contract = _complete_structured(
            client,
            prompt,
            schema_name="concept_claim_contract",
            schema=claim_contract_json_schema(
                group_refs,
                known_source_anomalies=known_source_anomalies,
            ),
            parser=lambda response: parse_claim_contract(
                response,
                group_id=group.group_id,
                refs=group_refs,
                known_source_anomalies=known_source_anomalies,
            ),
            max_attempts=3,
        )
        checkpoint["contract"] = contract.to_payload()
        checkpoint["evidence_provenance"] = _evidence_provenance(units)
        _write_json_atomic(checkpoint_path, checkpoint)

    synthetic = _synthetic_ref(group, group_refs, contract)
    raw_draft = checkpoint.get("draft")
    override = None if draft_overrides is None else draft_overrides.get(
        group.group_id
    )
    if isinstance(raw_draft, dict):
        # A persisted checkpoint draft is authoritative: on resume (or after a
        # seeded reopen) the override must not be injected a second time.
        draft = _draft_from_payload(raw_draft)
    elif override is not None:
        draft = DraftConcept(
            ref=synthetic,
            title=override["title"],
            description=override["description"],
            body=override["body"],
        )
        checkpoint["draft"] = _draft_payload(draft)
        checkpoint["draft_origin"] = "repair_override"
        checkpoint["recompile_attempts"] = 0
        _write_json_atomic(checkpoint_path, checkpoint)
    elif isinstance(source_draft, dict):
        draft = _draft_from_payload(source_draft)
        checkpoint["draft"] = _draft_payload(draft)
        checkpoint["draft_origin"] = "source_run"
        checkpoint["recompile_attempts"] = 0
        _write_json_atomic(checkpoint_path, checkpoint)
    else:
        draft = compile_one_concept(client, templates.compile, synthetic)
        checkpoint["draft"] = _draft_payload(draft)
        checkpoint["draft_origin"] = "claim_contract_required_excerpts"
        checkpoint["recompile_attempts"] = 0
        _write_json_atomic(checkpoint_path, checkpoint)

    while True:
        draft_hash = stable_hash(_draft_payload(draft))
        raw_matrix = checkpoint.get("coverage")
        if (
            isinstance(raw_matrix, dict)
            and raw_matrix.get("schema_version") == "okfolio.claim-coverage.v2"
            and checkpoint.get("coverage_draft_sha256") == draft_hash
        ):
            matrix = _matrix_from_payload(
                raw_matrix,
                known_source_anomalies=known_source_anomalies,
            )
        else:
            draft_payload = {
                "title": draft.title,
                "description": draft.description,
                "body": draft.body,
            }
            draft_sentences = build_draft_sentences(draft_payload)
            claim_ids = tuple(item.claim_id for item in contract.claims)
            batch_records = checkpoint.get("coverage_batches")
            if not isinstance(batch_records, dict) or checkpoint.get(
                "coverage_draft_sha256"
            ) != draft_hash:
                # Fresh draft (or first run): start from an empty batch table.
                # The draft hash guard invalidates every batch record as soon
                # as the audited draft changes.
                batch_records = {}
            batch_sets = chunk_draft_sentences(
                draft_sentences,
                batch_size=coverage_batch_size,
            )
            for index, batch_sentences in enumerate(batch_sets):
                key = str(index)
                if key in batch_records:
                    # This batch was already persisted atomically; do not
                    # repeat its model call on resume.
                    continue
                prompt = render_prompt(
                    templates.coverage,
                    claim_contract=_json(contract.to_payload()),
                    draft=_json(draft_payload),
                    draft_sentences=_json(
                        [asdict(sentence) for sentence in batch_sentences]
                    ),
                    known_source_anomalies=_json(list(known_source_anomalies)),
                )
                parsed = _complete_structured(
                    client,
                    prompt,
                    schema_name="concept_claim_coverage",
                    schema=claim_coverage_batch_json_schema(
                        claim_ids,
                        batch_sentences,
                    ),
                    parser=lambda response, index=index, batch_sentences=batch_sentences: parse_claim_coverage_batch(
                        response,
                        contract=contract,
                        batch_index=index,
                        batch_sentences=batch_sentences,
                    ),
                    max_attempts=3,
                )
                batch_records[key] = parsed.to_payload()
                checkpoint["coverage_batches"] = batch_records
                checkpoint["coverage_draft_sha256"] = draft_hash
                _write_json_atomic(checkpoint_path, checkpoint)
            matrix = merge_claim_coverage_batches(
                {
                    int(key): _coverage_batch_from_payload(record)
                    for key, record in batch_records.items()
                },
                contract=contract,
                draft=draft_payload,
                known_source_anomalies=known_source_anomalies,
            )
            history = checkpoint.get("coverage_history")
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "draft_sha256": draft_hash,
                    "matrix": matrix.to_payload(),
                }
            )
            checkpoint["coverage_history"] = history
            checkpoint["coverage"] = matrix.to_payload()
            checkpoint["coverage_draft_sha256"] = draft_hash
            checkpoint["decision"] = matrix.decision
            _write_json_atomic(checkpoint_path, checkpoint)

        if matrix.decision != "recompile":
            checkpoint["decision"] = matrix.decision
            checkpoint["status"] = "complete"
            _write_json_atomic(checkpoint_path, checkpoint)
            return

        attempts = int(checkpoint.get("recompile_attempts") or 0)
        if attempts >= max_recompile_attempts:
            checkpoint["decision"] = "human_review"
            checkpoint["review_reason"] = "recompile_budget_exhausted"
            checkpoint["status"] = "complete"
            _write_json_atomic(checkpoint_path, checkpoint)
            return
        audit = _quality_audit_for_recompile(matrix, contract)
        draft = recompile_concept(
            client,
            templates.recompile,
            ref=synthetic,
            previous=draft,
            audit=audit,
        )
        checkpoint["draft"] = _draft_payload(draft)
        checkpoint["recompile_attempts"] = attempts + 1
        checkpoint.pop("coverage", None)
        checkpoint.pop("coverage_draft_sha256", None)
        checkpoint.pop("coverage_batches", None)
        checkpoint.pop("decision", None)
        _write_json_atomic(checkpoint_path, checkpoint)


def _synthetic_ref(
    group: CompileGroup,
    refs: Sequence[Mapping[str, Any]],
    contract: ConceptClaimContract,
) -> ConceptRef:
    records = {_agent_ref(item).ref_id: _agent_ref(item) for item in refs}
    base = _group_ref(group, records)
    excerpts = tuple(
        dict.fromkeys(item.evidence_excerpt for item in contract.claims)
    )
    if not excerpts:
        raise ClaimReviewRunError("claim contract has no required excerpts")
    return replace(
        base,
        evidence=excerpts,
        evidence_block_ids=tuple(
            dict.fromkeys(
                block_id
                for claim in contract.claims
                for block_id in claim.evidence_block_ids
            )
        ),
    )


def _quality_audit_for_recompile(
    matrix: ClaimCoverageMatrix,
    contract: ConceptClaimContract,
) -> QualityAudit:
    claims = {item.claim_id: item for item in contract.claims}
    issues: list[str] = [
        json.dumps(
            {
                "kind": "all_required_claims_checklist",
                "claims": [
                    {"claim_id": item.claim_id, "claim": item.claim}
                    for item in contract.claims
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    ]
    for row in matrix.rows:
        if row.status != "covered":
            claim = claims.get(row.claim_id)
            issue: dict[str, Any] = {
                "kind": "claim_coverage_defect",
                "claim_id": row.claim_id,
                "status": row.status,
                "draft_excerpt": row.draft_excerpt,
                "coverage_finding": row.finding,
            }
            if claim is not None:
                issue.update(
                    {
                        "claim": claim.claim,
                        "scope": dict(claim.scope),
                    }
                )
            issues.append(_json(issue))
    issues.extend(
        _json(
            {
                "kind": "unsupported_draft_sentence",
                "draft_excerpt": item.draft_excerpt,
                "finding": item.finding,
            }
        )
        for item in matrix.unsupported_claims
    )
    issues.extend(
        _json(
            {
                "kind": "scope_violation",
                "claim_ids": list(item.claim_ids),
                "draft_excerpt": item.draft_excerpt,
                "finding": item.finding,
            }
        )
        for item in matrix.scope_violations
    )
    instructions = (
        "先把 all_required_claims_checklist 作为不可删减的完整保留清单；"
        "未被具体 defect 点名且已经正确支持的草稿句必须原样保留，不得因修复"
        "局部问题而删减。逐项执行其余定向修订：对 claim_coverage_defect，"
        "必须依据其中的 claim 和当前 ConceptRef 的逐字 evidence 补写或改写"
        "对应事实；完整政策名、"
        "数字、精确时间、主体和适用范围必须逐字保留，不得只写简称。对 "
        "unsupported_draft_sentence，删除给出的完整 draft_excerpt，或仅使用"
        "冻结 Claim Contract 中有逐字证据的事实重写。对 scope_violation，按"
        "列出的 claim scope 收窄表述。不得保留审计指出的原句，不得增加合同外事实。"
    )
    return QualityAudit(
        score=0.0,
        decision="recompile",
        issues=tuple(issues) or ("Claim Review 要求定向重编。",),
        recompile_instructions=instructions,
    )


def _attach_evidence_blocks(
    refs: Mapping[str, Mapping[str, Any]],
    *,
    structures_dir: Path,
    allow_partial: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    structures: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    warnings: list[dict[str, Any]] = []
    result: dict[str, dict[str, Any]] = {}
    for ref_id, raw in refs.items():
        source = str(raw.get("source") or "")
        structure_name = f"{Path(source).stem}.structure.json"
        structure_path = structures_dir / structure_name
        if structure_name not in structures:
            if structure_path.is_file():
                payload = _read_object(structure_path)
                blocks = payload.get("blocks")
                if not isinstance(blocks, list):
                    raise ClaimReviewRunError(
                        f"structure blocks must be an array: {structure_name}"
                    )
                structures[structure_name] = {
                    str(item.get("block_id")): item
                    for item in blocks
                    if isinstance(item, dict) and item.get("block_id")
                }
                hashes[structure_name] = _sha256(structure_path)
            else:
                structures[structure_name] = {}
        block_ids = [str(item) for item in raw.get("evidence_block_ids") or ()]
        selected: list[dict[str, Any]] = []
        missing: list[str] = []
        for block_id in block_ids:
            block = structures[structure_name].get(block_id)
            if not isinstance(block, dict) or not str(block.get("content") or "").strip():
                missing.append(block_id)
                continue
            selected.append(
                {
                    "block_id": block_id,
                    "content": str(block["content"]),
                    "page_number": _page_number(block),
                }
            )
        if not block_ids or missing:
            warning = {
                "ref_id": ref_id,
                "source": source,
                "missing_block_ids": missing or block_ids,
                "fallback": "ref_evidence",
            }
            if not allow_partial:
                detail = "no evidence_block_ids" if not block_ids else ", ".join(missing)
                raise ClaimReviewRunError(
                    f"formal review requires complete structure provenance for {ref_id}: {detail}"
                )
            warnings.append(warning)
        enriched = dict(raw)
        if selected:
            enriched["evidence_blocks"] = selected
        result[ref_id] = enriched
    return result, warnings, dict(sorted(hashes.items()))


def _page_number(block: Mapping[str, Any]) -> int | None:
    value = block.get("page_number")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 1:
        return value
    page_idx = block.get("page_idx")
    if isinstance(page_idx, int) and not isinstance(page_idx, bool) and page_idx >= 0:
        return page_idx + 1
    return None


def _finalize_outputs(
    output_dir: Path,
    groups: Sequence[CompileGroup],
    source_drafts: Mapping[str, Any],
    runtime_reviews: list[dict[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    accepted: list[str] = []
    completed: list[str] = []
    drafts: dict[str, Any] = {}
    reviews = list(runtime_reviews)
    review_payloads: dict[str, Any] = {}
    failures = 0
    recompiles = 0
    for group in groups:
        path = output_dir / "checkpoints" / f"{_checkpoint_name(group.group_id)}.json"
        if not path.is_file():
            reviews.append(
                {
                    "kind": "claim_review",
                    "group_id": group.group_id,
                    "reason": "group_not_processed",
                }
            )
            continue
        item = _read_object(path)
        decision = str(item.get("decision") or "")
        if item.get("status") == "complete":
            completed.append(group.group_id)
        else:
            failures += 1
        if isinstance(item.get("draft"), dict):
            drafts[group.group_id] = item["draft"]
        elif isinstance(source_drafts.get(group.group_id), dict):
            drafts[group.group_id] = source_drafts[group.group_id]
        recompiles += int(item.get("recompile_attempts") or 0)
        review_payloads[group.group_id] = {
            "contract": item.get("contract"),
            "coverage": item.get("coverage"),
            "decision": decision or "failed",
            "recompile_attempts": int(item.get("recompile_attempts") or 0),
        }
        if decision == "pass" and item.get("status") == "complete":
            accepted.append(group.group_id)
        else:
            reviews.append(
                {
                    "kind": "claim_review",
                    "group_id": group.group_id,
                    "decision": decision or "failed",
                    "reason": item.get("review_reason") or item.get("error") or "not_passed",
                }
            )
    status = "complete" if len(accepted) == len(groups) else "needs_review"
    if failures:
        status = "partial"
    reviewed = {
        "schema": "okfolio.reviewed-compile-progress.v1",
        "completed_groups": completed,
        "accepted_groups": accepted,
        "withheld_groups": sorted(set(group.group_id for group in groups) - set(accepted)),
        "drafts": drafts,
        "claim_reviews": review_payloads,
        "recompiles": recompiles,
    }
    _write_json_atomic(output_dir / "reviewed_compile_progress.json", reviewed)
    _write_json_atomic(output_dir / "review_queue.json", {"reviews": reviews})
    summary = {
        "groups": len(groups),
        "completed": len(completed),
        "accepted": len(accepted),
        "withheld": len(groups) - len(accepted),
        "recompiles": recompiles,
        "reviews": len(reviews),
    }
    final_manifest = {**dict(manifest), "status": status, "summary": summary}
    _write_json_atomic(output_dir / "manifest.json", final_manifest)
    _write_json_atomic(
        output_dir / "claim_review_progress.json",
        {"schema": "okfolio.claim-review-progress.v1", "status": status, **summary},
    )
    return final_manifest


def _write_progress(
    output_dir: Path, groups: Sequence[CompileGroup], *, position: int
) -> None:
    completed = 0
    accepted = 0
    for group in groups:
        path = output_dir / "checkpoints" / f"{_checkpoint_name(group.group_id)}.json"
        if not path.is_file():
            continue
        item = _read_object(path)
        completed += item.get("status") == "complete"
        accepted += item.get("status") == "complete" and item.get("decision") == "pass"
    _write_json_atomic(
        output_dir / "claim_review_progress.json",
        {
            "schema": "okfolio.claim-review-progress.v1",
            "status": "running",
            "position": position,
            "groups": len(groups),
            "completed": completed,
            "accepted": accepted,
        },
    )


_SNAPSHOT_INTEGRITY_KEYS = (
    "schema",
    "inputs",
    "structures",
    "provenance_warnings",
)


def _snapshot_integrity(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Hash-carried integrity of a frozen snapshot, independent of layout.

    ``source_run_name`` and ``structures_dir_name`` are informational
    metadata: the same data restored from the committed ``experiment-data/``
    snapshot on a fresh clone lives under different directory names (e.g.
    ``experiment-data/source-run`` vs the original
    ``data/agent-runs/public10-local-qwen36-semantic-v2-20260809``) with
    byte-identical files.  Integrity is carried by the input/structure
    hashes, so only those keys (plus the schema id and provenance warnings)
    participate in snapshot comparisons.
    """
    return {
        key: snapshot[key] for key in _SNAPSHOT_INTEGRITY_KEYS if key in snapshot
    }


def _source_snapshot(
    source_run: Path,
    source_paths: Mapping[str, Path],
    structures_dir: Path,
) -> dict[str, Any]:
    return {
        "schema": "okfolio.claim-review-source-snapshot.v1",
        "source_run_name": source_run.name,
        "inputs": {
            name: {"sha256": _sha256(path), "size": path.stat().st_size}
            for name, path in source_paths.items()
        },
        "structures_dir_name": structures_dir.name,
    }


def _verify_snapshot(
    source_paths: Mapping[str, Path],
    snapshot: Mapping[str, Any],
    *,
    structures_dir: Path,
) -> None:
    inputs = snapshot.get("inputs")
    if not isinstance(inputs, dict):
        raise ClaimReviewRunError("invalid frozen source snapshot")
    for name, path in source_paths.items():
        item = inputs.get(name)
        if not isinstance(item, dict) or item.get("sha256") != _sha256(path):
            raise ClaimReviewRunError(f"source input changed during review: {name}")
    structures = snapshot.get("structures")
    if not isinstance(structures, dict):
        raise ClaimReviewRunError("invalid frozen structure snapshot")
    for name, expected in structures.items():
        path = structures_dir / str(name)
        if not path.is_file() or _sha256(path) != expected:
            raise ClaimReviewRunError(
                f"structure input changed during review: {name}"
            )


def _validate_paths(source_run: Path, output_dir: Path, *, resume: bool) -> None:
    if not source_run.is_dir():
        raise ClaimReviewRunError(f"source run does not exist: {source_run}")
    if output_dir == source_run or source_run in output_dir.parents:
        raise ClaimReviewRunError("Claim Review output must be outside the source run")
    if resume:
        if not output_dir.is_dir():
            raise ClaimReviewRunError("resume output directory does not exist")
    elif output_dir.exists():
        raise ClaimReviewRunError("output directory already exists; use --resume")


def _prepare_seed_checkpoint(
    seed_run: Path,
    *,
    group: CompileGroup,
    refs: Mapping[str, Mapping[str, Any]],
    known_source_anomalies: Sequence[str],
    draft_overrides: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate and reopen one failed checkpoint under a new configuration.

    A seed is deliberately narrower than ``resume``: it may carry forward only
    immutable semantic work (Contract, Draft and Coverage) from the exact same
    source snapshot. Runtime errors are never copied. A failed checkpoint must
    end at a correctable ``recompile`` decision; a completed checkpoint must
    already be an audited pass under the same prompts and stage-client policy.

    When a seed is reopened as ``running`` (a budget-exhausted ``human_review``
    group or an interrupted recompile) and the repair run supplies a
    ``draft_overrides`` entry for the group, the inherited Draft is replaced by
    the manual repair and the inherited Coverage artifacts (``coverage``,
    ``coverage_batches``, ``coverage_history``, ``decision``) are invalidated
    so the repaired draft is audited from scratch.  The source snapshot and the
    seed checkpoint file are never modified.
    """
    path = seed_run / "checkpoints" / f"{_checkpoint_name(group.group_id)}.json"
    if not path.is_file():
        raise ClaimReviewRunError(
            f"seed run has no checkpoint for group: {group.group_id}"
        )
    payload = _read_object(path)
    if str(payload.get("group_id") or "") != group.group_id:
        raise ClaimReviewRunError("seed checkpoint group ID does not match")
    if tuple(str(item) for item in payload.get("ref_ids") or ()) != group.ref_ids:
        raise ClaimReviewRunError(
            f"seed checkpoint Ref membership changed: {group.group_id}"
        )
    raw_contract = payload.get("contract")
    raw_draft = payload.get("draft")
    raw_coverage = payload.get("coverage")
    if not all(isinstance(item, dict) for item in (raw_contract, raw_draft, raw_coverage)):
        raise ClaimReviewRunError(
            "seed checkpoint must contain Contract, Draft and Coverage"
        )
    contract = _contract_from_payload(
        raw_contract,
        known_source_anomalies=known_source_anomalies,
    )
    if contract.group_id != group.group_id:
        raise ClaimReviewRunError("seed Contract group ID does not match")
    expected_refs = set(group.ref_ids)
    if {item.ref_id for item in contract.members} != expected_refs:
        raise ClaimReviewRunError("seed Contract members do not match the frozen group")
    if any(item.ref_id not in expected_refs for item in contract.claims):
        raise ClaimReviewRunError("seed Contract contains a claim from another group")
    if any(
        item.source_text_anomalies or item.ocr_suspicions
        for item in contract.claims
    ):
        raise ClaimReviewRunError("seed Contract contains unresolved source warnings")

    draft = _draft_from_payload(raw_draft)
    if draft.ref.concept_id != group.group_id:
        raise ClaimReviewRunError("seed Draft concept ID does not match the group")
    draft_hash = stable_hash(_draft_payload(draft))
    if str(payload.get("coverage_draft_sha256") or "") != draft_hash:
        raise ClaimReviewRunError("seed Coverage does not describe the saved Draft")
    matrix = _matrix_from_payload(
        raw_coverage,
        known_source_anomalies=known_source_anomalies,
    )
    if matrix.schema_version != "okfolio.claim-coverage.v2":
        raise ClaimReviewRunError("seed Coverage must use Claim Coverage v2")
    seed_status = str(payload.get("status") or "")
    seed_decision = str(payload.get("decision") or "")
    if seed_status == "complete" and seed_decision == "pass":
        if matrix.decision != "pass":
            raise ClaimReviewRunError(
                "complete seed Coverage does not have a pass decision"
            )
        reopened_status = "complete"
    elif seed_status != "complete" and seed_decision == "recompile":
        if matrix.decision != "recompile":
            raise ClaimReviewRunError(
                "failed seed Coverage does not have a recompile decision"
            )
        reopened_status = "running"
    elif (
        seed_status == "complete"
        and seed_decision == "human_review"
        and payload.get("review_reason") == "recompile_budget_exhausted"
    ):
        if matrix.decision != "recompile":
            raise ClaimReviewRunError(
                "budget-exhausted seed Coverage is not recompile-correctable"
            )
        reopened_status = "running"
    else:
        raise ClaimReviewRunError(
            "seed checkpoint must be a passed complete group or stop at a "
            "deterministic recompile decision"
        )

    group_refs = tuple(refs[ref_id] for ref_id in group.ref_ids)
    expected_provenance = json.loads(
        json.dumps(
            _evidence_provenance(
                build_evidence_units(
                    group_refs,
                    known_source_anomalies=known_source_anomalies,
                )
            ),
            ensure_ascii=False,
        )
    )
    if payload.get("evidence_provenance") != expected_provenance:
        raise ClaimReviewRunError(
            "seed evidence provenance does not match the current anomaly policy"
        )
    prior_attempts = int(payload.get("recompile_attempts") or 0)
    if prior_attempts < 0:
        raise ClaimReviewRunError("seed recompile attempt count is invalid")
    attempts = 0 if reopened_status == "running" else prior_attempts

    checkpoint_hash = _sha256(path)
    seeded = {
        "schema": "okfolio.claim-review-group.v1",
        "group_id": group.group_id,
        "ref_ids": list(group.ref_ids),
        "status": reopened_status,
        "contract": raw_contract,
        "evidence_provenance": expected_provenance,
        "draft": raw_draft,
        "draft_origin": str(payload.get("draft_origin") or "seed_run"),
        "recompile_attempts": attempts,
        "coverage": raw_coverage,
        "coverage_draft_sha256": draft_hash,
        "coverage_history": list(payload.get("coverage_history") or ()),
        "decision": seed_decision,
        "seed_provenance": {
            "run_name": seed_run.name,
            "checkpoint_sha256": checkpoint_hash,
            "prior_recompile_attempts": prior_attempts,
        },
    }
    if reopened_status == "running" and draft_overrides is not None:
        override = draft_overrides.get(group.group_id)
        if override is not None:
            # The repair draft replaces the inherited Draft and the inherited
            # Coverage must not describe it; reset the whole audit trail so
            # the re-opened group is reviewed from scratch under the override.
            replaced = DraftConcept(
                ref=draft.ref,
                title=override["title"],
                description=override["description"],
                body=override["body"],
            )
            seeded["draft"] = _draft_payload(replaced)
            seeded["draft_origin"] = "repair_override"
            seeded["recompile_attempts"] = 0
            for key in (
                "coverage",
                "coverage_draft_sha256",
                "coverage_batches",
                "coverage_history",
                "decision",
            ):
                seeded.pop(key, None)
    return seeded, checkpoint_hash


def _refs_from_source_progress(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_sources = payload.get("sources")
    values = raw_sources.values() if isinstance(raw_sources, dict) else raw_sources
    if not isinstance(values, (list, tuple)) and not hasattr(values, "__iter__"):
        raise ClaimReviewRunError("source_progress sources must be a list or object")
    refs: dict[str, dict[str, Any]] = {}
    for source in values:
        if not isinstance(source, dict):
            continue
        for raw in source.get("refs", []):
            if not isinstance(raw, dict) or not raw.get("ref_id"):
                continue
            ref_id = str(raw["ref_id"])
            if ref_id in refs:
                raise ClaimReviewRunError(f"duplicate ref_id: {ref_id}")
            refs[ref_id] = dict(raw)
    if not refs:
        raise ClaimReviewRunError("source_progress has no ConceptRefs")
    return refs


def _groups_from_payload(
    payload: Mapping[str, Any], refs: Mapping[str, Any]
) -> tuple[CompileGroup, ...]:
    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ClaimReviewRunError("groups.json has no groups")
    groups: list[CompileGroup] = []
    seen: set[str] = set()
    for raw in raw_groups:
        if not isinstance(raw, dict):
            raise ClaimReviewRunError("group must be an object")
        group_id = str(raw.get("group_id") or "").strip()
        ref_ids = tuple(str(item) for item in raw.get("ref_ids") or ())
        if not group_id or group_id in seen or not ref_ids:
            raise ClaimReviewRunError("groups require unique IDs and non-empty refs")
        unknown = set(ref_ids) - set(refs)
        if unknown:
            raise ClaimReviewRunError(f"group contains unknown Ref: {min(unknown)}")
        seen.add(group_id)
        groups.append(
            CompileGroup(
                group_id=group_id,
                ref_ids=ref_ids,
                title=str(raw.get("title") or group_id),
                description=str(raw.get("description") or raw.get("title") or group_id),
                reason=str(raw.get("reason") or "frozen source group"),
            )
        )
    actual = [ref_id for group in groups for ref_id in group.ref_ids]
    if len(actual) != len(set(actual)) or set(actual) != set(refs):
        raise ClaimReviewRunError("groups must cover every Ref exactly once")
    return tuple(groups)


def _agent_ref(raw: Mapping[str, Any]) -> AgentRefRecord:
    return AgentRefRecord(
        ref_id=str(raw["ref_id"]),
        article_id=str(raw.get("article_id") or raw.get("source") or ""),
        local_id=str(raw.get("local_id") or raw.get("concept_id") or raw["ref_id"]),
        type=str(raw["type"]),
        title=str(raw["title"]),
        description=str(raw["description"]),
        evidence=tuple(str(item) for item in raw.get("evidence") or ()),
        asset_hints=tuple(str(item) for item in raw.get("asset_hints") or ()),
        source=str(raw.get("source") or ""),
        section_path=tuple(str(item) for item in raw.get("section_path") or ()),
        page_start=_optional_int(raw.get("page_start")),
        page_end=_optional_int(raw.get("page_end")),
        evidence_block_ids=tuple(str(item) for item in raw.get("evidence_block_ids") or ()),
        semantic_signature=dict(raw.get("semantic_signature") or {}),
        scope=dict(raw.get("scope") or {}),
        ref_family_hint=str(raw.get("ref_family_hint") or ""),
        ref_version_id=str(raw.get("ref_version_id") or ""),
        document_family_id=str(raw.get("document_family_id") or ""),
        document_version_id=str(raw.get("document_version_id") or ""),
    )


def _contract_ref_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in raw.items()
        if key not in {"evidence", "evidence_blocks"}
    }


def _evidence_provenance(units: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": unit.evidence_id,
            "ref_id": unit.ref_id,
            "source_blocks": [asdict(item) for item in unit.source_blocks],
            "excluded_fragments": [
                asdict(item) for item in unit.excluded_fragments
            ],
        }
        for unit in units
    ]


def _contract_from_payload(
    payload: Mapping[str, Any],
    *,
    known_source_anomalies: Sequence[str] = (),
) -> ConceptClaimContract:
    from .claim_review import (
        ClaimObligation,
        EvidenceUnitReview,
        MemberContribution,
        _ocr_suspicions,
        _source_text_anomalies,
    )

    return ConceptClaimContract(
        group_id=str(payload["group_id"]),
        canonical_question=str(payload["canonical_question"]),
        members=tuple(MemberContribution(**item) for item in payload["members"]),
        claims=tuple(
            ClaimObligation(
                **{
                    **item,
                    "evidence_block_ids": tuple(item.get("evidence_block_ids") or ()),
                    "page_numbers": tuple(item.get("page_numbers") or ()),
                    "scope": dict(item.get("scope") or {}),
                    **_compatibility_warning_fields(
                        item,
                        text=str(item.get("evidence_excerpt") or ""),
                        source_detector=lambda value: _source_text_anomalies(
                            value,
                            known_source_anomalies,
                        ),
                        ocr_detector=_ocr_suspicions,
                    ),
                }
            )
            for item in payload["claims"]
        ),
        evidence_units=tuple(
            EvidenceUnitReview(
                **{**item, "claim_ids": tuple(item.get("claim_ids") or ())}
            )
            for item in payload["evidence_units"]
        ),
        schema_version=str(payload.get("schema_version") or "okfolio.claim-contract.v1"),
    )


def _coverage_batch_from_payload(
    payload: Mapping[str, Any],
) -> ClaimCoverageBatch:
    from .claim_review import (
        ClaimCoverageRow,
        DraftSentenceAttribution,
        ScopeViolation,
        UnsupportedClaim,
    )

    return ClaimCoverageBatch(
        batch_index=int(payload["batch_index"]),
        rows=tuple(
            ClaimCoverageRow(**item) for item in payload.get("rows") or ()
        ),
        sentence_attributions=tuple(
            DraftSentenceAttribution(
                **{
                    **item,
                    "claim_ids": tuple(item.get("claim_ids") or ()),
                }
            )
            for item in payload.get("sentence_attributions") or ()
        ),
        unsupported_claims=tuple(
            UnsupportedClaim(**item)
            for item in payload.get("unsupported_claims") or ()
        ),
        scope_violations=tuple(
            ScopeViolation(
                **{
                    **item,
                    "claim_ids": tuple(item.get("claim_ids") or ()),
                }
            )
            for item in payload.get("scope_violations") or ()
        ),
    )


def _matrix_from_payload(
    payload: Mapping[str, Any],
    *,
    known_source_anomalies: Sequence[str] = (),
) -> ClaimCoverageMatrix:
    from .claim_review import (
        ClaimCoverageRow,
        DraftSentenceAttribution,
        ScopeViolation,
        UnsupportedClaim,
        _ocr_suspicions,
        _source_text_anomalies,
    )

    return ClaimCoverageMatrix(
        rows=tuple(ClaimCoverageRow(**item) for item in payload["rows"]),
        sentence_attributions=tuple(
            DraftSentenceAttribution(
                **{
                    **item,
                    "claim_ids": tuple(item.get("claim_ids") or ()),
                    **_compatibility_warning_fields(
                        item,
                        text=str(item.get("draft_excerpt") or ""),
                        source_detector=lambda value: _source_text_anomalies(
                            value,
                            known_source_anomalies,
                        ),
                        ocr_detector=_ocr_suspicions,
                    ),
                }
            )
            for item in payload.get("sentence_attributions", [])
        ),
        unsupported_claims=tuple(
            UnsupportedClaim(**item) for item in payload.get("unsupported_claims", [])
        ),
        scope_violations=tuple(
            ScopeViolation(
                **{**item, "claim_ids": tuple(item.get("claim_ids") or ())}
            )
            for item in payload.get("scope_violations", [])
        ),
        decision=str(payload["decision"]),  # type: ignore[arg-type]
        schema_version=str(payload.get("schema_version") or "okfolio.claim-coverage.v1"),
    )


def _compatibility_warning_fields(
    payload: Mapping[str, Any],
    *,
    text: str,
    source_detector: Any,
    ocr_detector: Any,
) -> dict[str, tuple[str, ...]]:
    """Split legacy ``ocr_suspicions`` into corrected warning classes."""
    legacy_ocr = tuple(str(item) for item in payload.get("ocr_suspicions") or ())
    source = tuple(
        dict.fromkeys(
            tuple(str(item) for item in payload.get("source_text_anomalies") or ())
            + tuple(source_detector(text))
            + tuple(item for item in legacy_ocr if source_detector(item))
        )
    )
    ocr = tuple(
        dict.fromkeys(
            tuple(item for item in legacy_ocr if not source_detector(item))
            + tuple(ocr_detector(text))
        )
    )
    return {"source_text_anomalies": source, "ocr_suspicions": ocr}


def _draft_payload(draft: DraftConcept) -> dict[str, Any]:
    return {
        "ref": asdict(draft.ref),
        "title": draft.title,
        "description": draft.description,
        "body": draft.body,
    }


def _draft_from_payload(payload: Mapping[str, Any]) -> DraftConcept:
    raw = payload.get("ref")
    if not isinstance(raw, dict):
        raise ClaimReviewRunError("draft has no embedded ConceptRef")
    ref = ConceptRef(
        concept_id=str(raw["concept_id"]),
        type=str(raw["type"]),
        title=str(raw["title"]),
        description=str(raw["description"]),
        source=str(raw["source"]),
        evidence=tuple(str(item) for item in raw.get("evidence") or ()),
        asset_hints=tuple(str(item) for item in raw.get("asset_hints") or ()),
        section_path=tuple(str(item) for item in raw.get("section_path") or ()),
        page_start=_optional_int(raw.get("page_start")),
        page_end=_optional_int(raw.get("page_end")),
        evidence_block_ids=tuple(str(item) for item in raw.get("evidence_block_ids") or ()),
        semantic_signature=dict(raw.get("semantic_signature") or {}),
        scope=dict(raw.get("scope") or {}),
        ref_family_hint=str(raw.get("ref_family_hint") or ""),
        ref_version_id=str(raw.get("ref_version_id") or ""),
        document_family_id=str(raw.get("document_family_id") or ""),
        document_version_id=str(raw.get("document_version_id") or ""),
    )
    return DraftConcept(
        ref=ref,
        title=str(payload["title"]),
        description=str(payload["description"]),
        body=str(payload["body"]),
    )


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 1 else None


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClaimReviewRunError(f"cannot read JSON object: {path.name}") from error
    if not isinstance(value, dict):
        raise ClaimReviewRunError(f"JSON root must be an object: {path.name}")
    return value


def _checkpoint_name(group_id: str) -> str:
    return hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:24]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
