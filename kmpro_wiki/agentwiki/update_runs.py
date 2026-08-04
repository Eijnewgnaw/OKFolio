"""Seed and audit immutable AgentWiki document-update runs.

The compiler treats a completed corpus run as an immutable snapshot.  A
follow-up version therefore starts in a new output directory: unchanged
Article discovery records and identical Concept compile results are copied as
*cache inputs*, while the baseline bundle remains untouched and queryable.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .versioning import reconcile_refs


def seed_update_run(baseline_run: Path, output_run: Path) -> dict[str, Any]:
    """Prepare an isolated resumable run using safe baseline cache entries."""
    baseline_run = baseline_run.resolve()
    output_run = output_run.resolve()
    manifest = _load(baseline_run / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("baseline Agent run must be complete")
    for name in ("source_progress.json", "compile_progress.json"):
        if not (baseline_run / name).is_file():
            raise ValueError(f"baseline Agent run lacks {name}")
    if output_run.exists():
        raise ValueError(f"update output already exists: {output_run}")

    output_run.mkdir(parents=True)
    for name in ("source_progress.json", "compile_progress.json"):
        shutil.copy2(baseline_run / name, output_run / name)
    seeded = {
        "version": 1,
        "status": "failed",
        "schema": "kmpro.agent-run.v2",
        "model": manifest["model"],
        "policy": manifest["policy"],
        "update_baseline": str(baseline_run),
        "reason": "seeded_for_incremental_document_update",
    }
    _write(output_run / "manifest.json", seeded)
    source_progress = _load(output_run / "source_progress.json")
    sources = _sources(source_progress)
    result = {
        "schema": "kmpro.agent-update-seed.v1",
        "baseline_run": str(baseline_run),
        "seeded_run": str(output_run),
        "cached_articles": len(sources),
        "cached_concept_compiles": len(
            _load(output_run / "compile_progress.json").get("drafts", {})
        ),
        "contract": (
            "Source hashes decide Article/ConceptRef reuse; per-group hashes "
            "decide Concept compile reuse. The baseline snapshot is never "
            "modified."
        ),
    }
    _write(output_run / "update_seed.json", result)
    return result


def audit_update_run(baseline_run: Path, update_run: Path) -> dict[str, Any]:
    """Produce a deterministic R0/R1 provenance and cache-reuse audit."""
    baseline_run = baseline_run.resolve()
    update_run = update_run.resolve()
    old_manifest = _load(baseline_run / "manifest.json")
    new_manifest = _load(update_run / "manifest.json")
    if old_manifest.get("status") != "complete":
        raise ValueError("baseline Agent run must be complete")
    if new_manifest.get("status") != "complete":
        raise ValueError("updated Agent run must be complete")

    old_refs = _load(baseline_run / "refs.json")["refs"]
    new_refs = _load(update_run / "refs.json")["refs"]
    old_sources = {item["source"]: item for item in _sources(_load(baseline_run / "source_progress.json"))}
    new_sources = {item["source"]: item for item in _sources(_load(update_run / "source_progress.json"))}
    if set(old_sources) != set(new_sources):
        raise ValueError("update audit requires the same document family set")

    by_source_old = _refs_by_source(old_refs)
    by_source_new = _refs_by_source(new_refs)
    source_updates: list[dict[str, Any]] = []
    changed_sources: list[str] = []
    for source in sorted(old_sources):
        old_hash = old_sources[source].get("source_hash")
        new_hash = new_sources[source].get("source_hash")
        reconciliation = reconcile_refs(
            by_source_old.get(source, []), by_source_new.get(source, [])
        )
        changed = old_hash != new_hash
        if changed:
            changed_sources.append(source)
        source_updates.append(
            {
                "source": source,
                "source_hash_changed": changed,
                "old_document": _document_identity(by_source_old.get(source, [])),
                "new_document": _document_identity(by_source_new.get(source, [])),
                "reconciliation": reconciliation,
            }
        )

    old_groups = _groups(baseline_run)
    new_groups = _groups(update_run)
    old_ids = set(old_groups)
    new_ids = set(new_groups)
    reused_groups = sorted(old_ids & new_ids)
    affected_groups = sorted(new_ids - old_ids)
    retired_groups = sorted(old_ids - new_ids)
    trace = _load(update_run / "agent_trace.json").get("events", [])
    reused_compile_events = [
        event
        for event in trace
        if event.get("stage") == "resume"
        and event.get("reused") == "compile_and_quality"
    ]
    all_statuses = Counter(
        status
        for source in source_updates
        for status, count in source["reconciliation"]["counts"].items()
        for _ in range(count)
    )
    result = {
        "schema": "kmpro.agent-update-audit.v1",
        "baseline_run": str(baseline_run),
        "update_run": str(update_run),
        "baseline_retained": baseline_run != update_run and baseline_run.is_dir(),
        "documents": {
            "total": len(old_sources),
            "changed": changed_sources,
            "unchanged": sorted(set(old_sources) - set(changed_sources)),
        },
        "refs": {
            "baseline": len(old_refs),
            "updated": len(new_refs),
            "statuses": dict(sorted(all_statuses.items())),
            "by_source": source_updates,
        },
        "concepts": {
            "baseline": len(old_groups),
            "updated": len(new_groups),
            "reused_group_ids": reused_groups,
            "affected_group_ids": affected_groups,
            "retired_group_ids": retired_groups,
            "cache_reused_compile_events": len(reused_compile_events),
            "cache_reuse_rate": (
                len(reused_groups) / len(new_groups) if new_groups else 1.0
            ),
        },
        "policy": {
            "history": "Baseline Article/Ref/Concept assets remain immutable.",
            "temporal_and_scenario_variants": (
                "Remain concurrent records; they are not global overwrites."
            ),
            "recompile_scope": (
                "Only groups whose member Ref payload hash changes are eligible "
                "for compilation; identical group hashes are reused."
            ),
        },
    }
    _write(update_run / "update_audit.json", result)
    return result


def _sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("source_progress.sources must be a list")
    return sources


def _refs_by_source(refs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for ref in refs:
        source = str(ref.get("source") or "")
        if source:
            result.setdefault(source, []).append(ref)
    return result


def _document_identity(refs: list[dict[str, Any]]) -> dict[str, str]:
    families = {str(item.get("document_family_id") or "") for item in refs}
    versions = {str(item.get("document_version_id") or "") for item in refs}
    return {
        "document_family_id": next(iter(families)) if len(families) == 1 else "",
        "document_version_id": next(iter(versions)) if len(versions) == 1 else "",
    }


def _groups(run_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        item["group_id"]: item
        for item in _load(run_dir / "groups.json").get("groups", [])
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
