from __future__ import annotations

import json
from pathlib import Path

from kmpro_wiki.agentwiki.update_runs import audit_update_run, seed_update_run


POLICY = {
    "quality_threshold": 0.8,
    "max_recompile_attempts": 2,
    "max_component_refs": 24,
}


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _ref(
    ref_id: str,
    source: str,
    version: str,
    evidence: str,
    scope: dict[str, str],
) -> dict[str, object]:
    return {
        "ref_id": ref_id,
        "source": source,
        "document_family_id": f"family-{source}",
        "document_version_id": version,
        "ref_family_hint": f"slot-{source}",
        "type": "政策建议",
        "title": f"{source} 政策",
        "evidence": [evidence],
        "scope": scope,
    }


def _baseline(run: Path) -> None:
    run.mkdir()
    _write(
        run / "manifest.json",
        {"status": "complete", "model": "test-model", "policy": POLICY},
    )
    _write(
        run / "source_progress.json",
        {
            "sources": [
                {"source": "a.md", "source_hash": "a0"},
                {"source": "b.md", "source_hash": "b0"},
            ]
        },
    )
    _write(
        run / "compile_progress.json",
        {"drafts": {"g-keep": {}, "g-old": {}}},
    )
    _write(
        run / "refs.json",
        {
            "refs": [
                _ref("old-a", "a.md", "v0-a", "稳定证据。", {"time": "2024年"}),
                _ref("old-b", "b.md", "v0-b", "补贴标准为30%。", {"time": "2024年"}),
            ]
        },
    )
    _write(
        run / "groups.json",
        {
            "groups": [
                {"group_id": "g-keep", "ref_ids": ["old-a"]},
                {"group_id": "g-old", "ref_ids": ["old-b"]},
            ]
        },
    )


def test_seed_and_audit_keep_baseline_and_reuse_unaffected_group(tmp_path: Path):
    baseline = tmp_path / "r0"
    update = tmp_path / "r1"
    _baseline(baseline)

    seed = seed_update_run(baseline, update)

    assert seed["cached_articles"] == 2
    assert seed["cached_concept_compiles"] == 2
    assert json.loads((baseline / "manifest.json").read_text())["status"] == "complete"
    seeded_manifest = json.loads((update / "manifest.json").read_text())
    assert seeded_manifest["status"] == "failed"
    assert seeded_manifest["update_baseline"] == str(baseline)

    _write(
        update / "manifest.json",
        {"status": "complete", "model": "test-model", "policy": POLICY},
    )
    _write(
        update / "source_progress.json",
        {
            "sources": [
                {"source": "a.md", "source_hash": "a0"},
                {"source": "b.md", "source_hash": "b1"},
            ]
        },
    )
    _write(
        update / "refs.json",
        {
            "refs": [
                _ref("old-a", "a.md", "v0-a", "稳定证据。", {"time": "2024年"}),
                _ref("new-b", "b.md", "v1-b", "补贴标准为50%。", {"time": "2025年"}),
            ]
        },
    )
    _write(
        update / "groups.json",
        {
            "groups": [
                {"group_id": "g-keep", "ref_ids": ["old-a"]},
                {"group_id": "g-new", "ref_ids": ["new-b"]},
            ]
        },
    )
    _write(
        update / "agent_trace.json",
        {"events": [{"stage": "resume", "reused": "compile_and_quality", "group_id": "g-keep"}]},
    )

    result = audit_update_run(baseline, update)

    assert result["baseline_retained"] is True
    assert result["documents"]["changed"] == ["b.md"]
    assert result["refs"]["statuses"] == {
        "temporal_variant": 1,
        "unchanged": 1,
    }
    assert result["concepts"]["reused_group_ids"] == ["g-keep"]
    assert result["concepts"]["affected_group_ids"] == ["g-new"]
    assert result["concepts"]["cache_reuse_rate"] == 0.5
    assert (update / "update_audit.json").is_file()
