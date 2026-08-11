import json
from pathlib import Path

import pytest

from okfolio.evaluation.corpus import (
    build_c1_audited_concepts,
    build_t0_fixed_chunks,
    build_t1_parent_child,
    select_context_by_token_budget,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _structure(
    root: Path,
    stem: str,
    article_id: str,
    blocks: list[dict[str, object]],
) -> None:
    _write_json(
        root / f"{stem}.structure.json",
        {
            "schema_version": "kmpro.document-structure.v1",
            "status": "complete",
            "document_id": article_id,
            "blocks": blocks,
        },
    )


@pytest.fixture
def structures(tmp_path: Path) -> Path:
    root = tmp_path / "structures"
    root.mkdir()
    _structure(
        root,
        "a",
        "article-a",
        [
            {
                "block_id": "blk-title",
                "block_type": "title",
                "content": "封面",
                "page_number": 1,
                "heading_path": [],
                "evidence_eligible": False,
            },
            {
                "block_id": "blk-a1",
                "block_type": "text",
                "content": "甲" * 8,
                "page_number": 2,
                "heading_path": ["第一章"],
                "evidence_eligible": True,
                "content_hash": "h1",
            },
            {
                "block_id": "blk-a2",
                "block_type": "text",
                "content": "乙" * 8,
                "page_number": 2,
                "heading_path": ["第一章"],
                "evidence_eligible": True,
                "content_hash": "h2",
            },
            {
                "block_id": "blk-a3",
                "block_type": "text",
                "content": "丙" * 8,
                "page_number": 3,
                "heading_path": ["第二章"],
                "evidence_eligible": True,
                "content_hash": "h3",
            },
        ],
    )
    _structure(
        root,
        "b",
        "article-b",
        [
            {
                "block_id": "blk-b1",
                "block_type": "text",
                "content": "丁" * 8,
                "page_number": 4,
                "heading_path": ["跨文档"],
                "evidence_eligible": True,
                "content_hash": "h4",
            }
        ],
    )
    return root


def test_t0_fixed_chunks_preserve_all_canonical_atoms(structures: Path):
    build = build_t0_fixed_chunks(structures, max_chars=10)

    assert build.arm == "T0"
    assert build.article_ids == ("article-a", "article-b")
    assert build.audit()["status"] == "pass"
    assert build.audit()["evidence_atoms"] == 4
    assert all(unit.retrieval_evidence_atom_ids == unit.context_evidence_atom_ids for unit in build.units)
    assert all(str(atom.atom_id).startswith(atom.article_id + ":p") for atom in build.evidence_atoms)
    assert "blk-title" not in {atom.block_id for atom in build.evidence_atoms}


def test_t1_separates_child_hits_from_parent_context(structures: Path):
    build = build_t1_parent_child(
        structures,
        child_max_chars=10,
        parent_max_chars=30,
    )

    first, second = build.units[0], build.units[1]
    assert first.metadata["parent_id"] == second.metadata["parent_id"]
    assert len(first.retrieval_evidence_atom_ids) == 1
    assert len(first.context_evidence_atom_ids) == 2
    assert "第一章" in first.retrieval_text
    assert first.context_text == second.context_text
    assert build.audit()["status"] == "pass"


def test_context_selection_deduplicates_parent_and_obeys_joined_budget(structures: Path):
    build = build_t1_parent_child(
        structures,
        child_max_chars=10,
        parent_max_chars=30,
    )
    first, duplicate_parent, next_parent = build.units[:3]
    exact_budget = len(first.context_text + "\n\n---\n\n" + next_parent.context_text)

    selection = select_context_by_token_budget(
        [first, duplicate_parent, next_parent],
        token_budget=exact_budget,
        count_tokens=len,
    )

    assert [item.unit_id for item in selection.units] == [first.unit_id, next_parent.unit_id]
    assert selection.token_count == exact_budget
    assert selection.token_count <= selection.token_budget


def _agent_run(root: Path, *, acceptance: str = "pass", bad_block: bool = False) -> Path:
    run = root / "run"
    (run / "concepts").mkdir(parents=True)
    _write_json(run / "manifest.json", {"status": "complete"})
    _write_json(run / "acceptance.json", {"status": acceptance})
    _write_json(
        run / "refs.json",
        {
            "refs": [
                {
                    "ref_id": "ref-a",
                    "article_id": "article-a",
                    "source": "a.md",
                    "evidence_block_ids": ["blk-missing" if bad_block else "blk-a1"],
                },
                {
                    "ref_id": "ref-b",
                    "article_id": "article-b",
                    "source": "b.md",
                    "evidence_block_ids": ["blk-b1"],
                },
            ]
        },
    )
    _write_json(
        run / "concepts.json",
        {
            "concepts": [
                {
                    "group_id": "joint",
                    "ref_ids": ["ref-a", "ref-b"],
                    "status": "publishable",
                }
            ]
        },
    )
    (run / "concepts" / "joint.md").write_text(
        """---
type: 分析框架
title: 联合概念
description: 来自两篇公开材料
source: 多来源联合编译
concept_refs:
  - ref-a
  - ref-b
articles:
  - article-a
  - article-b
agent_quality_score: 0.91
---
这是经过审计的联合概念正文。
""",
        encoding="utf-8",
    )
    return run


def test_c1_imports_only_audited_concepts_and_maps_source_blocks(
    tmp_path: Path,
    structures: Path,
):
    build = build_c1_audited_concepts(
        run_dir=_agent_run(tmp_path),
        structures_dir=structures,
    )

    assert build.arm == "C1"
    assert len(build.units) == 1
    unit = build.units[0]
    assert unit.article_ids == ("article-a", "article-b")
    assert len(unit.context_evidence_atom_ids) == 2
    assert "联合概念" in unit.context_text
    assert build.audit()["unknown_evidence_atom_count"] == 0
    assert build.audit()["status"] == "pass"


def test_c1_rejects_unaccepted_run(tmp_path: Path, structures: Path):
    with pytest.raises(ValueError, match="acceptance.json"):
        build_c1_audited_concepts(
            run_dir=_agent_run(tmp_path, acceptance="fail"),
            structures_dir=structures,
        )


def test_c1_rejects_dangling_block_provenance(tmp_path: Path, structures: Path):
    with pytest.raises(ValueError, match="unknown or ineligible block"):
        build_c1_audited_concepts(
            run_dir=_agent_run(tmp_path, bad_block=True),
            structures_dir=structures,
        )
