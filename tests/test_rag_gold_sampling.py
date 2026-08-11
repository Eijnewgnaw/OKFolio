import json
from pathlib import Path

import pytest

from okfolio.evaluation.gold import GoldDataError, load_gold_jsonl
from okfolio.evaluation.gold_sampling import (
    GoldDraftSlot,
    GoldSamplingQuota,
    audit_gold_sampling_plan,
    prepare_gold_sampling_plan,
    write_gold_template_jsonl,
)


def _write_structure(root: Path, stem: str, article: str, heading: str) -> None:
    blocks = []
    for index in range(1, 7):
        blocks.append(
            {
                "block_id": f"blk-{stem}-{index}",
                "block_type": "text",
                "content": f"{heading}，202{index}年区域样本事实{index}。" * 4,
                "content_hash": f"hash-{stem}-{index}",
                "page_number": index,
                "heading_path": [heading, "区域发展"],
                "evidence_eligible": True,
            }
        )
    (root / f"{stem}.structure.json").write_text(
        json.dumps(
            {
                "schema_version": "kmpro.document-structure.v1",
                "status": "complete",
                "document_id": article,
                "blocks": blocks,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def structures(tmp_path: Path) -> Path:
    root = tmp_path / "structures"
    root.mkdir()
    _write_structure(root, "book-a", "article-a", "经济政策")
    _write_structure(root, "book-b", "article-b", "经济监管")
    return root


def test_sampling_is_deterministic_balanced_and_provenance_safe(structures: Path):
    quota = GoldSamplingQuota(2, 1, 1, 1, 1)
    first = prepare_gold_sampling_plan(structures, quota=quota, seed=17)
    second = prepare_gold_sampling_plan(structures, quota=quota, seed=17)

    assert [slot.question_id for slot in first.slots] == [
        slot.question_id for slot in second.slots
    ]
    assert len(first.slots) == 12
    assert first.audit["status"] == "pass"
    assert first.audit["books"] == 2
    assert first.audit["answerable_slots"] == 10
    assert first.audit["unanswerable_slots"] == 2
    assert first.audit["unknown_evidence_atom_count"] == 0
    assert first.audit["generation_policy"]["llm_calls"] == 0
    assert not first.audit["shortfalls"]
    assert all(
        len({atom.article_id for atom in slot.evidence_atoms}) == 2
        for slot in first.slots
        if slot.question_type == "cross_document_synthesis"
    )


def test_template_is_gold_shaped_but_cannot_be_mistaken_for_finished_gold(
    structures: Path,
    tmp_path: Path,
):
    plan = prepare_gold_sampling_plan(
        structures,
        quota=GoldSamplingQuota(1, 0, 0, 0, 1),
    )
    output = tmp_path / "worksheet.jsonl"
    write_gold_template_jsonl(plan, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert all(row["question"] == "" for row in rows)
    assert all(row["annotation"]["author"] == "" for row in rows)
    answerable = next(row for row in rows if row["answerable"])
    unanswerable = next(row for row in rows if not row["answerable"])
    assert answerable["required_facts"][0]["claim"] == ""
    assert answerable["evidence_sets"][0][0].startswith("article-")
    assert unanswerable["evidence_sets"] == []
    assert "source_excerpt" in answerable["scope"]["candidate_provenance"][0]
    with pytest.raises(GoldDataError, match="non-empty"):
        load_gold_jsonl(output)
    with pytest.raises(FileExistsError):
        write_gold_template_jsonl(plan, output)


def test_provenance_audit_detects_unknown_or_modified_atoms(structures: Path):
    plan = prepare_gold_sampling_plan(
        structures,
        quota=GoldSamplingQuota(1, 0, 0, 0, 0),
    )
    catalog = tuple(atom for slot in plan.slots for atom in slot.evidence_atoms)
    original = plan.slots[0]
    modified_atom = type(original.evidence_atoms[0])(
        **{
            **original.evidence_atoms[0].__dict__,
            "text": "tampered",
        }
    )
    tampered = GoldDraftSlot(
        question_id=original.question_id,
        question_type=original.question_type,
        answerable=True,
        primary_article_id=original.primary_article_id,
        evidence_atoms=(modified_atom,),
        selection_reason=original.selection_reason,
    )

    audit = audit_gold_sampling_plan([tampered], evidence_catalog=catalog)

    assert audit["status"] == "fail"
    assert audit["provenance_content_mismatch_count"] == 1

