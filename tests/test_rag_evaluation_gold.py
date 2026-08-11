import json
from pathlib import Path

import pytest

from okfolio.evaluation.contracts import EvidenceAtomId
from okfolio.evaluation.gold import (
    GoldDataError,
    gold_question_schema,
    load_gold_jsonl,
)


def _answerable(question_id: str = "q-1") -> dict:
    return {
        "question_id": question_id,
        "question": "成渝地区的相关政策是什么？",
        "question_type": "cross_document_synthesis",
        "answerable": True,
        "scope": {"region": "成渝", "time": "2025"},
        "required_facts": [
            {"fact_id": "f1", "claim": "政策包含措施甲", "weight": 2},
            {
                "fact_id": "f2",
                "claim": "政策包含措施乙",
                "critical": False,
            },
        ],
        "forbidden_facts": [
            {
                "fact_id": "x1",
                "claim": "错误时期的措施",
                "reason": "时期冲突",
            }
        ],
        "evidence_sets": [
            ["article-a:p003:s007", "article-b:p041:b002"],
        ],
        "reference_answer": "措施甲与措施乙。",
        "annotation": {
            "author": "annotator-a",
            "reviewer": "reviewer-b",
            "status": "adjudicated",
        },
    }


def _unanswerable(question_id: str = "q-2") -> dict:
    payload = _answerable(question_id)
    payload.update(
        {
            "question": "材料没有覆盖的问题",
            "answerable": False,
            "required_facts": [],
            "evidence_sets": [],
            "reference_answer": None,
        }
    )
    return payload


def _write_jsonl(path: Path, *rows: dict) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_evidence_atom_id_round_trips_canonical_identity():
    segment = EvidenceAtomId.parse("article-a:p003:s007")
    block = EvidenceAtomId.parse("article-b:p41:btable-2")

    assert segment.article_id == "article-a"
    assert segment.page == 3
    assert segment.atom_kind == "segment"
    assert segment.canonical == "article-a:p003:s007"
    assert block.canonical == "article-b:p041:btable-2"


@pytest.mark.parametrize(
    "value",
    ["article-a:p000:s1", "article a:p001:s1", "article:p1:x1", "p001:s1"],
)
def test_evidence_atom_id_rejects_ambiguous_or_zero_based_values(value: str):
    with pytest.raises(ValueError):
        EvidenceAtomId.parse(value)


def test_gold_loader_validates_and_builds_typed_questions(tmp_path: Path):
    questions = load_gold_jsonl(
        _write_jsonl(tmp_path / "gold.jsonl", _answerable(), _unanswerable())
    )

    assert [question.question_id for question in questions] == ["q-1", "q-2"]
    assert questions[0].required_facts[0].weight == 2.0
    assert questions[0].forbidden_claims[0].fact_id == "x1"
    assert questions[0].evidence_sets[0][1].atom_kind == "block"
    assert questions[1].answerable is False


def test_gold_loader_reports_line_and_schema_failure(tmp_path: Path):
    invalid = _answerable()
    invalid["evidence_sets"] = []

    with pytest.raises(GoldDataError, match=r"gold.jsonl:1:.*non-empty"):
        load_gold_jsonl(_write_jsonl(tmp_path / "gold.jsonl", invalid))


def test_gold_loader_rejects_duplicate_question_and_fact_ids(tmp_path: Path):
    with pytest.raises(GoldDataError, match="duplicate question_id"):
        load_gold_jsonl(
            _write_jsonl(tmp_path / "questions.jsonl", _answerable(), _answerable())
        )

    duplicate_fact = _answerable("q-facts")
    duplicate_fact["forbidden_facts"][0]["fact_id"] = "f1"
    with pytest.raises(GoldDataError, match="fact_ids must be unique"):
        load_gold_jsonl(_write_jsonl(tmp_path / "facts.jsonl", duplicate_fact))


def test_gold_schema_is_returned_as_a_safe_copy():
    first = gold_question_schema()
    first["required"].clear()

    assert "question_id" in gold_question_schema()["required"]

