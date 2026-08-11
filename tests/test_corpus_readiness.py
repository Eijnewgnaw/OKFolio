import json
from pathlib import Path

from kmpro_wiki.evaluation.corpus_readiness import audit_concept_rag_inputs


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_readiness_requires_every_source_in_refs_and_concepts(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "a.md").write_text("A", encoding="utf-8")
    (sources / "b.md").write_text("B", encoding="utf-8")
    refs = _write(
        tmp_path / "refs.json",
        {"refs": [{"ref_id": "ref-a", "source": "a.md"}]},
    )
    concepts = _write(
        tmp_path / "concepts.json",
        {
            "concepts": [
                {"ref_ids": ["ref-a"], "sources": ["a.md"]},
            ]
        },
    )

    result = audit_concept_rag_inputs(
        source_dir=sources,
        refs_path=refs,
        concepts_path=concepts,
    )

    assert result["status"] == "incomplete"
    assert result["uncovered_ref_sources"] == ["b.md"]
    assert result["uncovered_concept_sources"] == ["b.md"]


def test_readiness_accepts_full_multi_source_concept(tmp_path: Path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "a.md").write_text("A", encoding="utf-8")
    (sources / "b.md").write_text("B", encoding="utf-8")
    refs = _write(
        tmp_path / "refs.json",
        {
            "refs": [
                {"ref_id": "ref-a", "source": "a.md"},
                {"ref_id": "ref-b", "source": "b.md"},
            ]
        },
    )
    concepts = _write(
        tmp_path / "concepts.json",
        {
            "concepts": [
                {
                    "ref_ids": ["ref-a", "ref-b"],
                    "sources": ["a.md", "b.md"],
                }
            ]
        },
    )

    result = audit_concept_rag_inputs(
        source_dir=sources,
        refs_path=refs,
        concepts_path=concepts,
    )

    assert result["status"] == "ready"
    assert result["multi_source_concepts"] == 1
    assert result["dangling_concept_ref_ids"] == []
