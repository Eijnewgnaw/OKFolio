from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from okfolio.agentwiki.claim_review import build_draft_sentences
from okfolio.agentwiki.okf import parse_concept_markdown
from okfolio.agentwiki.state import stable_hash
from okfolio.evaluation.c1_materialize import (
    C1MaterializationError,
    materialize_c1_run,
)
from okfolio.evaluation.corpus import build_c1_audited_concepts


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fingerprint(path: Path) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _ref(
    ref_id: str,
    *,
    article_id: str,
    source: str,
    block_id: str,
    page: int,
) -> dict[str, object]:
    return {
        "ref_id": ref_id,
        "article_id": article_id,
        "local_id": ref_id,
        "type": "分析框架",
        "title": f"{article_id} 的区域协同依据",
        "description": "保留政策对象、时间和区域范围。",
        "evidence": [f"{article_id} 在第 {page} 页给出区域协同依据。"],
        "asset_hints": [],
        "source": source,
        "section_path": ["区域协同", "政策依据"],
        "page_start": page,
        "page_end": page,
        "evidence_block_ids": [block_id],
        "scope": {"region": article_id, "time": "2025年"},
    }


def _claim(
    claim_id: str,
    *,
    ref_id: str,
    block_id: str,
    page: int,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "ref_id": ref_id,
        "evidence_id": f"{ref_id}:bundle-test",
        "claim": f"{ref_id} 给出可追溯的区域协同依据。",
        "slot": "evidence",
        "kind": "fact",
        "evidence_excerpt": "给出可追溯的区域协同依据",
        "evidence_block_ids": [block_id],
        "page_numbers": [page],
        "scope": {"time": "2025年"},
        "source_text_anomalies": [],
        "ocr_suspicions": [],
    }


def _fixture(
    tmp_path: Path,
    *,
    group_ref_ids: tuple[str, ...] = ("ref-a", "ref-b"),
) -> tuple[Path, Path, Path]:
    source_run = tmp_path / "source-run"
    review_run = tmp_path / "claim-review"
    structures = tmp_path / "structures"
    source_run.mkdir()
    review_run.mkdir()
    structures.mkdir()

    refs = [
        _ref(
            "ref-a",
            article_id="article-a",
            source="a.md",
            block_id="blk-a1",
            page=3,
        ),
        _ref(
            "ref-b",
            article_id="article-b",
            source="b.md",
            block_id="blk-b1",
            page=7,
        ),
    ]
    _write_json(
        source_run / "source_progress.json",
        {
            "sources": [
                {"source": "a.md", "refs": [refs[0]]},
                {"source": "b.md", "refs": [refs[1]]},
            ]
        },
    )
    _write_json(
        source_run / "groups.json",
        {
            "groups": [
                {
                    "group_id": "group-joint",
                    "ref_ids": list(group_ref_ids),
                    "title": "跨区域协同政策依据",
                    "description": "联合两篇材料的可追溯依据。",
                    "reason": "回答同一规范化问题。",
                }
            ]
        },
    )
    draft = {
        "ref": {
            "concept_id": "group-joint",
            "type": "分析框架",
            "title": "跨区域协同政策依据",
            "description": "联合两篇材料的可追溯依据。",
            "source": "多来源联合编译",
            "evidence": [],
            "asset_hints": [],
        },
        "title": "跨区域协同政策依据",
        "description": "联合两篇材料的可追溯依据。",
        "body": (
            "## 核心判断\n\n"
            "两篇材料分别给出区域协同政策依据。"
        ),
    }
    _write_json(
        source_run / "compile_progress.json",
        {"drafts": {"group-joint": draft}},
    )

    structure_hashes: dict[str, str] = {}
    for stem, article_id, block_id, page in (
        ("a", "article-a", "blk-a1", 3),
        ("b", "article-b", "blk-b1", 7),
    ):
        structure_path = structures / f"{stem}.structure.json"
        _write_json(
            structure_path,
            {
                "schema_version": "kmpro.document-structure.v1",
                "status": "complete",
                "document_id": article_id,
                "blocks": [
                    {
                        "block_id": block_id,
                        "block_type": "text",
                        "content": (
                            f"{article_id} 给出可追溯的区域协同依据。"
                        ),
                        "page_number": page,
                        "heading_path": ["区域协同", "政策依据"],
                        "evidence_eligible": True,
                        "content_hash": f"hash-{block_id}",
                    }
                ],
            },
        )
        structure_hashes[structure_path.name] = hashlib.sha256(
            structure_path.read_bytes()
        ).hexdigest()

    source_inputs = {
        name: _fingerprint(source_run / name)
        for name in (
            "source_progress.json",
            "groups.json",
            "compile_progress.json",
        )
    }
    source_snapshot = {
        "schema": "okfolio.claim-review-source-snapshot.v1",
        "source_run_name": source_run.name,
        "inputs": source_inputs,
        "structures_dir_name": structures.name,
        "structures": structure_hashes,
        "provenance_warnings": [],
    }
    _write_json(review_run / "source_snapshot.json", source_snapshot)

    claims = [
        _claim(
            "claim-a",
            ref_id="ref-a",
            block_id="blk-a1",
            page=3,
        ),
        _claim(
            "claim-b",
            ref_id="ref-b",
            block_id="blk-b1",
            page=7,
        ),
    ]
    contract = {
        "schema_version": "okfolio.claim-contract.v1",
        "group_id": "group-joint",
        "canonical_question": "两篇材料如何界定区域协同政策依据？",
        "members": [
            {
                "ref_id": "ref-a",
                "relation": "supports",
                "contribution": "给出甲材料的政策依据。",
            },
            {
                "ref_id": "ref-b",
                "relation": "qualifies",
                "contribution": "补充乙材料的适用范围。",
            },
        ],
        "claims": claims,
        "evidence_units": [],
    }
    sentence_attributions = [
        {
            "sentence_id": sentence.sentence_id,
            "status": "supported",
            "claim_ids": ["claim-a", "claim-b"],
            "draft_excerpt": sentence.text,
            "finding": "该句由 Claim Contract 支持。",
            "source_text_anomalies": [],
            "ocr_suspicions": [],
        }
        for sentence in build_draft_sentences(
            {"description": draft["description"], "body": draft["body"]}
        )
    ]
    coverage = {
        "schema_version": "okfolio.claim-coverage.v2",
        "rows": [
            {
                "claim_id": claim["claim_id"],
                "status": "covered",
                "draft_excerpt": "区域协同政策依据",
                "finding": "草稿覆盖对应事实。",
            }
            for claim in claims
        ],
        "sentence_attributions": sentence_attributions,
        "unsupported_claims": [],
        "scope_violations": [],
        "decision": "pass",
    }
    reviewed = {
        "schema": "okfolio.reviewed-compile-progress.v1",
        "completed_groups": ["group-joint"],
        "accepted_groups": ["group-joint"],
        "withheld_groups": [],
        "drafts": {"group-joint": draft},
        "claim_reviews": {
            "group-joint": {
                "contract": contract,
                "coverage": coverage,
                "decision": "pass",
                "recompile_attempts": 1,
            }
        },
        "recompiles": 1,
    }
    _write_json(review_run / "reviewed_compile_progress.json", reviewed)
    _write_json(review_run / "review_queue.json", {"reviews": []})
    _write_json(
        review_run / "manifest.json",
        {
            "schema": "okfolio.claim-review-run.v1",
            "status": "complete",
            "source_run_name": source_run.name,
            "source_snapshot_sha256": stable_hash(source_snapshot),
            "configuration": {
                "allow_partial": False,
                "selected_group_ids": [],
            },
            "summary": {
                "groups": 1,
                "completed": 1,
                "accepted": 1,
                "withheld": 0,
                "recompiles": 1,
                "reviews": 0,
            },
        },
    )
    return source_run, review_run, structures


def test_materializes_source_immutable_standard_c1_run(tmp_path: Path):
    source_run, review_run, structures = _fixture(tmp_path)
    output = tmp_path / "c1-published"
    source_before = _tree_hashes(source_run)
    review_before = _tree_hashes(review_run)

    manifest = materialize_c1_run(
        source_run=source_run,
        review_run=review_run,
        output_dir=output,
    )

    assert manifest["schema"] == "okfolio.c1-materialized-run.v1"
    assert manifest["status"] == "complete"
    assert manifest["refs"] == 2
    assert manifest["concepts"] == 1
    assert source_before == _tree_hashes(source_run)
    assert review_before == _tree_hashes(review_run)

    acceptance = json.loads((output / "acceptance.json").read_text())
    assert acceptance["status"] == "pass"
    assert acceptance["expected_groups"] == 1
    assert acceptance["accepted_groups"] == 1
    assert acceptance["claim_coverage_schema"] == "okfolio.claim-coverage.v2"
    document = parse_concept_markdown(
        "group-joint.md",
        (output / "concepts" / "group-joint.md").read_text(encoding="utf-8"),
    )
    assert document.frontmatter["concept_refs"] == ["ref-a", "ref-b"]
    assert (
        document.frontmatter["canonical_question"]
        == "两篇材料如何界定区域协同政策依据？"
    )
    assert {item["claim_id"] for item in document.frontmatter["claims"]} == {
        "claim-a",
        "claim-b",
    }
    assert len(document.frontmatter["sentence_attributions"]) == 2
    assert {
        item["status"]
        for item in document.frontmatter["sentence_attributions"]
    } == {"supported"}
    assert document.frontmatter["source_locations"] == [
        {
            "ref_id": "ref-a",
            "article_id": "article-a",
            "source": "a.md",
            "section_path": ["区域协同", "政策依据"],
            "page_start": 3,
            "page_end": 3,
            "evidence_block_ids": ["blk-a1"],
            "scope": {"region": "article-a", "time": "2025年"},
        },
        {
            "ref_id": "ref-b",
            "article_id": "article-b",
            "source": "b.md",
            "section_path": ["区域协同", "政策依据"],
            "page_start": 7,
            "page_end": 7,
            "evidence_block_ids": ["blk-b1"],
            "scope": {"region": "article-b", "time": "2025年"},
        },
    ]

    corpus = build_c1_audited_concepts(
        run_dir=output,
        structures_dir=structures,
    )
    assert corpus.audit()["status"] == "pass"
    assert len(corpus.units) == 1
    assert len(corpus.units[0].context_evidence_atom_ids) == 2


def test_rejects_partial_claim_review_without_creating_output(tmp_path: Path):
    source_run, review_run, _structures = _fixture(tmp_path)
    manifest_path = review_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "partial"
    _write_json(manifest_path, manifest)
    output = tmp_path / "c1-published"

    with pytest.raises(C1MaterializationError, match="partial or incomplete"):
        materialize_c1_run(
            source_run=source_run,
            review_run=review_run,
            output_dir=output,
            expected_groups=1,
        )
    assert not output.exists()


def test_explicit_expected_group_count_is_an_additional_gate(tmp_path: Path):
    source_run, review_run, _structures = _fixture(tmp_path)
    output = tmp_path / "c1-published"

    with pytest.raises(C1MaterializationError, match="explicit expected"):
        materialize_c1_run(
            source_run=source_run,
            review_run=review_run,
            output_dir=output,
            expected_groups=332,
        )
    assert not output.exists()


def test_rejects_withheld_group_even_if_manifest_claims_complete(tmp_path: Path):
    source_run, review_run, _structures = _fixture(tmp_path)
    reviewed_path = review_run / "reviewed_compile_progress.json"
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    reviewed["accepted_groups"] = []
    reviewed["withheld_groups"] = ["group-joint"]
    _write_json(reviewed_path, reviewed)
    output = tmp_path / "c1-published"

    with pytest.raises(C1MaterializationError, match="non-pass groups"):
        materialize_c1_run(
            source_run=source_run,
            review_run=review_run,
            output_dir=output,
            expected_groups=1,
        )
    assert not output.exists()


def test_rejects_nonpass_group_decision(tmp_path: Path):
    source_run, review_run, _structures = _fixture(tmp_path)
    reviewed_path = review_run / "reviewed_compile_progress.json"
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    reviewed["claim_reviews"]["group-joint"]["decision"] = "human_review"
    _write_json(reviewed_path, reviewed)
    output = tmp_path / "c1-published"

    with pytest.raises(C1MaterializationError, match="did not pass"):
        materialize_c1_run(
            source_run=source_run,
            review_run=review_run,
            output_dir=output,
            expected_groups=1,
        )
    assert not output.exists()


def test_rejects_legacy_coverage_v1(tmp_path: Path):
    source_run, review_run, _structures = _fixture(tmp_path)
    reviewed_path = review_run / "reviewed_compile_progress.json"
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    reviewed["claim_reviews"]["group-joint"]["coverage"][
        "schema_version"
    ] = "okfolio.claim-coverage.v1"
    _write_json(reviewed_path, reviewed)
    output = tmp_path / "c1-published"

    with pytest.raises(C1MaterializationError, match="unsupported coverage"):
        materialize_c1_run(
            source_run=source_run,
            review_run=review_run,
            output_dir=output,
            expected_groups=1,
        )
    assert not output.exists()


def test_rejects_incomplete_sentence_catalog_attribution(tmp_path: Path):
    source_run, review_run, _structures = _fixture(tmp_path)
    reviewed_path = review_run / "reviewed_compile_progress.json"
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    reviewed["claim_reviews"]["group-joint"]["coverage"][
        "sentence_attributions"
    ].pop()
    _write_json(reviewed_path, reviewed)
    output = tmp_path / "c1-published"

    with pytest.raises(C1MaterializationError, match="do not cover catalog"):
        materialize_c1_run(
            source_run=source_run,
            review_run=review_run,
            output_dir=output,
            expected_groups=1,
        )
    assert not output.exists()


@pytest.mark.parametrize("status", ["unsupported", "uncertain"])
def test_rejects_non_supported_sentence_attribution(
    tmp_path: Path,
    status: str,
):
    source_run, review_run, _structures = _fixture(tmp_path)
    reviewed_path = review_run / "reviewed_compile_progress.json"
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    attribution = reviewed["claim_reviews"]["group-joint"]["coverage"][
        "sentence_attributions"
    ][0]
    attribution["status"] = status
    if status == "unsupported":
        attribution["claim_ids"] = []
    _write_json(reviewed_path, reviewed)
    output = tmp_path / "c1-published"

    with pytest.raises(C1MaterializationError, match="is not supported"):
        materialize_c1_run(
            source_run=source_run,
            review_run=review_run,
            output_dir=output,
            expected_groups=1,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("target", "field"),
    [
        ("claim", "source_text_anomalies"),
        ("sentence", "ocr_suspicions"),
    ],
)
def test_warning_blocks_c1_publication(
    tmp_path: Path,
    target: str,
    field: str,
):
    source_run, review_run, _structures = _fixture(tmp_path)
    reviewed_path = review_run / "reviewed_compile_progress.json"
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    claim_review = reviewed["claim_reviews"]["group-joint"]
    if target == "claim":
        claim_review["contract"]["claims"][0][field] = ["可疑源文本"]
    else:
        claim_review["coverage"]["sentence_attributions"][0][field] = ["�"]
    _write_json(reviewed_path, reviewed)
    output = tmp_path / "c1-published"

    with pytest.raises(C1MaterializationError, match="contains"):
        materialize_c1_run(
            source_run=source_run,
            review_run=review_run,
            output_dir=output,
            expected_groups=1,
        )
    assert not output.exists()


def test_rejects_source_group_that_omits_a_ref(tmp_path: Path):
    source_run, review_run, _structures = _fixture(
        tmp_path,
        group_ref_ids=("ref-a",),
    )
    output = tmp_path / "c1-published"

    with pytest.raises(C1MaterializationError, match="groups omit ConceptRef"):
        materialize_c1_run(
            source_run=source_run,
            review_run=review_run,
            output_dir=output,
            expected_groups=1,
        )
    assert not output.exists()


def test_rejects_source_mutated_after_claim_review(tmp_path: Path):
    source_run, review_run, _structures = _fixture(tmp_path)
    groups_path = source_run / "groups.json"
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    groups["groups"][0]["reason"] = "评审完成后被改写。"
    _write_json(groups_path, groups)
    output = tmp_path / "c1-published"

    with pytest.raises(C1MaterializationError, match="changed after review"):
        materialize_c1_run(
            source_run=source_run,
            review_run=review_run,
            output_dir=output,
            expected_groups=1,
        )
    assert not output.exists()
