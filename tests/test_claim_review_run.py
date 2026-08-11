from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kmpro_wiki.agentwiki.claim_review_run import (
    ClaimReviewRunError,
    ClaimReviewStageClients,
    ClaimReviewTemplates,
    _quality_audit_for_recompile,
    run_claim_review,
)
from kmpro_wiki.agentwiki.claim_review import (
    ClaimCoverageMatrix,
    ClaimCoverageRow,
    ClaimObligation,
    ConceptClaimContract,
    EvidenceUnitReview,
    MemberContribution,
    UnsupportedClaim,
    build_draft_sentences,
)
from kmpro_wiki.agentwiki.state import stable_hash


TEMPLATES = ClaimReviewTemplates(
    contract=(
        "{group}\n## REF\n{concept_refs}\n## UNITS\n{evidence_units}"
        "\n## KNOWN\n{known_source_anomalies}"
    ),
    coverage=(
        "{claim_contract}\n## DRAFT\n{draft}"
        "\n## KNOWN\n{known_source_anomalies}"
    ),
    compile="{concept_ref}\n## EVIDENCE\n{evidence}",
    recompile=(
        "{concept_ref}\n{evidence}\n{previous_draft}\n{quality_issues}\n"
        "{recompile_instructions}"
    ),
)

KNOWN_ANOMALY = "异常占位短语甲"


class PassingClient:
    model = "fake-model"
    max_tokens = 4096
    response_format = "json_schema"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(
        self,
        prompt: str,
        *,
        json_schema_name: str | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> str:
        assert json_schema_name is not None
        assert json_schema is not None
        self.calls.append((json_schema_name, prompt))
        if json_schema_name == "concept_claim_contract":
            properties = json_schema["properties"]  # type: ignore[index]
            member = properties["members"]["items"]["properties"]["ref_id"]["enum"][0]  # type: ignore[index]
            evidence_id = properties["evidence_units"]["items"]["properties"]["evidence_id"]["enum"][0]  # type: ignore[index]
            slot = properties["evidence_units"]["items"]["properties"]["claims"]["items"]["properties"]["slot"]["enum"][0]  # type: ignore[index]
            return json.dumps(
                {
                    "canonical_question": "核心事实是什么？",
                    "members": [
                        {
                            "ref_id": member,
                            "relation": "supports",
                            "contribution": "给出核心事实。",
                        }
                    ],
                    "evidence_units": [
                        {
                            "evidence_id": evidence_id,
                            "disposition": "required",
                            "reason": "直接回答中心问题。",
                            "claims": [
                                {
                                    "claim": "核心事实为30%。",
                                    "slot": slot,
                                    "kind": "metric",
                                    "evidence_excerpt": "核心事实为30%。",
                                    "scope": {},
                                }
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        if json_schema_name in {"concept_draft", "agent_recompiled_concept"}:
            return json.dumps(
                {
                    "title": "核心事实",
                    "description": "核心事实为30%。",
                    "sections": [
                        {
                            "heading": "核心判断",
                            "paragraphs": ["核心事实为30%。"],
                            "bullets": [],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        if json_schema_name == "concept_claim_coverage":
            properties = json_schema["properties"]  # type: ignore[index]
            claim_id = properties["rows"]["items"]["properties"]["claim_id"]["enum"][0]  # type: ignore[index]
            sentence_ids = properties["sentence_attributions"]["items"]["properties"]["sentence_id"]["enum"]  # type: ignore[index]
            return json.dumps(
                {
                    "rows": [
                        {
                            "claim_id": claim_id,
                            "status": "covered",
                            "draft_excerpt": "核心事实为30%。",
                            "finding": "草稿准确覆盖。",
                        }
                    ],
                    "sentence_attributions": [
                        {
                            "sentence_id": sentence_id,
                            "status": "supported",
                            "claim_ids": [claim_id],
                            "draft_excerpt": "核心事实为30%。",
                            "finding": "该句由核心事实 claim 支持。",
                        }
                        for sentence_id in sentence_ids
                    ],
                    "unsupported_claims": [],
                    "scope_violations": [],
                },
                ensure_ascii=False,
            )
        raise AssertionError(json_schema_name)


class NoCallClient(PassingClient):
    def complete(self, *args: object, **kwargs: object) -> str:
        raise AssertionError("resume repeated an already completed model call")


class FailOnSecondGroupClient(PassingClient):
    def complete(self, prompt: str, **kwargs: object) -> str:
        if len(self.calls) == 2:
            raise RuntimeError("simulated interruption")
        return super().complete(prompt, **kwargs)  # type: ignore[arg-type]


class SeedCheckpointClient(PassingClient):
    """Create a failed checkpoint after valid Contract, Draft and Coverage.

    Coverage runs in deterministic sentence batches, so the deterministic
    recompile signal is produced through ``unsupported_claims`` (model rows
    must never mark a claim omitted; code completes omitted rows at merge).
    """

    def complete(self, prompt: str, **kwargs: object) -> str:
        schema_name = kwargs.get("json_schema_name")
        if schema_name == "concept_claim_coverage":
            response = json.loads(super().complete(prompt, **kwargs))
            response["unsupported_claims"] = [
                {
                    "draft_excerpt": "核心事实为30%。",
                    "finding": "草稿未按要求组织。",
                }
            ]
            return json.dumps(response, ensure_ascii=False)
        if schema_name == "agent_recompiled_concept":
            raise RuntimeError("simulated recompile interruption")
        return super().complete(prompt, **kwargs)


class ContinueSeedClient(PassingClient):
    def complete(self, prompt: str, **kwargs: object) -> str:
        schema_name = kwargs.get("json_schema_name")
        if schema_name in {"concept_claim_contract", "concept_draft"}:
            raise AssertionError("seeded run repeated an inherited model stage")
        return super().complete(prompt, **kwargs)


BATCH_TEMPLATES = ClaimReviewTemplates(
    contract=TEMPLATES.contract,
    coverage=(
        "## CONTRACT\n{claim_contract}\n## DRAFT\n{draft}"
        "\n## SENTENCES\n{draft_sentences}\n## KNOWN\n{known_source_anomalies}"
    ),
    compile=TEMPLATES.compile,
    recompile=TEMPLATES.recompile,
)


def _normalized(value: str) -> str:
    return "".join(value.split())


class BatchCoverageClient(PassingClient):
    """Emit each claim in exactly the batches whose sentences carry its evidence.

    The model-visible batch sentences are read from the rendered prompt, so
    this client works for any coverage batch size.  A claim row is emitted in
    every batch whose sentences contain the claim's evidence excerpt; every
    batch sentence is attributed as supported to the frozen claims.  The
    mapping is a pure function of the batch content, so resumed runs re-derive
    the same rows for skipped and re-run batches (fixtures keep each claim's
    evidence in exactly one batch to avoid intentional cross-batch duplicates).
    """

    def __init__(self, *, interrupt_after: int | None = None) -> None:
        super().__init__()
        self.interrupt_after = interrupt_after
        self.coverage_calls = 0
        self.coverage_sentence_ids: list[list[str]] = []
        self.force_recompile = False

    def complete(self, prompt: str, **kwargs: object) -> str:
        schema_name = kwargs.get("json_schema_name")
        if schema_name == "concept_claim_coverage":
            if (
                self.interrupt_after is not None
                and self.coverage_calls >= self.interrupt_after
            ):
                raise RuntimeError("simulated coverage interruption")
            self.coverage_calls += 1
            return self._coverage_response(
                prompt,
                schema=kwargs["json_schema"],  # type: ignore[arg-type]
            )
        return super().complete(prompt, **kwargs)

    def _coverage_response(self, prompt: str, *, schema: dict[str, object]) -> str:
        sentences = self._batch_sentences(prompt)
        contract = self._claim_contract(prompt)
        properties = schema["properties"]  # type: ignore[index]
        sentence_ids = properties["sentence_attributions"]["items"]["properties"][
            "sentence_id"
        ]["enum"]  # type: ignore[index]
        self.coverage_sentence_ids.append(list(sentence_ids))
        batch = {item["sentence_id"]: item["text"] for item in sentences}
        claim_ids = properties["rows"]["items"]["properties"]["claim_id"][
            "enum"
        ]  # type: ignore[index]
        claims = {item["claim_id"]: item for item in contract["claims"]}
        rows = []
        for claim_id in claim_ids:
            needle = _normalized(str(claims[claim_id]["evidence_excerpt"]))
            matches = [
                sentence_id
                for sentence_id, text in batch.items()
                if needle and needle in _normalized(text)
            ]
            if not matches:
                continue
            rows.append(
                {
                    "claim_id": claim_id,
                    "status": "covered",
                    "draft_excerpt": batch[matches[0]],
                    "finding": "草稿准确覆盖。",
                }
            )
        unsupported = []
        if self.force_recompile:
            unsupported.append(
                {
                    "draft_excerpt": next(iter(batch.values())),
                    "finding": "草稿尚未按要求组织。",
                }
            )
        return json.dumps(
            {
                "rows": rows,
                "sentence_attributions": [
                    {
                        "sentence_id": sentence_id,
                        "status": "supported",
                        "claim_ids": list(claim_ids),
                        "draft_excerpt": text,
                        "finding": "该句由冻结合同 claim 支持。",
                    }
                    for sentence_id, text in batch.items()
                ],
                "unsupported_claims": unsupported,
                "scope_violations": [],
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _batch_sentences(prompt: str) -> list[dict[str, str]]:
        marker = "## SENTENCES\n"
        if marker not in prompt:
            raise AssertionError("batch prompt must render the draft sentences")
        payload = prompt.split(marker, 1)[1].split("\n## KNOWN", 1)[0].strip()
        return json.loads(payload)

    @staticmethod
    def _claim_contract(prompt: str) -> dict[str, object]:
        marker = "## CONTRACT\n"
        if marker not in prompt:
            raise AssertionError("batch prompt must render the claim contract")
        payload = prompt.split(marker, 1)[1].split("\n## DRAFT", 1)[0].strip()
        return json.loads(payload)


class PersistentContractFailureClient(PassingClient):
    """Deterministically fail group-2's contract stage (non-verbatim excerpt)."""

    def complete(self, prompt: str, **kwargs: object) -> str:
        schema_name = kwargs.get("json_schema_name")
        if schema_name == "concept_claim_contract" and "group-2" in prompt:
            response = json.loads(super().complete(prompt, **kwargs))
            response["evidence_units"][0]["claims"][0][
                "evidence_excerpt"
            ] = "不属于证据的摘录"
            return json.dumps(response, ensure_ascii=False)
        return super().complete(prompt, **kwargs)


class RecompileChangesDraftClient(BatchCoverageClient):
    """Force one recompile whose draft differs from the source draft."""

    def __init__(self) -> None:
        super().__init__()
        self.force_recompile = True

    def complete(self, prompt: str, **kwargs: object) -> str:
        schema_name = kwargs.get("json_schema_name")
        if schema_name == "agent_recompiled_concept":
            # A fresh coverage round audits the new draft from scratch.
            self.force_recompile = False
            return json.dumps(
                {
                    "title": "核心事实",
                    "description": "核心事实为30%。",
                    "sections": [
                        {
                            "heading": "核心判断",
                            "paragraphs": [
                                "补充说明甲。补充说明乙。补充说明丙。"
                            ],
                            "bullets": [],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        return super().complete(prompt, **kwargs)


class InterruptedRecompileClient(BatchCoverageClient):
    """Force one recompile, then interrupt the fresh coverage round.

    Leaves a failed checkpoint whose persisted draft is the recompiled draft
    (not the override), so a resumed run must keep it instead of re-injecting
    the override.
    """

    def __init__(self) -> None:
        super().__init__(interrupt_after=1)
        self.force_recompile = True

    def complete(self, prompt: str, **kwargs: object) -> str:
        schema_name = kwargs.get("json_schema_name")
        if schema_name == "agent_recompiled_concept":
            return json.dumps(
                {
                    "title": "核心事实",
                    "description": "核心事实为30%。",
                    "sections": [
                        {
                            "heading": "核心判断",
                            "paragraphs": [
                                "补充说明甲。补充说明乙。补充说明丙。"
                            ],
                            "bullets": [],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        return super().complete(prompt, **kwargs)


class BudgetExhaustedClient(BatchCoverageClient):
    """Drive every coverage round to a deterministic recompile decision.

    Exhausts the recompile budget so the group ends as a complete
    ``human_review`` checkpoint with ``recompile_budget_exhausted`` — the seed
    shape a repair run reopens under ``_prepare_seed_checkpoint``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.force_recompile = True

    def complete(self, prompt: str, **kwargs: object) -> str:
        schema_name = kwargs.get("json_schema_name")
        if schema_name == "agent_recompiled_concept":
            return json.dumps(
                {
                    "title": "核心事实",
                    "description": "核心事实为30%。",
                    "sections": [
                        {
                            "heading": "核心判断",
                            "paragraphs": [
                                "补充说明甲。补充说明乙。补充说明丙。"
                            ],
                            "bullets": [],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        return super().complete(prompt, **kwargs)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _source_fixture(tmp_path: Path, *, with_draft: bool) -> tuple[Path, Path]:
    root = tmp_path / "corpus"
    source_run = root / "agent-runs" / "run-v1"
    source_run.mkdir(parents=True)
    ref = {
        "ref_id": "ref-1",
        "article_id": "article-1",
        "local_id": "local-1",
        "type": "分析框架",
        "title": "核心事实",
        "description": "核心事实为30%。",
        "evidence": ["核心事实为30%。无关背景不应进入缺失草稿的编译输入。"],
        "asset_hints": [],
        "source": "doc.md",
        "evidence_block_ids": ["block-1"],
        "scope": {},
    }
    _write(source_run / "source_progress.json", {"sources": [{"refs": [ref]}]})
    _write(
        source_run / "groups.json",
        {
            "groups": [
                {
                    "group_id": "group-1",
                    "ref_ids": ["ref-1"],
                    "title": "核心事实",
                    "description": "核心事实为30%。",
                    "reason": "单 Ref。",
                }
            ]
        },
    )
    drafts: dict[str, object] = {}
    if with_draft:
        drafts["group-1"] = {
            "ref": {
                "concept_id": "group-1",
                "type": "分析框架",
                "title": "核心事实",
                "description": "核心事实为30%。",
                "source": "doc.md",
                "evidence": ["核心事实为30%。"],
                "asset_hints": [],
            },
            "title": "核心事实",
            "description": "核心事实为30%。",
            "body": "## 核心判断\n\n核心事实为30%。",
        }
    _write(source_run / "compile_progress.json", {"drafts": drafts})
    structures = root / "normalized-sources"
    _write(
        structures / "doc.structure.json",
        {
            "blocks": [
                {
                    "block_id": "block-1",
                    "content": "核心事实为30%。无关背景不应进入缺失草稿的编译输入。",
                    "page_number": 7,
                }
            ]
        },
    )
    return source_run, structures


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _add_third_group(source_run: Path, structures: Path) -> None:
    source_progress = json.loads((source_run / "source_progress.json").read_text())
    third = dict(source_progress["sources"][0]["refs"][0])
    third.update(
        {
            "ref_id": "ref-3",
            "article_id": "article-3",
            "local_id": "local-3",
            "evidence_block_ids": ["block-3"],
        }
    )
    source_progress["sources"][0]["refs"].append(third)
    _write(source_run / "source_progress.json", source_progress)
    groups = json.loads((source_run / "groups.json").read_text())
    groups["groups"].append(
        {
            "group_id": "group-3",
            "ref_ids": ["ref-3"],
            "title": "核心事实",
            "description": "核心事实为30%。",
            "reason": "单 Ref。",
        }
    )
    _write(source_run / "groups.json", groups)
    drafts = json.loads((source_run / "compile_progress.json").read_text())
    drafts["drafts"]["group-3"] = {
        **drafts["drafts"]["group-1"],
        "ref": {
            **drafts["drafts"]["group-1"]["ref"],
            "concept_id": "group-3",
        },
    }
    _write(source_run / "compile_progress.json", drafts)
    structure = json.loads((structures / "doc.structure.json").read_text())
    structure["blocks"].append(
        {
            "block_id": "block-3",
            "content": "核心事实为30%。无关背景不应进入缺失草稿的编译输入。",
            "page_number": 9,
        }
    )
    _write(structures / "doc.structure.json", structure)


def _add_second_group(source_run: Path, structures: Path) -> None:
    source_progress = json.loads((source_run / "source_progress.json").read_text())
    second = dict(source_progress["sources"][0]["refs"][0])
    second.update(
        {
            "ref_id": "ref-2",
            "article_id": "article-2",
            "local_id": "local-2",
            "evidence_block_ids": ["block-2"],
        }
    )
    source_progress["sources"][0]["refs"].append(second)
    _write(source_run / "source_progress.json", source_progress)
    groups = json.loads((source_run / "groups.json").read_text())
    groups["groups"].append(
        {
            "group_id": "group-2",
            "ref_ids": ["ref-2"],
            "title": "核心事实",
            "description": "核心事实为30%。",
            "reason": "单 Ref。",
        }
    )
    _write(source_run / "groups.json", groups)
    drafts = json.loads((source_run / "compile_progress.json").read_text())
    drafts["drafts"]["group-2"] = {
        **drafts["drafts"]["group-1"],
        "ref": {
            **drafts["drafts"]["group-1"]["ref"],
            "concept_id": "group-2",
        },
    }
    _write(source_run / "compile_progress.json", drafts)
    structure = json.loads((structures / "doc.structure.json").read_text())
    structure["blocks"].append(
        {
            "block_id": "block-2",
            "content": "核心事实为30%。无关背景不应进入缺失草稿的编译输入。",
            "page_number": 8,
        }
    )
    _write(structures / "doc.structure.json", structure)


def test_happy_path_reuses_source_draft_and_never_mutates_source(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    output = tmp_path / "reviewed"
    before = _tree_hashes(source_run)
    client = PassingClient()

    result = run_claim_review(
        client,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=TEMPLATES,
    )

    assert result["status"] == "complete"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["configuration"]["known_source_anomalies"] == []
    assert manifest["configuration"]["client"]["send_chat_template_kwargs"] is False
    assert manifest["configuration"]["client"]["enable_thinking"] is False
    assert [name for name, _prompt in client.calls] == [
        "concept_claim_contract",
        "concept_claim_coverage",
    ]
    contract_prompt = next(
        prompt for name, prompt in client.calls if name == "concept_claim_contract"
    )
    assert contract_prompt.count("无关背景不应进入缺失草稿的编译输入。") == 1
    assert before == _tree_hashes(source_run)
    assert json.loads((output / "reviewed_compile_progress.json").read_text())["accepted_groups"] == ["group-1"]
    snapshot = json.loads((output / "source_snapshot.json").read_text())
    assert set(snapshot["inputs"]) == {
        "source_progress.json",
        "groups.json",
        "compile_progress.json",
    }
    checkpoint = next((output / "checkpoints").glob("*.json"))
    evidence = json.loads(checkpoint.read_text())["evidence_provenance"][0]
    assert evidence["source_blocks"][0]["page_number"] == 7


def test_checkpoint_records_excluded_source_anomaly_provenance(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    structure_path = structures / "doc.structure.json"
    structure = json.loads(structure_path.read_text())
    structure["blocks"][0]["content"] = (
        f"产业集群{KNOWN_ANOMALY}，产业发展能级持续提升。"
        "核心事实为30%。无关背景不应进入缺失草稿的编译输入。"
    )
    _write(structure_path, structure)
    output = tmp_path / "reviewed"
    client = PassingClient()

    result = run_claim_review(
        client,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=TEMPLATES,
        known_source_anomalies=(KNOWN_ANOMALY,),
    )

    assert result["status"] == "complete"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["configuration"]["known_source_anomalies"] == [
        KNOWN_ANOMALY
    ]
    contract_prompt = next(
        prompt for name, prompt in client.calls if name == "concept_claim_contract"
    )
    assert contract_prompt.count(KNOWN_ANOMALY) == 1
    assert f"产业集群{KNOWN_ANOMALY}" not in contract_prompt
    checkpoint = next((output / "checkpoints").glob("*.json"))
    provenance = json.loads(checkpoint.read_text())["evidence_provenance"][0]
    assert KNOWN_ANOMALY in provenance["source_blocks"][0]["text"]
    assert provenance["excluded_fragments"] == [
        {
            "text": f"产业集群{KNOWN_ANOMALY}，产业发展能级持续提升。",
            "block_id": "block-1",
            "page_number": 7,
            "source_text_anomalies": [KNOWN_ANOMALY],
        }
    ]


def test_resume_skips_completed_group_and_rejects_configuration_change(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    output = tmp_path / "reviewed"
    run_claim_review(
        PassingClient(),
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=TEMPLATES,
    )

    resumed = run_claim_review(
        NoCallClient(),
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=TEMPLATES,
        resume=True,
    )
    assert resumed["status"] == "complete"

    changed = NoCallClient()
    changed.model = "different-model"
    with pytest.raises(ClaimReviewRunError, match="configuration changed"):
        run_claim_review(
            changed,
            source_run=source_run,
            output_dir=output,
            structures_dir=structures,
            templates=TEMPLATES,
            resume=True,
        )

    with pytest.raises(ClaimReviewRunError, match="configuration changed"):
        run_claim_review(
            NoCallClient(),
            source_run=source_run,
            output_dir=output,
            structures_dir=structures,
            templates=TEMPLATES,
            resume=True,
            known_source_anomalies=(KNOWN_ANOMALY,),
        )

    changed_template_mode = NoCallClient()
    changed_template_mode.send_chat_template_kwargs = True
    changed_template_mode.enable_thinking = False
    with pytest.raises(ClaimReviewRunError, match="configuration changed"):
        run_claim_review(
            changed_template_mode,
            source_run=source_run,
            output_dir=output,
            structures_dir=structures,
            templates=TEMPLATES,
            resume=True,
        )


def test_missing_source_draft_compiles_only_frozen_required_excerpt(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=False)
    output = tmp_path / "reviewed"
    client = PassingClient()

    result = run_claim_review(
        client,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=TEMPLATES,
    )

    assert result["status"] == "complete"
    compile_prompt = next(
        prompt for name, prompt in client.calls if name == "concept_draft"
    )
    assert "核心事实为30%。" in compile_prompt
    assert "无关背景不应进入缺失草稿的编译输入" not in compile_prompt


def test_formal_mode_refuses_missing_structure_provenance_before_model_call(
    tmp_path: Path,
):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    (structures / "doc.structure.json").unlink()
    client = PassingClient()

    with pytest.raises(ClaimReviewRunError, match="complete structure provenance"):
        run_claim_review(
            client,
            source_run=source_run,
            output_dir=tmp_path / "reviewed",
            structures_dir=structures,
            templates=TEMPLATES,
        )
    assert client.calls == []


def test_probe_group_selector_only_reviews_requested_group(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    _add_second_group(source_run, structures)
    client = PassingClient()

    result = run_claim_review(
        client,
        source_run=source_run,
        output_dir=tmp_path / "probe",
        structures_dir=structures,
        templates=TEMPLATES,
        allow_partial=True,
        selected_group_ids=("group-2",),
    )

    assert result["summary"]["groups"] == 1
    assert [name for name, _prompt in client.calls] == [
        "concept_claim_contract",
        "concept_claim_coverage",
    ]


def test_resume_after_interruption_does_not_repeat_completed_group(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    _add_second_group(source_run, structures)
    output = tmp_path / "reviewed"
    interrupted = FailOnSecondGroupClient()

    first = run_claim_review(
        interrupted,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=TEMPLATES,
        allow_partial=True,
    )
    assert first["status"] == "partial"
    assert [name for name, _prompt in interrupted.calls] == [
        "concept_claim_contract",
        "concept_claim_coverage",
    ]

    resumed = PassingClient()
    final = run_claim_review(
        resumed,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=TEMPLATES,
        allow_partial=True,
        resume=True,
    )
    assert final["status"] == "complete"
    assert [name for name, _prompt in resumed.calls] == [
        "concept_claim_contract",
        "concept_claim_coverage",
    ]


def test_stage_clients_route_evidence_reasoning_away_from_draft_rendering(
    tmp_path: Path,
):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    contract = PassingClient()
    coverage = PassingClient()
    compile_client = PassingClient()
    recompile_client = PassingClient()
    for client, thinking in (
        (contract, True),
        (coverage, True),
        (compile_client, False),
        (recompile_client, False),
    ):
        client.send_chat_template_kwargs = True
        client.enable_thinking = thinking
    routed = ClaimReviewStageClients(
        contract=contract,
        coverage=coverage,
        compile=compile_client,
        recompile=recompile_client,
    )
    output = tmp_path / "reviewed"

    result = run_claim_review(
        routed,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=TEMPLATES,
    )

    assert result["status"] == "complete"
    assert [name for name, _prompt in contract.calls] == [
        "concept_claim_contract"
    ]
    assert [name for name, _prompt in coverage.calls] == [
        "concept_claim_coverage"
    ]
    assert compile_client.calls == []
    assert recompile_client.calls == []
    configuration = json.loads((output / "manifest.json").read_text())[
        "configuration"
    ]["client"]
    assert configuration["routing"] == "schema_stage"
    assert configuration["stages"]["contract"]["enable_thinking"] is True
    assert configuration["stages"]["coverage"]["enable_thinking"] is True
    assert configuration["stages"]["compile"]["enable_thinking"] is False
    assert configuration["stages"]["recompile"]["enable_thinking"] is False


def test_new_run_can_seed_validated_recompile_checkpoint_without_repeating_work(
    tmp_path: Path,
):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    seed_run = tmp_path / "seed-run"
    first = run_claim_review(
        SeedCheckpointClient(),
        source_run=source_run,
        output_dir=seed_run,
        structures_dir=structures,
        templates=TEMPLATES,
        allow_partial=True,
    )
    assert first["status"] == "partial"
    seed_checkpoint = next((seed_run / "checkpoints").glob("*.json"))
    before = seed_checkpoint.read_bytes()

    output = tmp_path / "reviewed"
    client = ContinueSeedClient()
    final = run_claim_review(
        client,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=TEMPLATES,
        allow_partial=True,
        seed_run=seed_run,
    )

    assert final["status"] == "complete"
    assert [name for name, _prompt in client.calls] == [
        "agent_recompiled_concept",
        "concept_claim_coverage",
    ]
    assert seed_checkpoint.read_bytes() == before
    checkpoint = json.loads(next((output / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["status"] == "complete"
    assert checkpoint["decision"] == "pass"
    assert checkpoint["seed_provenance"]["run_name"] == seed_run.name
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["configuration"]["seed"]["run_name"] == seed_run.name

    resumed = run_claim_review(
        NoCallClient(),
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=TEMPLATES,
        allow_partial=True,
        resume=True,
        seed_run=seed_run,
    )
    assert resumed["status"] == "complete"


def test_seed_run_rejects_different_frozen_snapshot_before_model_call(
    tmp_path: Path,
):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    seed_run = tmp_path / "seed-run"
    run_claim_review(
        SeedCheckpointClient(),
        source_run=source_run,
        output_dir=seed_run,
        structures_dir=structures,
        templates=TEMPLATES,
        allow_partial=True,
    )
    snapshot = json.loads((seed_run / "source_snapshot.json").read_text())
    snapshot["inputs"]["groups.json"]["sha256"] = "0" * 64
    _write(seed_run / "source_snapshot.json", snapshot)
    client = NoCallClient()

    with pytest.raises(ClaimReviewRunError, match="different source"):
        run_claim_review(
            client,
            source_run=source_run,
            output_dir=tmp_path / "reviewed",
            structures_dir=structures,
            templates=TEMPLATES,
            allow_partial=True,
            seed_run=seed_run,
        )


def test_full_run_can_sparsely_seed_one_completed_probe_group(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    _add_second_group(source_run, structures)
    seed_run = tmp_path / "probe-seed"
    probe = run_claim_review(
        PassingClient(),
        source_run=source_run,
        output_dir=seed_run,
        structures_dir=structures,
        templates=TEMPLATES,
        allow_partial=True,
        selected_group_ids=("group-1",),
    )
    assert probe["status"] == "complete"

    client = PassingClient()
    output = tmp_path / "formal"
    final = run_claim_review(
        client,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=TEMPLATES,
        seed_run=seed_run,
    )

    assert final["status"] == "complete"
    assert [name for name, _prompt in client.calls] == [
        "concept_claim_contract",
        "concept_claim_coverage",
    ]
    reviewed = json.loads((output / "reviewed_compile_progress.json").read_text())
    assert reviewed["accepted_groups"] == ["group-1", "group-2"]
    seeded = json.loads(
        next(
            path
            for path in (output / "checkpoints").glob("*.json")
            if json.loads(path.read_text())["group_id"] == "group-1"
        ).read_text()
    )
    assert seeded["status"] == "complete"
    assert seeded["decision"] == "pass"
    assert seeded["seed_provenance"]["run_name"] == seed_run.name


def test_recompile_audit_includes_all_claims_and_exact_defect_without_dup_evidence():
    full_mechanism = (
        "《加强重庆成都双核联动联建引领带动成渝地区双城经济圈建设工作机制》"
    )
    full_list = "《强化重庆成都双核联动联建合作项目事项清单》"
    claim = ClaimObligation(
        claim_id="claim-policy",
        ref_id="ref-1",
        evidence_id="ref-1:bundle-test",
        claim=f"会议审议了{full_mechanism}和{full_list}。",
        slot="evidence",
        kind="policy",
        evidence_excerpt=(
            f"会议审议了{full_mechanism}（以下简称《工作机制》）、"
            f"{full_list}（以下简称《事项清单》）。"
        ),
        evidence_block_ids=("block-1",),
        page_numbers=(135,),
        scope={},
    )
    covered_claim = ClaimObligation(
        claim_id="claim-covered",
        ref_id="ref-1",
        evidence_id="ref-1:bundle-test",
        claim="2023年会议在成都召开。",
        slot="evidence",
        kind="fact",
        evidence_excerpt="2023年会议在成都召开。",
        evidence_block_ids=("block-2",),
        page_numbers=(136,),
        scope={},
    )
    contract = ConceptClaimContract(
        group_id="group-policy",
        canonical_question="会议审议了哪些文件？",
        members=(
            MemberContribution(
                ref_id="ref-1",
                relation="supports",
                contribution="给出会议审议文件。",
            ),
        ),
        claims=(claim, covered_claim),
        evidence_units=(
            EvidenceUnitReview(
                evidence_id=claim.evidence_id,
                disposition="required",
                reason="直接回答问题。",
                claim_ids=(claim.claim_id, covered_claim.claim_id),
            ),
        ),
    )
    unsupported_sentence = "会议确立了战略机制。"
    matrix = ClaimCoverageMatrix(
        rows=(
            ClaimCoverageRow(
                claim_id=claim.claim_id,
                status="contradicted",
                draft_excerpt="会议只审议了《工作机制》。",
                finding="草稿只写《工作机制》和《事项清单》简称。",
            ),
            ClaimCoverageRow(
                claim_id=covered_claim.claim_id,
                status="covered",
                draft_excerpt="2023年会议在成都召开。",
                finding="草稿已准确覆盖。",
            ),
        ),
        sentence_attributions=(),
        unsupported_claims=(
            UnsupportedClaim(
                draft_excerpt=unsupported_sentence,
                finding="该判断没有冻结 claim 支持。",
            ),
        ),
        scope_violations=(),
        decision="recompile",
    )

    audit = _quality_audit_for_recompile(matrix, contract)
    rendered_issues = "\n".join(audit.issues)
    checklist = json.loads(audit.issues[0])

    assert claim.claim in rendered_issues
    assert claim.evidence_excerpt not in rendered_issues
    assert "会议只审议了《工作机制》。" in rendered_issues
    assert unsupported_sentence in rendered_issues
    assert checklist == {
        "kind": "all_required_claims_checklist",
        "claims": [
            {"claim_id": claim.claim_id, "claim": claim.claim},
            {
                "claim_id": covered_claim.claim_id,
                "claim": covered_claim.claim,
            },
        ],
    }
    assert '"kind": "claim_coverage_defect"' in rendered_issues
    assert '"kind": "unsupported_draft_sentence"' in rendered_issues
    assert "不得只写简称" in audit.recompile_instructions
    assert "删除给出的完整 draft_excerpt" in audit.recompile_instructions
    assert "未被具体 defect 点名且已经正确支持的草稿句必须原样保留" in (
        audit.recompile_instructions
    )


def _rewrite_draft_body(source_run: Path, body: str) -> None:
    drafts = json.loads((source_run / "compile_progress.json").read_text())
    drafts["drafts"]["group-1"]["body"] = body
    _write(source_run / "compile_progress.json", drafts)


def test_run_rejects_invalid_coverage_batch_size(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    with pytest.raises(ValueError, match="positive integer"):
        run_claim_review(
            NoCallClient(),
            source_run=source_run,
            output_dir=tmp_path / "reviewed",
            structures_dir=structures,
            templates=TEMPLATES,
            coverage_batch_size=0,
        )


def test_manifest_records_coverage_batch_size_and_resume_rejects_change(
    tmp_path: Path,
):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    output = tmp_path / "reviewed"
    result = run_claim_review(
        BatchCoverageClient(),
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=BATCH_TEMPLATES,
        coverage_batch_size=2,
    )
    assert result["status"] == "complete"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["configuration"]["coverage_batch_size"] == 2
    # The batch knob stays top-level so stage-client policy comparison and
    # resume equality keep their existing semantics; it never leaks into the
    # client configuration (including stage-routed clients).
    assert "coverage_batch_size" not in manifest["configuration"]["client"]

    resumed = run_claim_review(
        NoCallClient(),
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=BATCH_TEMPLATES,
        resume=True,
        coverage_batch_size=2,
    )
    assert resumed["status"] == "complete"

    with pytest.raises(ClaimReviewRunError, match="configuration changed"):
        run_claim_review(
            NoCallClient(),
            source_run=source_run,
            output_dir=output,
            structures_dir=structures,
            templates=BATCH_TEMPLATES,
            resume=True,
            coverage_batch_size=3,
        )


def test_coverage_batch_resume_skips_completed_batches_and_matches_full_run(
    tmp_path: Path,
):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    # Only the first batch carries the claim's evidence sentence, so the
    # resumed run re-derives exactly the same rows for the re-run batch.
    _rewrite_draft_body(
        source_run,
        "核心事实为30%。补充说明甲。补充说明乙。补充说明丙。补充说明丁。",
    )
    output = tmp_path / "reviewed"
    interrupted = BatchCoverageClient(interrupt_after=2)
    first = run_claim_review(
        interrupted,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=BATCH_TEMPLATES,
        coverage_batch_size=2,
        allow_partial=True,
    )
    assert first["status"] == "partial"
    assert len(interrupted.coverage_sentence_ids) == 2
    checkpoint = json.loads(next((output / "checkpoints").glob("*.json")).read_text())
    assert sorted(checkpoint["coverage_batches"]) == ["0", "1"]
    assert checkpoint["status"] == "failed"

    resumed = BatchCoverageClient()
    final = run_claim_review(
        resumed,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=BATCH_TEMPLATES,
        coverage_batch_size=2,
        allow_partial=True,
        resume=True,
    )
    assert final["status"] == "complete"
    # Only the missing batch was re-run; completed batches were not repeated.
    assert len(resumed.coverage_sentence_ids) == 1

    clean_client = BatchCoverageClient()
    clean = run_claim_review(
        clean_client,
        source_run=source_run,
        output_dir=tmp_path / "clean",
        structures_dir=structures,
        templates=BATCH_TEMPLATES,
        coverage_batch_size=2,
        allow_partial=True,
    )
    assert clean["status"] == "complete"
    assert len(clean_client.coverage_sentence_ids) == 3
    assert resumed.coverage_sentence_ids == [
        clean_client.coverage_sentence_ids[2]
    ]
    clean_checkpoint = json.loads(
        next((tmp_path / "clean" / "checkpoints").glob("*.json")).read_text()
    )
    resumed_checkpoint = json.loads(
        next((output / "checkpoints").glob("*.json")).read_text()
    )
    assert resumed_checkpoint["status"] == "complete"
    assert resumed_checkpoint["decision"] == "pass"
    assert resumed_checkpoint["coverage"] == clean_checkpoint["coverage"]


def test_recompile_invalidates_batch_records_and_rebatches(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    _rewrite_draft_body(source_run, "核心事实为30%。背景说明。")
    output = tmp_path / "reviewed"
    result = run_claim_review(
        RecompileChangesDraftClient(),
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=BATCH_TEMPLATES,
        coverage_batch_size=2,
    )
    assert result["status"] == "complete"
    checkpoint = json.loads(next((output / "checkpoints").glob("*.json")).read_text())
    assert checkpoint["status"] == "complete"
    assert checkpoint["decision"] == "pass"
    assert checkpoint["recompile_attempts"] == 1
    assert "补充说明甲。补充说明乙。补充说明丙。" in checkpoint["draft"]["body"]

    new_draft = {
        "title": checkpoint["draft"]["title"],
        "description": checkpoint["draft"]["description"],
        "body": checkpoint["draft"]["body"],
    }
    old_draft = {
        "title": "核心事实",
        "description": "核心事实为30%。",
        "body": "核心事实为30%。背景说明。",
    }
    new_sentence_ids = {
        item.sentence_id for item in build_draft_sentences(new_draft)
    }
    old_distinctive_ids = {
        item.sentence_id
        for item in build_draft_sentences(old_draft)
        if "背景说明" in item.text
    }
    batch_sentence_ids = {
        item["sentence_id"]
        for record in checkpoint["coverage_batches"].values()
        for item in record["sentence_attributions"]
    }
    assert len(checkpoint["coverage_batches"]) == 2
    assert batch_sentence_ids == new_sentence_ids
    assert not (batch_sentence_ids & old_distinctive_ids)


def test_seed_reuses_complete_coverage_with_relaxed_coverage_prompt(
    tmp_path: Path,
):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    seed_run = tmp_path / "seed-run"
    run_claim_review(
        BatchCoverageClient(),
        source_run=source_run,
        output_dir=seed_run,
        structures_dir=structures,
        templates=BATCH_TEMPLATES,
        coverage_batch_size=2,
    )
    seed_manifest = json.loads((seed_run / "manifest.json").read_text())
    assert seed_manifest["configuration"]["coverage_batch_size"] == 2

    relaxed_templates = ClaimReviewTemplates(
        contract=TEMPLATES.contract,
        coverage=BATCH_TEMPLATES.coverage + "\n## EXTRA\nbatch-coverage-rewrite",
        compile=TEMPLATES.compile,
        recompile=TEMPLATES.recompile,
    )
    client = NoCallClient()
    output = tmp_path / "reviewed"
    final = run_claim_review(
        client,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=relaxed_templates,
        coverage_batch_size=2,
        seed_run=seed_run,
    )
    assert final["status"] == "complete"
    assert client.calls == []
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["configuration"]["seed"]["run_name"] == seed_run.name
    assert manifest["configuration"]["seed"]["coverage_prompt_relaxed"] is True
    seeded = json.loads(next((output / "checkpoints").glob("*.json")).read_text())
    assert seeded["status"] == "complete"
    assert seeded["decision"] == "pass"


@pytest.mark.parametrize("stage", ["contract", "compile", "recompile"])
def test_seed_accepts_non_coverage_prompt_drift_and_records_relaxed(
    tmp_path: Path, stage: str
):
    # A repair run strengthens the Contract/Compile/Recompile prompts while
    # inheriting the audited pass checkpoints; each observed drift is recorded
    # next to coverage_prompt_relaxed instead of rejecting the whole seed.
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    seed_run = tmp_path / "seed-run"
    run_claim_review(
        BatchCoverageClient(),
        source_run=source_run,
        output_dir=seed_run,
        structures_dir=structures,
        templates=BATCH_TEMPLATES,
        coverage_batch_size=2,
    )
    changed = ClaimReviewTemplates(
        contract=(
            TEMPLATES.contract
            if stage != "contract"
            else TEMPLATES.contract + " changed"
        ),
        coverage=BATCH_TEMPLATES.coverage,
        compile=(
            TEMPLATES.compile
            if stage != "compile"
            else TEMPLATES.compile + " changed"
        ),
        recompile=(
            TEMPLATES.recompile
            if stage != "recompile"
            else TEMPLATES.recompile + " changed"
        ),
    )
    client = NoCallClient()
    output = tmp_path / "reviewed"
    final = run_claim_review(
        client,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=changed,
        coverage_batch_size=2,
        seed_run=seed_run,
    )
    assert final["status"] == "complete"
    assert client.calls == []
    seeded = json.loads(next((output / "checkpoints").glob("*.json")).read_text())
    assert seeded["status"] == "complete"
    assert seeded["decision"] == "pass"
    manifest = json.loads((output / "manifest.json").read_text())
    seed_config = manifest["configuration"]["seed"]
    assert seed_config[f"{stage}_prompt_relaxed"] is True
    assert seed_config["coverage_prompt_relaxed"] is False


def test_seed_prompt_relaxations_combine_across_stages(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    seed_run = tmp_path / "seed-run"
    run_claim_review(
        BatchCoverageClient(),
        source_run=source_run,
        output_dir=seed_run,
        structures_dir=structures,
        templates=BATCH_TEMPLATES,
        coverage_batch_size=2,
    )
    changed = ClaimReviewTemplates(
        contract=TEMPLATES.contract + " contract-repair",
        coverage=BATCH_TEMPLATES.coverage,
        compile=TEMPLATES.compile,
        recompile=TEMPLATES.recompile + " recompile-repair",
    )
    final = run_claim_review(
        NoCallClient(),
        source_run=source_run,
        output_dir=tmp_path / "reviewed",
        structures_dir=structures,
        templates=changed,
        coverage_batch_size=2,
        seed_run=seed_run,
    )
    assert final["status"] == "complete"
    seed_config = json.loads(
        (tmp_path / "reviewed" / "manifest.json").read_text()
    )["configuration"]["seed"]
    assert seed_config["contract_prompt_relaxed"] is True
    assert seed_config["recompile_prompt_relaxed"] is True
    assert seed_config["compile_prompt_relaxed"] is False
    assert seed_config["coverage_prompt_relaxed"] is False


def test_seed_accepts_model_id_drift_and_records_model_relaxed(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    seed_run = tmp_path / "seed-run"
    run_claim_review(
        PassingClient(),
        source_run=source_run,
        output_dir=seed_run,
        structures_dir=structures,
        templates=TEMPLATES,
    )
    client = NoCallClient()
    client.model = "qwen3p6-35b-a3b"

    final = run_claim_review(
        client,
        source_run=source_run,
        output_dir=tmp_path / "reviewed",
        structures_dir=structures,
        templates=TEMPLATES,
        seed_run=seed_run,
    )

    assert final["status"] == "complete"
    assert client.calls == []
    manifest = json.loads((tmp_path / "reviewed" / "manifest.json").read_text())
    assert manifest["configuration"]["seed"]["model_relaxed"] is True
    assert manifest["configuration"]["seed"]["coverage_prompt_relaxed"] is False


def test_seed_rejects_non_model_client_policy_drift(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    seed_run = tmp_path / "seed-run"
    run_claim_review(
        PassingClient(),
        source_run=source_run,
        output_dir=seed_run,
        structures_dir=structures,
        templates=TEMPLATES,
    )
    client = NoCallClient()
    client.model = "qwen3p6-35b-a3b"
    client.send_chat_template_kwargs = True

    with pytest.raises(ClaimReviewRunError, match="different stage client policy"):
        run_claim_review(
            client,
            source_run=source_run,
            output_dir=tmp_path / "reviewed",
            structures_dir=structures,
            templates=TEMPLATES,
            seed_run=seed_run,
        )


def test_seed_model_and_coverage_relaxation_combine(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    seed_run = tmp_path / "seed-run"

    def routed(model: str, *, coverage_thinking: bool) -> ClaimReviewStageClients:
        contract = PassingClient()
        coverage = PassingClient()
        compile_client = PassingClient()
        recompile_client = PassingClient()
        for client in (contract, coverage, compile_client, recompile_client):
            client.model = model
            client.send_chat_template_kwargs = True
        contract.enable_thinking = True
        coverage.enable_thinking = coverage_thinking
        coverage.max_tokens = 16384
        compile_client.enable_thinking = False
        recompile_client.enable_thinking = False
        return ClaimReviewStageClients(
            contract=contract,
            coverage=coverage,
            compile=compile_client,
            recompile=recompile_client,
        )

    run_claim_review(
        routed("fake-model", coverage_thinking=True),
        source_run=source_run,
        output_dir=seed_run,
        structures_dir=structures,
        templates=TEMPLATES,
    )

    final = run_claim_review(
        routed("qwen3p6-35b-a3b", coverage_thinking=False),
        source_run=source_run,
        output_dir=tmp_path / "reviewed",
        structures_dir=structures,
        templates=TEMPLATES,
        seed_run=seed_run,
    )

    assert final["status"] == "complete"
    manifest = json.loads((tmp_path / "reviewed" / "manifest.json").read_text())
    assert manifest["configuration"]["seed"]["model_relaxed"] is True
    assert manifest["configuration"]["seed"]["coverage_prompt_relaxed"] is False


def _routed_client(
    model: str,
    *,
    contract_thinking: bool,
    coverage_thinking: bool,
    contract_max_tokens: int = 8192,
) -> ClaimReviewStageClients:
    contract = PassingClient()
    coverage = PassingClient()
    compile_client = PassingClient()
    recompile_client = PassingClient()
    for client in (contract, coverage, compile_client, recompile_client):
        client.model = model
        client.send_chat_template_kwargs = True
        client.max_tokens = contract_max_tokens
    contract.enable_thinking = contract_thinking
    coverage.enable_thinking = coverage_thinking
    coverage.max_tokens = 16384
    compile_client.enable_thinking = False
    recompile_client.enable_thinking = False
    return ClaimReviewStageClients(
        contract=contract,
        coverage=coverage,
        compile=compile_client,
        recompile=recompile_client,
    )


def test_seed_accepts_thinking_drift_and_records_thinking_relaxed(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    seed_run = tmp_path / "seed-run"
    run_claim_review(
        _routed_client(
            "fake-model",
            contract_thinking=True,
            coverage_thinking=True,
        ),
        source_run=source_run,
        output_dir=seed_run,
        structures_dir=structures,
        templates=TEMPLATES,
    )

    final = run_claim_review(
        _routed_client(
            "fake-model",
            contract_thinking=False,
            coverage_thinking=False,
        ),
        source_run=source_run,
        output_dir=tmp_path / "reviewed",
        structures_dir=structures,
        templates=TEMPLATES,
        seed_run=seed_run,
    )

    assert final["status"] == "complete"
    manifest = json.loads((tmp_path / "reviewed" / "manifest.json").read_text())
    assert manifest["configuration"]["seed"]["thinking_relaxed"] is True
    assert manifest["configuration"]["seed"]["model_relaxed"] is False


def test_seed_rejects_contract_max_tokens_drift(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    seed_run = tmp_path / "seed-run"
    run_claim_review(
        _routed_client(
            "fake-model",
            contract_thinking=False,
            coverage_thinking=False,
        ),
        source_run=source_run,
        output_dir=seed_run,
        structures_dir=structures,
        templates=TEMPLATES,
    )

    with pytest.raises(ClaimReviewRunError, match="different stage client policy"):
        run_claim_review(
            _routed_client(
                "fake-model",
                contract_thinking=False,
                coverage_thinking=False,
                contract_max_tokens=16384,
            ),
            source_run=source_run,
            output_dir=tmp_path / "reviewed",
            structures_dir=structures,
            templates=TEMPLATES,
            seed_run=seed_run,
        )


def test_seed_accepts_model_and_thinking_drift_together(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    seed_run = tmp_path / "seed-run"
    run_claim_review(
        _routed_client(
            "fake-model",
            contract_thinking=True,
            coverage_thinking=True,
        ),
        source_run=source_run,
        output_dir=seed_run,
        structures_dir=structures,
        templates=TEMPLATES,
    )

    final = run_claim_review(
        _routed_client(
            "qwen3p6-35b-a3b",
            contract_thinking=False,
            coverage_thinking=False,
        ),
        source_run=source_run,
        output_dir=tmp_path / "reviewed",
        structures_dir=structures,
        templates=TEMPLATES,
        seed_run=seed_run,
    )

    assert final["status"] == "complete"
    manifest = json.loads((tmp_path / "reviewed" / "manifest.json").read_text())
    assert manifest["configuration"]["seed"]["model_relaxed"] is True
    assert manifest["configuration"]["seed"]["thinking_relaxed"] is True


def test_cli_plain_mode_with_coverage_max_tokens_builds_stage_clients(
    tmp_path: Path,
):
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "review_concept_claims.py"
    )
    spec = importlib.util.spec_from_file_location(
        "review_concept_claims", str(script_path)
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    calls: list[tuple[bool, int | None]] = []

    def make_client(*, enable_thinking: bool, max_tokens: int | None = None):
        calls.append((enable_thinking, max_tokens))
        client = PassingClient()
        client.enable_thinking = enable_thinking
        if max_tokens is not None:
            client.max_tokens = max_tokens
        return client

    args = type(
        "Args",
        (),
        {
            "evidence_stage_thinking": False,
            "contract_stage_thinking": False,
            "coverage_max_tokens": 16384,
            "enable_thinking": False,
        },
    )()
    client = cli._stage_clients(args, make_client)

    assert isinstance(client, ClaimReviewStageClients)
    assert calls == [(False, None), (False, 16384), (False, None), (False, None)]
    assert client.contract.enable_thinking is False
    assert client.coverage.enable_thinking is False
    assert client.coverage.max_tokens == 16384
    assert client.compile.enable_thinking is False
    assert client.recompile.enable_thinking is False

    plain_args = type(
        "Args",
        (),
        {
            "evidence_stage_thinking": False,
            "contract_stage_thinking": False,
            "coverage_max_tokens": None,
            "enable_thinking": True,
        },
    )()
    plain = cli._stage_clients(plain_args, make_client)
    assert not isinstance(plain, ClaimReviewStageClients)


def test_formal_mode_records_failed_group_and_continues(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    _add_second_group(source_run, structures)
    _add_third_group(source_run, structures)
    output = tmp_path / "formal"
    client = PersistentContractFailureClient()

    # Formal mode processes every group: group-2 exhausts its retries with a
    # persistent non-verbatim evidence excerpt, is recorded as failed, and the
    # run continues to group-3.  The run still ends with the formal gate
    # raising because not every group passed.
    with pytest.raises(ClaimReviewRunError, match="did not pass every group"):
        run_claim_review(
            client,
            source_run=source_run,
            output_dir=output,
            structures_dir=structures,
            templates=TEMPLATES,
        )

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "partial"
    assert manifest["summary"]["accepted"] == 2
    assert manifest["summary"]["completed"] == 2
    assert manifest["summary"]["withheld"] == 1
    reviewed = json.loads((output / "reviewed_compile_progress.json").read_text())
    assert reviewed["accepted_groups"] == ["group-1", "group-3"]
    queue = json.loads((output / "review_queue.json").read_text())
    failed = [
        item for item in queue["reviews"] if item.get("group_id") == "group-2"
    ]
    assert len(failed) == 1
    assert failed[0]["decision"] == "failed"
    assert "not verbatim evidence" in failed[0]["reason"]
    checkpoint = json.loads(
        next(
            path
            for path in (output / "checkpoints").glob("*.json")
            if json.loads(path.read_text())["group_id"] == "group-2"
        ).read_text()
    )
    assert checkpoint["status"] == "failed"
    assert "not verbatim evidence" in checkpoint["error"]
    # group-3 was still processed after the failure (3 contract attempts for
    # group-2 plus one contract each for group-1 and group-3).
    assert [name for name, _prompt in client.calls].count(
        "concept_claim_contract"
    ) == 5

    # Resume retries the failed group and re-records it without interrupting
    # the already completed groups.
    resumed_client = PersistentContractFailureClient()
    with pytest.raises(ClaimReviewRunError, match="did not pass every group"):
        run_claim_review(
            resumed_client,
            source_run=source_run,
            output_dir=output,
            structures_dir=structures,
            templates=TEMPLATES,
            resume=True,
        )
    assert [name for name, _prompt in resumed_client.calls].count(
        "concept_claim_contract"
    ) == 3
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["summary"]["accepted"] == 2
    assert manifest["summary"]["withheld"] == 1


def _override_payload(*, title: str = "修复标题") -> dict[str, str]:
    return {
        "title": title,
        "description": "核心事实为30%。",
        "body": "## 核心判断\n\n核心事实为30%。",
    }


def test_draft_override_used_when_source_draft_missing(tmp_path: Path):
    source_run, structures = _source_fixture(tmp_path, with_draft=False)
    before = _tree_hashes(source_run)
    client = PassingClient()

    result = run_claim_review(
        client,
        source_run=source_run,
        output_dir=tmp_path / "reviewed",
        structures_dir=structures,
        templates=TEMPLATES,
        draft_overrides={"group-1": _override_payload()},
    )

    assert result["status"] == "complete"
    assert [name for name, _prompt in client.calls] == [
        "concept_claim_contract",
        "concept_claim_coverage",
    ]
    # The override replaces the compile stage entirely.
    assert "concept_draft" not in [name for name, _prompt in client.calls]
    checkpoint = json.loads(
        next((tmp_path / "reviewed" / "checkpoints").glob("*.json")).read_text()
    )
    assert checkpoint["draft_origin"] == "repair_override"
    assert checkpoint["draft"]["title"] == "修复标题"
    assert checkpoint["recompile_attempts"] == 0
    assert before == _tree_hashes(source_run)
    manifest = json.loads(
        (tmp_path / "reviewed" / "manifest.json").read_text()
    )
    assert manifest["configuration"]["draft_overrides"] == {
        "group-1": _override_payload()
    }


def test_draft_override_precedes_source_draft_and_persists_origin(
    tmp_path: Path,
):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    result = run_claim_review(
        PassingClient(),
        source_run=source_run,
        output_dir=tmp_path / "reviewed",
        structures_dir=structures,
        templates=TEMPLATES,
        draft_overrides={"group-1": _override_payload()},
    )
    assert result["status"] == "complete"
    checkpoint = json.loads(
        next((tmp_path / "reviewed" / "checkpoints").glob("*.json")).read_text()
    )
    assert checkpoint["draft_origin"] == "repair_override"
    assert checkpoint["draft"]["title"] == "修复标题"


def test_checkpoint_draft_takes_priority_over_override_on_resume(
    tmp_path: Path,
):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    output = tmp_path / "reviewed"
    first = run_claim_review(
        InterruptedRecompileClient(),
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=BATCH_TEMPLATES,
        coverage_batch_size=2,
        allow_partial=True,
        draft_overrides={"group-1": _override_payload()},
    )
    assert first["status"] == "partial"
    checkpoint_path = next((output / "checkpoints").glob("*.json"))
    checkpoint = json.loads(checkpoint_path.read_text())
    # The override was injected once, then a recompile replaced the draft.
    assert checkpoint["draft_origin"] == "repair_override"
    assert checkpoint["recompile_attempts"] == 1
    assert checkpoint["draft"]["title"] == "核心事实"
    assert "补充说明甲。补充说明乙。补充说明丙。" in checkpoint["draft"]["body"]

    resumed = BatchCoverageClient()
    final = run_claim_review(
        resumed,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=BATCH_TEMPLATES,
        coverage_batch_size=2,
        allow_partial=True,
        resume=True,
        draft_overrides={"group-1": _override_payload()},
    )
    assert final["status"] == "complete"
    # The persisted checkpoint draft wins: the override is not re-injected.
    assert resumed.calls == []
    assert resumed.coverage_calls == 2
    final_checkpoint = json.loads(checkpoint_path.read_text())
    assert final_checkpoint["status"] == "complete"
    assert final_checkpoint["decision"] == "pass"
    assert final_checkpoint["draft"]["title"] == "核心事实"
    assert "修复标题" not in final_checkpoint["draft"]["title"]
    assert "补充说明甲。补充说明乙。补充说明丙。" in final_checkpoint["draft"]["body"]


def test_seed_reopen_with_override_replaces_draft_and_clears_coverage(
    tmp_path: Path,
):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    seed_run = tmp_path / "seed-run"
    run_claim_review(
        BudgetExhaustedClient(),
        source_run=source_run,
        output_dir=seed_run,
        structures_dir=structures,
        templates=BATCH_TEMPLATES,
        coverage_batch_size=2,
        allow_partial=True,
    )
    seed_checkpoint_path = next((seed_run / "checkpoints").glob("*.json"))
    seed_checkpoint = json.loads(seed_checkpoint_path.read_text())
    assert seed_checkpoint["status"] == "complete"
    assert seed_checkpoint["decision"] == "human_review"
    assert seed_checkpoint["review_reason"] == "recompile_budget_exhausted"
    assert seed_checkpoint["recompile_attempts"] == 2
    seed_bytes = seed_checkpoint_path.read_bytes()

    client = BatchCoverageClient()
    output = tmp_path / "reviewed"
    final = run_claim_review(
        client,
        source_run=source_run,
        output_dir=output,
        structures_dir=structures,
        templates=BATCH_TEMPLATES,
        coverage_batch_size=2,
        seed_run=seed_run,
        draft_overrides={"group-1": _override_payload()},
    )
    assert final["status"] == "complete"
    # Only the fresh coverage audit runs; Contract/Compile/Recompile are
    # inherited from the seed checkpoint and never repeated.
    assert client.calls == []
    assert client.coverage_calls == 1
    assert seed_checkpoint_path.read_bytes() == seed_bytes
    checkpoint = json.loads(
        next((output / "checkpoints").glob("*.json")).read_text()
    )
    assert checkpoint["status"] == "complete"
    assert checkpoint["decision"] == "pass"
    assert checkpoint["draft_origin"] == "repair_override"
    assert checkpoint["draft"]["title"] == "修复标题"
    assert checkpoint["recompile_attempts"] == 0
    # The inherited audit trail was cleared: only the fresh override audit
    # survives, and the repaired draft was audited from scratch.
    assert len(checkpoint["coverage_history"]) == 1
    assert checkpoint["coverage_draft_sha256"] == stable_hash(
        checkpoint["draft"]
    )


@pytest.mark.parametrize(
    "override, match",
    [
        (
            {"unknown-group": _override_payload()},
            "unknown group",
        ),
        (
            {"group-1": {"title": "t", "description": "d"}},
            "non-empty body",
        ),
        (
            {"group-1": {"title": "", "description": "d", "body": "b"}},
            "non-empty title",
        ),
        ({"group-1": "not-an-object"}, "must be an object"),
        ({"": _override_payload()}, "non-empty strings"),
    ],
)
def test_invalid_draft_override_rejected(tmp_path: Path, override, match):
    source_run, structures = _source_fixture(tmp_path, with_draft=True)
    with pytest.raises(ClaimReviewRunError, match=match):
        run_claim_review(
            NoCallClient(),
            source_run=source_run,
            output_dir=tmp_path / "reviewed",
            structures_dir=structures,
            templates=TEMPLATES,
            draft_overrides=override,
        )
