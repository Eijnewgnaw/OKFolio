import json
from pathlib import Path

from okfolio.agentwiki.global_cluster import CandidateEdge
from okfolio.agentwiki.llm import LLMError
from okfolio.agentwiki.okf import parse_concept_markdown
from scripts.system_experiment import RefRecord, _compile_concepts, _load_or_judge


class SplitOnlyClient:
    """Models a timeout on a multi-edge request and success after bisection."""

    def complete(self, _prompt: str, *, json_schema_name: str, json_schema: dict):
        assert json_schema_name == "edge_judgements"
        size = json_schema["properties"]["judgements"]["maxItems"]
        if size > 1:
            raise LLMError("request failed after 2 attempts: ReadTimeout")
        return json.dumps({"judgements": [{"decision": "complementary", "reason": "test"}]})


def test_timeout_batch_is_bisected_and_all_successful_edges_are_persisted(tmp_path: Path):
    refs = [
        RefRecord("a", "article-1", "a", "政策建议", "A", "A", ("证据 A",), ()),
        RefRecord("b", "article-2", "b", "政策建议", "B", "B", ("证据 B",), ()),
        RefRecord("c", "article-3", "c", "政策建议", "C", "C", ("证据 C",), ()),
    ]
    edges = [
        CandidateEdge("edge:a:b", "a", "b", {}),
        CandidateEdge("edge:a:c", "a", "c", {}),
    ]

    judged = _load_or_judge(tmp_path, SplitOnlyClient(), refs, edges, resume=False, dry_run=False)

    assert [item["edge_id"] for item in judged] == ["edge:a:b", "edge:a:c"]
    assert all(item["decision"] == "complementary" for item in judged)
    assert not (tmp_path / "judgement_failures.json").exists()


class PartialCompileClient:
    def complete(self, _prompt: str, *, json_schema_name: str, json_schema: dict):
        assert json_schema_name == "joint_concept"
        if "fail-me" in _prompt:
            raise LLMError("request failed after 3 attempts: ReadTimeout")
        return json.dumps({"title": "标题", "description": "摘要", "body": "## 正文\n\n内容"})


def test_concept_timeout_is_deferred_without_discarding_other_concepts(tmp_path: Path):
    (tmp_path / "concepts").mkdir()
    refs = [
        RefRecord("a", "article-1", "a", "政策建议", "A", "A", ("证据 A",), ()),
        RefRecord("b", "article-2", "b", "政策建议", "B", "B", ("证据 B",), ()),
        RefRecord("c", "article-3", "c", "政策建议", "C", "C", ("证据 C",), ()),
        RefRecord("d", "article-4", "d", "政策建议", "D", "D", ("证据 D",), ()),
    ]
    clusters = [
        {"id": "ok", "kind": "Proposition", "type": "政策建议", "title": "ok", "description": "ok", "ref_ids": ["a", "b"], "article_count": 2},
        {"id": "fail-me", "kind": "Proposition", "type": "政策建议", "title": "fail", "description": "fail", "ref_ids": ["c", "d"], "article_count": 2},
    ]

    concepts = _compile_concepts(tmp_path, PartialCompileClient(), refs, clusters, resume=False)

    assert [item["id"] for item in concepts] == ["ok"]
    assert (tmp_path / "compile_failures.json").exists()


class NeverCallClient:
    def complete(self, *_args, **_kwargs):
        raise AssertionError("a singleton Concept must not call the LLM")


def test_singleton_ref_is_published_as_valid_okf_without_model_call(tmp_path: Path):
    (tmp_path / "concepts").mkdir()
    refs = [
        RefRecord(
            "article-1:metric",
            "article-1",
            "metric",
            "数据口径",
            "指标口径",
            "指标的统计定义。",
            ("原文证据。",),
            (),
        )
    ]
    clusters = [
        {
            "id": "evidence-1",
            "kind": "Evidence",
            "type": "数据口径",
            "title": "指标口径",
            "description": "指标的统计定义。",
            "ref_ids": ["article-1:metric"],
            "article_count": 1,
        }
    ]

    concepts = _compile_concepts(
        tmp_path, NeverCallClient(), refs, clusters, resume=False
    )

    assert len(concepts) == 1
    content = (tmp_path / "concepts" / "evidence-1.md").read_text(
        encoding="utf-8"
    )
    parsed = parse_concept_markdown("evidence-1.md", content)
    assert parsed.frontmatter["type"] == "数据口径"
    assert parsed.frontmatter["provenance_refs"] == ["article-1:metric"]
