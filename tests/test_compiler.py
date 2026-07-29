from __future__ import annotations

import json
from pathlib import Path

import pytest

from kmpro_wiki.agentwiki.compiler import CompilationBatchError, Compiler
from kmpro_wiki.agentwiki.config import Settings
from kmpro_wiki.agentwiki.state import Manifest


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        json_schema_name: str | None = None,
        json_schema: dict | None = None,
    ) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def discovery_json(
    concept_id: str = "a",
    *,
    title: str = "概念 A",
    evidence: str = "evidence-0001",
    asset_hints: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "concepts": [
                {
                    "id": concept_id,
                    "type": "分析框架",
                    "title": title,
                    "description": f"{title}摘要。",
                    "evidence": [evidence],
                    "asset_hints": asset_hints or [],
                }
            ]
        },
        ensure_ascii=False,
    )


def discovery_json_for(ids: tuple[str, ...]) -> str:
    return json.dumps(
        {
            "concepts": [
                {
                    "id": concept_id,
                    "type": "分析框架",
                    "title": f"概念 {concept_id}",
                    "description": f"概念 {concept_id}摘要。",
                    "evidence": ["evidence-0001"],
                    "asset_hints": [],
                }
                for concept_id in ids
            ]
        },
        ensure_ascii=False,
    )


def draft_json(*, title: str = "概念 A", body: str = "正文锚点。") -> str:
    return json.dumps(
        {
            "title": title,
            "description": f"{title}摘要。",
            "sections": [
                {"heading": "正文", "paragraphs": [body], "bullets": []}
            ],
        },
        ensure_ascii=False,
    )


def placement_json(*, concept_id: str = "a") -> str:
    return json.dumps(
        {
            "placements": [
                {
                    "asset_id": "image-001",
                    "concept_id": concept_id,
                    "anchor_id": "anchor-001",
                    "position": "after",
                    "reason": "相关图",
                }
            ]
        },
        ensure_ascii=False,
    )


def no_links_json() -> str:
    return '{"status":"no_links","links":[]}'


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.data = root / "data"
        self.sources = self.data / "sources"
        self.images = self.sources / "images"
        self.wiki = self.data / "wiki"
        self.concepts = self.wiki / "concepts"
        self.prompts = root / "prompts"
        self.images.mkdir(parents=True)
        self.concepts.mkdir(parents=True)
        self.prompts.mkdir(parents=True)
        (self.prompts / "discover.md").write_text(
            "STAGE=discover {title} {source_file} {asset_inventory} "
            "{evidence_catalog}",
            encoding="utf-8",
        )
        (self.prompts / "compile.md").write_text(
            "STAGE=compile {concept_ref} {evidence}", encoding="utf-8"
        )
        (self.prompts / "preserve.md").write_text(
            "STAGE=preserve {asset_inventory} {concepts}", encoding="utf-8"
        )
        (self.prompts / "enrich.md").write_text(
            "STAGE=enrich CURRENT={current_concept} CANDIDATES={candidate_index}",
            encoding="utf-8",
        )

    def source(self, name: str, content: str) -> None:
        (self.sources / name).write_text(content, encoding="utf-8")

    def compiler(self, llm: FakeLLM, *, events: list[str] | None = None) -> Compiler:
        settings = Settings(
            data_dir=self.data,
            prompts_dir=self.prompts,
            llm_api_base="http://unused/v1",
            llm_api_key="unused",
            llm_model="test-model",
        )
        return Compiler(
            settings, llm, on_event=None if events is None else events.append
        )


@pytest.fixture
def project(tmp_path: Path) -> Project:
    return Project(tmp_path)


def test_cold_start_runs_four_stages_and_publishes_valid_concept(project: Project):
    project.images.joinpath("x.jpg").write_bytes(b"image")
    project.source("a.md", "# A\n正文锚点。\n\n![](images/x.jpg)")
    fake = FakeLLM(
        [
            discovery_json(asset_hints=["image-001"]),
            draft_json(),
            placement_json(),
            no_links_json(),
        ]
    )

    summary = project.compiler(fake).run()

    assert summary.compiled == ("a.md",)
    assert len(fake.prompts) == 3
    output = (project.concepts / "a.md").read_text(encoding="utf-8")
    assert "source: a.md" in output
    assert "![](../images/x.jpg)" in output
    state = Manifest.load(project.data / ".state/manifest.json").sources["a.md"]
    assert state.status == "complete"
    assert state.discovery_status == "success"
    assert state.concept_status == "success"
    assert state.preservation_status == "success"
    assert state.relation_status == "no_links"


def test_no_change_makes_zero_llm_calls_and_writes(project: Project):
    project.source("a.md", "# A\n正文锚点。")
    project.compiler(
        FakeLLM([discovery_json(), draft_json(), no_links_json()])
    ).run()
    before = tree_bytes(project.wiki)
    fake = FakeLLM([])

    summary = project.compiler(fake).run()

    assert summary.skipped == ("a.md",)
    assert fake.prompts == []
    assert tree_bytes(project.wiki) == before


def test_incremental_new_source_links_to_old_without_rewriting_old(project: Project):
    project.source("a.md", "# A\n正文锚点。")
    project.compiler(
        FakeLLM(
            [
                discovery_json(title="已有概念定义"),
                draft_json(title="已有概念定义"),
                no_links_json(),
            ]
        )
    ).run()
    old_bytes = (project.concepts / "a.md").read_bytes()
    project.source("b.md", "# B\n已有概念定义是约束。")
    relation = json.dumps(
        {
            "status": "linked",
                "links": [
                    {
                        "anchor_id": "a--anchor-001",
                        "reason": "约束关系",
                    }
            ],
        },
        ensure_ascii=False,
    )

    summary = project.compiler(
        FakeLLM(
            [
                discovery_json(
                    "b", title="概念 B"
                ),
                draft_json(title="概念 B", body="已有概念定义是约束。"),
                relation,
            ]
        )
    ).run()

    assert summary.compiled == ("b.md",)
    assert (project.concepts / "a.md").read_bytes() == old_bytes
    output = (project.concepts / "b.md").read_text(encoding="utf-8")
    assert "[已有概念定义](../concepts/a.md)" in output


def test_invalid_discovery_keeps_previous_output_and_records_incomplete(project: Project):
    project.source("a.md", "# A\n正文锚点。")
    project.compiler(
        FakeLLM([discovery_json(), draft_json(), no_links_json()])
    ).run()
    previous = (project.concepts / "a.md").read_bytes()
    project.source("a.md", "# A\n正文锚点。\n修改。")

    with pytest.raises(CompilationBatchError):
        project.compiler(FakeLLM(["invalid response"])).run()

    assert (project.concepts / "a.md").read_bytes() == previous
    state = Manifest.load(project.data / ".state/manifest.json").sources["a.md"]
    assert state.status == "incomplete"
    traces = list((project.data / ".staging/failures").glob("*.txt"))
    assert len(traces) == 1
    assert "invalid response" in traces[0].read_text(encoding="utf-8")


def test_invalid_relation_is_failure_not_fallback(project: Project):
    project.source("a.md", "# A\n共享指标影响当前分析。")
    discovery = json.dumps(
        {
            "concepts": [
                {
                    "id": "a",
                    "type": "数据口径",
                    "title": "共享指标",
                    "description": "定义共享指标。",
                    "evidence": ["evidence-0001"],
                    "asset_hints": [],
                },
                {
                    "id": "b",
                    "type": "分析框架",
                    "title": "当前分析",
                    "description": "分析共享指标影响。",
                    "evidence": ["evidence-0001"],
                    "asset_hints": [],
                },
            ]
        },
        ensure_ascii=False,
    )

    with pytest.raises(CompilationBatchError):
        project.compiler(
            FakeLLM(
                [
                    discovery,
                    draft_json(title="共享指标", body="指标定义。"),
                    draft_json(title="当前分析", body="共享指标影响当前分析。"),
                    '{"status":"linked","links":[]}',
                ]
            )
        ).run()

    assert not (project.concepts / "a.md").exists()
    state = Manifest.load(project.data / ".state/manifest.json").sources["a.md"]
    assert state.relation_status == "failed"
    assert state.status == "incomplete"


def test_changed_enrich_prompt_reuses_first_three_stages(project: Project):
    project.source("a.md", "# A\n正文锚点。")
    project.compiler(
        FakeLLM([discovery_json(), draft_json(), no_links_json()])
    ).run()
    (project.prompts / "enrich.md").write_text(
        "CHANGED {current_concept} {candidate_index}", encoding="utf-8"
    )
    fake = FakeLLM([no_links_json()])

    project.compiler(fake).run()

    assert fake.prompts == []


def test_missing_referenced_image_is_removed_before_llm(project: Project):
    project.source("a.md", "# A\n正文锚点。\n![](images/missing.jpg)")
    fake = FakeLLM(
        [discovery_json(), draft_json(), no_links_json()]
    )

    project.compiler(fake).run()

    assert fake.prompts
    assert all("images/missing.jpg" not in prompt for prompt in fake.prompts)
    concept = next(project.concepts.glob("*.md")).read_text(encoding="utf-8")
    assert "images/missing.jpg" not in concept


def test_failed_third_concept_reuses_first_two_drafts_on_retry(project: Project):
    project.source(
        "a.md",
        "\n\n".join(f"第 {index} 段。" for index in range(1, 9)),
    )
    first = FakeLLM(
        [
            discovery_json_for(("a", "b", "c")),
            draft_json(title="概念 a", body="第 1 段。"),
            draft_json(title="概念 b", body="第 1 段。"),
            RuntimeError("third concept failed"),
        ]
    )

    with pytest.raises(CompilationBatchError):
        project.compiler(first).run()

    retry = FakeLLM(
        [
            draft_json(title="概念 c", body="第 1 段。"),
            no_links_json(),
            no_links_json(),
            no_links_json(),
        ]
    )
    summary = project.compiler(retry).run()

    assert summary.compiled == ("a.md",)
    assert len(retry.prompts) == 1
    assert retry.prompts[0].startswith("STAGE=compile")


def tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
