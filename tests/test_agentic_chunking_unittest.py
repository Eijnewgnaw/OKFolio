from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from okfolio.agentwiki.agentic import (
    AgentRunError,
    AgentRefRecord,
    _attach_agent_ref_provenance,
    _asset_candidate_group_ids,
    _load_source_structure,
    discover_agent_concepts,
)
from okfolio.agentwiki.assets import SourceAsset, inventory_assets
from okfolio.agentwiki.contracts import ConceptRef, DraftConcept


class ChunkAwareLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str, **_kwargs) -> str:
        self.prompts.append(prompt)
        evidence_ids = re.findall(r'"evidence_id":\s*"([^"]+)"', prompt)
        return json.dumps(
            {
                "concepts": [
                    {
                        "id": f"概念-{len(self.prompts)}",
                        "type": "分析框架",
                        "title": f"区域判断 {len(self.prompts)}",
                        "description": "形成一项可独立引用的区域经济判断。",
                        "evidence": [evidence_ids[-1]],
                        "asset_hints": [],
                    }
                ]
            },
            ensure_ascii=False,
        )


class AgenticChunkingTests(unittest.TestCase):
    def test_structure_sidecar_attaches_ref_provenance(self) -> None:
        structure = {
            "schema_version": "kmpro.document-structure.v1",
            "status": "complete",
            "pages": [],
            "blocks": [
                {
                    "block_id": "blk-body",
                    "block_type": "text",
                    "content": "区域协同水平持续提高。",
                    "page_idx": 8,
                    "heading_path": ["综合篇", "一、发展基础"],
                    "evidence_eligible": True,
                }
            ],
        }
        ref = AgentRefRecord(
            "ref-1",
            "article-1",
            "发展基础",
            "分析框架",
            "区域协同发展基础",
            "区域协同水平持续提高。",
            ("## 一、发展基础\n\n区域协同水平持续提高。",),
            (),
            "报告.md",
        )

        enriched = _attach_agent_ref_provenance(ref, structure)

        self.assertEqual(
            enriched.section_path,
            ("综合篇", "一、发展基础"),
        )
        self.assertEqual(enriched.page_start, 9)
        self.assertEqual(enriched.page_end, 9)
        self.assertEqual(enriched.evidence_block_ids, ("blk-body",))

    def test_unresolved_structure_sidecar_blocks_agent(self) -> None:
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "报告.md"
            source.write_text("# 报告", encoding="utf-8")
            source.with_suffix(".structure.json").write_text(
                json.dumps(
                    {
                        "schema_version": "kmpro.document-structure.v1",
                        "status": "needs_review",
                        "pages": [
                            {"page_number": 7, "role": "content_retry"}
                        ],
                        "blocks": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AgentRunError, r"pages=\[7\]"):
                _load_source_structure(source)

    def test_asset_batch_prefers_ref_hint_over_full_concept_set(self) -> None:
        remote = SourceAsset(
            asset_id="image-001",
            kind="image",
            raw="![图](http://minio/assets/chart.jpg)",
            target="http://minio/assets/chart.jpg",
            before="区域生产总值变化",
            after="增速分析",
            ordinal=1,
            sha256=None,
        )
        refs = (
            AgentRefRecord(
                "r1",
                "a1",
                "one",
                "分析框架",
                "其他判断",
                "其他摘要。",
                ("其他证据。",),
                (),
                "book.md",
            ),
            AgentRefRecord(
                "r2",
                "a1",
                "two",
                "分析框架",
                "区域生产总值变化",
                "地区生产总值增速判断。",
                ("增速证据。",),
                ("image-001",),
                "book.md",
            ),
        )

        def draft(group: str, title: str) -> DraftConcept:
            return DraftConcept(
                ConceptRef(
                    group,
                    "分析框架",
                    title,
                    f"{title}摘要。",
                    "book.md",
                    ("证据。",),
                    (),
                ),
                title,
                f"{title}摘要。",
                f"{title}正文。",
            )

        selected = _asset_candidate_group_ids(
            (remote,),
            refs,
            {"r1": "g1", "r2": "g2"},
            {"g1": draft("g1", "其他判断"), "g2": draft("g2", "生产总值")},
            ("g1", "g2"),
            limit=16,
        )

        self.assertEqual(selected, ("g2",))

    def test_minio_image_is_a_remote_asset_without_fake_checksum(self) -> None:
        target = "http://minio:9000/assets/book/page-1.jpg"
        with TemporaryDirectory() as temporary:
            assets = inventory_assets(
                f"图前。\n\n![区域图]({target})\n\n图后。",
                Path(temporary),
            )

        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].target, target)
        self.assertIsNone(assets[0].sha256)

    def test_long_article_discovery_chunks_but_keeps_one_source(self) -> None:
        content = "# 区域报告\n\n" + "\n\n".join(
            f"第{index}项区域经济变化形成了可独立引用的判断，"
            f"并包含本段专属证据标记{index}。"
            for index in range(1, 9)
        )
        llm = ChunkAwareLLM()
        decisions: list[dict] = []

        with patch(
            "okfolio.agentwiki.agentic.DISCOVERY_CHUNK_CHARS",
            150,
        ):
            refs = discover_agent_concepts(
                llm,
                Path("prompts/discover.md").read_text(encoding="utf-8"),
                title="区域报告",
                source_name="book.md",
                source_content=content,
                assets=(),
                on_decision=decisions.append,
            )

        self.assertGreater(len(llm.prompts), 1)
        self.assertEqual(len(refs), len(llm.prompts))
        self.assertTrue(all(ref.source == "book.md" for ref in refs))
        self.assertEqual(
            decisions[-1]["chunks"],
            len(llm.prompts),
        )
        self.assertEqual(
            len({ref.concept_id for ref in refs}),
            len(refs),
        )


if __name__ == "__main__":
    unittest.main()
