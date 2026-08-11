from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from okfolio.data_processing.models import Block, DocumentIR
from okfolio.data_processing.structure import (
    normalize_document_structure,
)


def _block(
    ordinal: int,
    page_idx: int,
    block_type: str,
    content: str,
    *,
    asset_path: str | None = None,
) -> Block:
    return Block(
        block_id=f"blk-{ordinal}",
        block_type=block_type,
        content=content,
        page_idx=page_idx,
        bbox=(0, 0, 100, 100),
        reading_order=ordinal,
        asset_path=asset_path,
        content_hash=f"hash-{ordinal}",
    )


def _page_record(
    root: Path,
    page_idx: int,
    *,
    finish_reason: str = "two_step_complete",
    color: str = "white",
    page_role: str | None = None,
    recovery_content: str | None = None,
) -> None:
    image = root / "pages" / f"page-{page_idx + 1:04d}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (120, 180), color).save(image, "JPEG")
    record = {
        "schema_version": "kmpro.page-result.v1",
        "page_idx": page_idx,
        "status": "complete",
        "image_path": image.relative_to(root).as_posix(),
        "finish_reason": finish_reason,
    }
    if page_role is not None:
        record.update(
            {
                "page_role": page_role,
                "page_role_confidence": 0.97,
            }
        )
    if recovery_content is not None:
        record["recovery_content"] = recovery_content
    path = root / "page-results" / f"page-{page_idx + 1:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


class StructureNormalizationTests(unittest.TestCase):
    def test_toc_is_structural_only_and_body_hierarchy_is_restored(self) -> None:
        blocks = (
            _block(1, 0, "title", "区域合作蓝皮书（2023）"),
            _block(2, 1, "image", "", asset_path="images/blank.jpg"),
            _block(3, 2, "title", "目录"),
            _block(4, 2, "title", "专题篇"),
            _block(5, 2, "title", "一、建设现代化都市圈 27"),
            _block(6, 3, "title", "（一）总体发展水平评价 27"),
            _block(7, 3, "title", "（二）重点领域主要进展 40"),
            _block(8, 4, "title", "专题篇"),
            _block(9, 5, "title", "一、建设现代化都市圈"),
            _block(10, 5, "text", "本节分析都市圈发展水平。"),
            _block(11, 6, "title", "（一）总体发展水平评价"),
            _block(12, 6, "text", "发展水平持续提升，但协同机制仍需完善。"),
            _block(13, 7, "title", "1. 纵向对比分析"),
            _block(14, 7, "text", "与上一年度相比，多项指标明显提升。"),
            _block(15, 8, "title", "（1）基础设施指标"),
            _block(16, 8, "text", "轨道交通覆盖率持续提高。"),
        )
        document = DocumentIR(
            document_id="article-test",
            source_file="report.pdf",
            source_sha256="abc",
            title="区域合作蓝皮书（2023）",
            parser="mineru",
            parser_output="content_list.json",
            page_count=9,
            blocks=blocks,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for page_idx in range(9):
                _page_record(root, page_idx)
            _page_record(
                root,
                1,
                finish_reason="two_step_empty_page_preserved",
                color="white",
            )
            result = normalize_document_structure(document, root)

        roles = {item.page_idx: item.role for item in result.pages}
        self.assertEqual(roles[0], "cover")
        self.assertEqual(roles[1], "blank")
        self.assertEqual(roles[2], "toc")
        self.assertEqual(roles[3], "toc")
        self.assertEqual(
            [item.title for item in result.outline],
            [
                "专题篇",
                "一、建设现代化都市圈",
                "（一）总体发展水平评价",
                "1. 纵向对比分析",
                "（1）基础设施指标",
            ],
        )
        self.assertEqual(
            [item.level for item in result.outline],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            result.outline[-1].path,
            (
                "专题篇",
                "一、建设现代化都市圈",
                "（一）总体发展水平评价",
                "1. 纵向对比分析",
                "（1）基础设施指标",
            ),
        )
        normalized_ids = {item.block_id for item in result.document.blocks}
        self.assertNotIn("blk-3", normalized_ids)
        self.assertNotIn("blk-4", normalized_ids)
        self.assertNotIn("blk-5", normalized_ids)
        self.assertEqual(result.status, "complete")

    def test_toc_continuation_uses_text_and_table_rows(self) -> None:
        blocks = (
            _block(1, 0, "title", "报告"),
            _block(2, 1, "title", "目录"),
            _block(3, 1, "title", "第一章 总报告"),
            _block(4, 1, "text", "3 第一节 总体情况"),
            _block(
                5,
                2,
                "table",
                (
                    "<table><tr><td>第二章</td><td>产业协同</td></tr>"
                    "<tr><td>35</td><td>第一节 总体情况</td></tr></table>"
                ),
            ),
            _block(6, 3, "text", "100 第三章 体制机制"),
            _block(7, 4, "title", "第一章 总报告"),
            _block(8, 4, "text", "这是正文内容。" * 30),
        )
        document = DocumentIR(
            document_id="article-test",
            source_file="report.pdf",
            source_sha256="abc",
            title="报告",
            parser="mineru",
            parser_output="content_list.json",
            page_count=5,
            blocks=blocks,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for page_idx in range(5):
                _page_record(root, page_idx)
            result = normalize_document_structure(document, root)

        roles = {item.page_idx: item.role for item in result.pages}
        self.assertEqual(
            [roles[index] for index in range(1, 5)],
            ["toc", "toc", "toc", "content"],
        )
        self.assertEqual(
            [item.title for item in result.toc_entries],
            [
                "第一章 总报告",
                "第一节 总体情况",
                "第二章 产业协同",
                "第一节 总体情况",
                "第三章 体制机制",
            ],
        )
        self.assertEqual(
            [item.printed_page for item in result.toc_entries],
            [None, 3, None, 35, 100],
        )

    def test_headingless_toc_is_detected_without_consuming_body(self) -> None:
        blocks = (
            _block(1, 0, "title", "报告"),
            _block(2, 1, "title", "综合篇"),
            _block(3, 1, "title", "第一章 发展概况 3"),
            _block(4, 1, "text", "一、主要进展 3"),
            _block(5, 2, "title", "第二章 政策建议 35"),
            _block(6, 2, "text", "一、完善政策体系 36"),
            _block(7, 3, "image", "", asset_path="images/divider.jpg"),
            _block(8, 4, "title", "综合篇"),
            _block(9, 4, "text", "这是正文内容。" * 30),
        )
        document = DocumentIR(
            document_id="article-test",
            source_file="report.pdf",
            source_sha256="abc",
            title="报告",
            parser="mineru",
            parser_output="content_list.json",
            page_count=5,
            blocks=blocks,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for page_idx in range(5):
                _page_record(root, page_idx)
            result = normalize_document_structure(document, root)

        roles = {item.page_idx: item.role for item in result.pages}
        self.assertEqual(roles[1], "toc")
        self.assertEqual(roles[2], "toc")
        self.assertNotEqual(roles[4], "toc")
        self.assertEqual(
            [item.title for item in result.toc_entries],
            [
                "综合篇",
                "第一章 发展概况",
                "一、主要进展",
                "第二章 政策建议",
                "一、完善政策体系",
            ],
        )

    def test_nonblank_empty_layout_requires_resolution(self) -> None:
        document = DocumentIR(
            document_id="article-test",
            source_file="report.pdf",
            source_sha256="abc",
            title="报告",
            parser="mineru",
            parser_output="content_list.json",
            page_count=1,
            blocks=(
                _block(1, 0, "image", "", asset_path="images/page.jpg"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _page_record(
                root,
                0,
                finish_reason="two_step_empty_page_preserved",
                color="black",
            )
            result = normalize_document_structure(document, root)
        # First nonblank visual page is conservatively a cover.
        self.assertEqual(result.pages[0].role, "cover")
        self.assertEqual(result.status, "complete")

    def test_external_decorative_role_is_excluded(self) -> None:
        document = DocumentIR(
            document_id="article-test",
            source_file="report.pdf",
            source_sha256="abc",
            title="报告",
            parser="mineru",
            parser_output="content_list.json",
            page_count=2,
            blocks=(
                _block(1, 0, "title", "报告"),
                _block(2, 1, "image", "", asset_path="images/page.jpg"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _page_record(root, 0)
            _page_record(
                root,
                1,
                finish_reason="two_step_empty_page_preserved",
                color="blue",
                page_role="decorative",
            )
            result = normalize_document_structure(document, root)
        self.assertEqual(result.pages[1].role, "decorative")
        self.assertNotIn(
            "blk-2",
            {item.block_id for item in result.document.blocks},
        )
        self.assertEqual(result.status, "complete")

    def test_recovered_content_replaces_visual_fallback_in_normalized_view(
        self,
    ) -> None:
        document = DocumentIR(
            document_id="article-test",
            source_file="report.pdf",
            source_sha256="abc",
            title="报告",
            parser="mineru",
            parser_output="content_list.json",
            page_count=2,
            blocks=(
                _block(1, 0, "title", "报告"),
                _block(2, 1, "image", "", asset_path="images/page.jpg"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _page_record(root, 0)
            _page_record(
                root,
                1,
                finish_reason="two_step_empty_page_preserved",
                color="black",
                page_role="content",
                recovery_content=(
                    "## 一、区域协同现状\n\n"
                    "区域协同水平持续提高，但跨区域机制仍需完善。"
                ),
            )
            result = normalize_document_structure(document, root)
        recovered = [
            item
            for item in result.document.blocks
            if item.page_idx == 1
        ]
        self.assertEqual(
            [item.block_type for item in recovered],
            ["title", "text"],
        )
        self.assertIn("区域协同水平持续提高", recovered[1].content)
        self.assertEqual(result.pages[1].role, "content")
        self.assertEqual(result.status, "complete")

    def test_recovery_markdown_preserves_order_levels_and_unique_ids(self) -> None:
        document = DocumentIR(
            document_id="article-test",
            source_file="report.pdf",
            source_sha256="abc",
            title="报告",
            parser="mineru",
            parser_output="content_list.json",
            page_count=1,
            blocks=(
                _block(1, 0, "image", "", asset_path="images/page.jpg"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _page_record(
                root,
                0,
                finish_reason="two_step_empty_page_preserved",
                color="black",
                page_role="content",
                recovery_content=(
                    "## 一级标题\n"
                    "第一段\n\n"
                    "### 二级标题\n"
                    "| 指标 | 值 |\n"
                    "|---|---|\n"
                    "| A | 1 |\n\n"
                    "重复\n\n重复"
                ),
            )
            result = normalize_document_structure(document, root)

        recovered = list(result.document.blocks)
        self.assertEqual(
            [item.block_type for item in recovered],
            ["title", "text", "title", "table", "text", "text"],
        )
        self.assertEqual(
            len({item.block_id for item in recovered}),
            len(recovered),
        )
        self.assertEqual(recovered[3].content.splitlines()[0], "| 指标 | 值 |")

    def test_affiliation_is_demoted_from_title(self) -> None:
        document = DocumentIR(
            document_id="article-test",
            source_file="report.pdf",
            source_sha256="abc",
            title="报告",
            parser="mineru",
            parser_output="content_list.json",
            page_count=2,
            blocks=(
                _block(1, 0, "title", "报告"),
                _block(2, 1, "title", "现代都市圈规划理论框架与实践探索"),
                _block(
                    3,
                    1,
                    "title",
                    "（1. 清华大学建筑学院；2. 某规划设计研究院）",
                ),
                _block(4, 1, "text", "文章分析现代都市圈规划方法。"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _page_record(root, 0)
            _page_record(root, 1)
            result = normalize_document_structure(document, root)
        affiliation = next(
            item for item in result.document.blocks if item.block_id == "blk-3"
        )
        self.assertEqual(affiliation.block_type, "text")
        self.assertIsNone(affiliation.heading_level)

    def test_fragmented_numbering_marker_is_joined_to_following_title(
        self,
    ) -> None:
        document = DocumentIR(
            document_id="article-test",
            source_file="report.pdf",
            source_sha256="abc",
            title="报告",
            parser="mineru",
            parser_output="content_list.json",
            page_count=2,
            blocks=(
                _block(1, 0, "title", "区域发展报告"),
                _block(2, 1, "title", "（一）"),
                _block(3, 1, "title", "总体发展情况"),
                _block(4, 1, "text", "区域协同水平持续提高。"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _page_record(root, 0)
            _page_record(root, 1)
            result = normalize_document_structure(document, root)

        headings = [
            item.content
            for item in result.document.blocks
            if item.block_type == "title"
        ]
        self.assertEqual(headings, ["（一） 总体发展情况"])
        self.assertIn("blk-2", result.excluded_block_ids)
        self.assertNotIn(
            "blk-2",
            {item.block_id for item in result.document.blocks},
        )
        self.assertEqual(result.quality_issues, ())

    def test_orphan_marker_is_demoted_and_symbol_heading_is_excluded(
        self,
    ) -> None:
        document = DocumentIR(
            document_id="article-test",
            source_file="report.pdf",
            source_sha256="abc",
            title="报告",
            parser="mineru",
            parser_output="content_list.json",
            page_count=2,
            blocks=(
                _block(1, 0, "title", "区域发展报告"),
                _block(2, 1, "title", "1"),
                _block(3, 1, "text", "平均路径长度为 2.4。"),
                _block(4, 1, "title", "#"),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _page_record(root, 0)
            _page_record(root, 1)
            result = normalize_document_structure(document, root)

        marker = next(
            item for item in result.document.blocks if item.block_id == "blk-2"
        )
        self.assertEqual(marker.block_type, "text")
        self.assertNotIn(
            "blk-4",
            {item.block_id for item in result.document.blocks},
        )
        self.assertIn("blk-4", result.excluded_block_ids)
        self.assertEqual(result.outline, ())
        self.assertEqual(result.quality_issues, ())
        self.assertEqual(result.status, "complete")


if __name__ == "__main__":
    unittest.main()
