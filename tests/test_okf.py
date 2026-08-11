from pathlib import Path

import pytest

from okfolio.agentwiki.okf import (
    ConceptDocument,
    OKFValidationError,
    parse_compile_response,
    parse_concept_markdown,
    rewrite_image_paths,
    restore_missing_assets,
    validate_link_only_enrichment,
    validate_concept,
    validate_preserved_assets,
)


def concept_markdown(*, source: str = "a.md", body: str = "正文") -> str:
    return f"""---
type: 政策建议
title: 示例概念
description: 示例摘要
source: {source}
---
{body}
"""


def test_parse_multiple_concepts():
    response = (
        "模型说明\n"
        "===FILE: one.md===\n"
        f"{concept_markdown()}"
        "===END===\n"
        "===FILE: two.md===\n"
        f"{concept_markdown(body='第二篇')}"
        "===END==="
    )

    concepts = parse_compile_response(response)

    assert [item.filename for item in concepts] == ["one.md", "two.md"]
    assert concepts[1].body == "第二篇"


def test_compile_response_rejects_unclosed_file_block():
    with pytest.raises(OKFValidationError, match="unclosed"):
        parse_compile_response("===FILE: one.md===\n---\ntype: 政策建议\n---")


@pytest.mark.parametrize(
    "filename",
    ["../../secret.md", "/tmp/secret.md", "folder/secret.md", "index.md", "log.md"],
)
def test_rejects_unsafe_or_reserved_filename(filename: str):
    concept = parse_concept_markdown(filename, concept_markdown())

    with pytest.raises(OKFValidationError):
        validate_concept(concept, "a.md")


def test_requires_nonempty_okf_fields():
    invalid = """---
type: ""
title: title
description: description
source: a.md
---
body
"""
    concept = parse_concept_markdown("concept.md", invalid)

    with pytest.raises(OKFValidationError, match="type"):
        validate_concept(concept, "a.md")


def test_requires_expected_source():
    concept = parse_concept_markdown("concept.md", concept_markdown(source="other.md"))

    with pytest.raises(OKFValidationError, match="source"):
        validate_concept(concept, "a.md")


def test_detects_missing_html_table():
    source = "<table><tr><td>1</td></tr></table>"

    with pytest.raises(OKFValidationError, match="HTML table"):
        validate_preserved_assets(source, [concept_markdown()], Path("images"))


def test_detects_missing_markdown_table():
    source = "| A | B |\n|---|---|\n| 1 | 2 |"

    with pytest.raises(OKFValidationError, match="Markdown table"):
        validate_preserved_assets(source, [concept_markdown()], Path("images"))


def test_requires_referenced_source_image(tmp_path: Path):
    source = "![](images/missing.jpg)"

    with pytest.raises(OKFValidationError, match="missing.jpg"):
        validate_preserved_assets(source, [concept_markdown(body=source)], tmp_path)


def test_preserved_assets_accept_exact_tables_and_existing_image(tmp_path: Path):
    image = tmp_path / "figure.jpg"
    image.write_bytes(b"image")
    html_table = "<table><tr><td>1</td></tr></table>"
    markdown_table = "| A | B |\n|---|---|\n| 1 | 2 |"
    source = f"![](images/figure.jpg)\n{html_table}\n{markdown_table}"

    validate_preserved_assets(source, [concept_markdown(body=source)], tmp_path)


def test_rewrite_image_paths_keeps_alt_text():
    content = "![图 1](images/figure.jpg) and ![](https://example.com/remote.jpg)"

    assert rewrite_image_paths(content) == (
        "![图 1](../images/figure.jpg) and ![](https://example.com/remote.jpg)"
    )


def test_restore_missing_assets_appends_exact_blocks_to_data_concept():
    table = "<table><tr><td>原始值</td></tr></table>"
    image = "![图 1](images/chart.jpg)"
    source = f"# 报告\n{table}\n{image}\n"
    concepts = [
        ConceptDocument(
            "analysis.md",
            {
                "type": "分析框架",
                "title": "分析",
                "description": "摘要",
                "source": "a.md",
            },
            "分析正文",
        ),
        ConceptDocument(
            "metric.md",
            {
                "type": "数据口径",
                "title": "指标",
                "description": "摘要",
                "source": "a.md",
            },
            "指标正文",
        ),
    ]

    restored = restore_missing_assets(source, concepts)

    assert restored[0].body == "分析正文"
    assert "## 原文图表（自动保真）" in restored[1].body
    assert table in restored[1].body
    assert image in restored[1].body


def test_link_only_enrichment_accepts_links_without_text_rewrite():
    original = ConceptDocument(
        "a.md",
        {"type": "分析框架", "title": "分析", "description": "摘要", "source": "s.md"},
        "参考 [[融资需求指数]]。",
    )
    candidate = ConceptDocument(
        "a.md",
        dict(original.frontmatter),
        "参考 [融资需求指数](../concepts/demand.md)。",
    )

    validate_link_only_enrichment(original, candidate)


def test_link_only_enrichment_rejects_rewritten_text():
    original = ConceptDocument(
        "a.md",
        {"type": "分析框架", "title": "分析", "description": "摘要", "source": "s.md"},
        "原始判断。",
    )
    candidate = ConceptDocument(
        "a.md",
        dict(original.frontmatter),
        "改写后的判断。",
    )

    with pytest.raises(OKFValidationError, match="rewrote"):
        validate_link_only_enrichment(original, candidate)
