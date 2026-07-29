import pytest

from kmpro_wiki.agentwiki.contracts import LinkSuggestion, RelationAudit
from kmpro_wiki.agentwiki.okf import ConceptDocument, validate_link_only_enrichment
from kmpro_wiki.agentwiki.relations import (
    RelationError,
    apply_relation_audit,
    build_relation_anchor_catalog,
    classify_links,
)


def concept(
    filename: str,
    body: str,
    *,
    source: str = "source.md",
) -> ConceptDocument:
    return ConceptDocument(
        filename=filename,
        frontmatter={
            "type": "分析框架",
            "title": filename[:-3],
            "description": "摘要。",
            "source": source,
        },
        body=body,
    )


def test_wraps_existing_unique_body_anchor_without_rewrite():
    original = concept("a.md", "融资需求受到信用水平约束。")

    updated = apply_relation_audit(
        original,
        RelationAudit(
            "linked", (LinkSuggestion("b", "信用水平", "构成约束"),)
        ),
        {"b": concept("b.md", "信用水平定义。")},
    )

    assert updated.body == "融资需求受到[信用水平](../concepts/b.md)约束。"
    validate_link_only_enrichment(original, updated)


def test_applies_multiple_links_without_offset_corruption():
    updated = apply_relation_audit(
        concept("a.md", "融资需求受到信用水平和融资成本约束。"),
        RelationAudit(
            "linked",
            (
                LinkSuggestion("b", "信用水平", "约束"),
                LinkSuggestion("c", "融资成本", "约束"),
            ),
        ),
        {
            "b": concept("b.md", "定义。"),
            "c": concept("c.md", "定义。"),
        },
    )

    assert "[信用水平](../concepts/b.md)" in updated.body
    assert "[融资成本](../concepts/c.md)" in updated.body


@pytest.mark.parametrize(
    "body",
    [
        "# 信用水平\n正文。",
        "`信用水平`",
        "```text\n信用水平\n```",
        "![信用水平](../images/x.jpg)",
        "[信用水平](https://example.com)",
        "<table><tr><td>信用水平</td></tr></table>",
        "| 指标 | 数值 |\n|---|---|\n| 信用水平 | 高 |",
    ],
)
def test_rejects_heading_code_image_and_existing_link_anchors(body: str):
    with pytest.raises(RelationError, match="protected Markdown"):
        apply_relation_audit(
            concept("a.md", body),
            RelationAudit(
                "linked", (LinkSuggestion("b", "信用水平", "约束"),)
            ),
            {"b": concept("b.md", "定义。")},
        )


def test_rejects_missing_or_nonunique_anchor():
    with pytest.raises(RelationError, match="exactly once"):
        apply_relation_audit(
            concept("a.md", "信用水平与信用水平。"),
            RelationAudit(
                "linked", (LinkSuggestion("b", "信用水平", "约束"),)
            ),
            {"b": concept("b.md", "定义。")},
        )


def test_wraps_the_catalog_selected_occurrence_of_repeated_anchor():
    original = concept("a.md", "融资成本下降，融资成本仍需监测。")

    updated = apply_relation_audit(
        original,
        RelationAudit(
            "linked",
            (LinkSuggestion("b", "融资成本", "指标复用", occurrence=1),),
        ),
        {"b": concept("b.md", "融资成本定义。")},
    )

    assert updated.body == (
        "融资成本下降，[融资成本](../concepts/b.md)仍需监测。"
    )


def test_rejects_self_link_and_missing_target():
    original = concept("a.md", "信用水平。")
    audit = RelationAudit(
        "linked", (LinkSuggestion("target", "信用水平", "定义"),)
    )

    with pytest.raises(RelationError, match="self-link"):
        apply_relation_audit(original, audit, {"target": original})
    with pytest.raises(RelationError, match="unknown target"):
        apply_relation_audit(original, audit, {})


def test_no_links_is_successful_and_byte_unchanged():
    original = concept("a.md", "正文。")

    assert (
        apply_relation_audit(original, RelationAudit("no_links", ()), {})
        == original
    )


def test_anchor_catalog_excludes_cross_target_generic_and_evaluative_phrases():
    current = concept(
        "current.md",
        "2026年企业融资有待提升，融资供给指数仍需监测。",
    )
    candidates = {
        "supply": ConceptDocument(
            "supply.md",
            {
                "type": "分析框架",
                "title": "企业融资供给有待提升",
                "description": "2026年企业融资供给指数表现。",
                "source": "source.md",
            },
            "定义。",
        ),
        "cost": concept("cost.md", "定义。"),
        "efficiency": concept("efficiency.md", "定义。"),
    }
    for key in ("cost", "efficiency"):
        candidates[key].frontmatter["title"] = f"企业融资{key}"
        candidates[key].frontmatter["description"] = "企业融资分析。"

    anchors = build_relation_anchor_catalog(current, candidates)
    texts = {item.text for item in anchors}

    assert "企业融资" not in texts
    assert "有待提升" not in texts
    assert "2026年" not in texts
    assert "融资供给指数" in texts


def test_classifies_same_cross_broken_and_self_links():
    concepts = {
        "a.md": concept(
            "a.md",
            "[同源](../concepts/b.md) [跨源](../concepts/c.md) "
            "[坏链](../concepts/missing.md) [自身](../concepts/a.md)",
            source="one.md",
        ),
        "b.md": concept("b.md", "正文。", source="one.md"),
        "c.md": concept("c.md", "正文。", source="two.md"),
    }

    assert classify_links(concepts) == {
        "same_source": 1,
        "cross_source": 1,
        "broken": 1,
        "self": 1,
    }
