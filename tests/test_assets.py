import hashlib

import pytest

from kmpro_wiki.agentwiki.assets import (
    AssetError,
    SourceAsset,
    apply_asset_placements,
    inventory_assets,
    strip_missing_image_references,
    validate_asset_preservation,
)
from kmpro_wiki.agentwiki.contracts import AssetPlacement, ConceptRef, DraftConcept


def draft(concept_id: str = "a", body: str = "第一段。\n\n正文锚点。\n\n第三段。"):
    return DraftConcept(
        ref=ConceptRef(
            concept_id=concept_id,
            type="分析框架",
            title=f"概念 {concept_id}",
            description="摘要。",
            source="报告.md",
            evidence=("正文锚点。",),
            asset_hints=(),
        ),
        title=f"概念 {concept_id}",
        description="摘要。",
        body=body,
    )


def asset(asset_id: str, raw: str, *, kind: str = "image") -> SourceAsset:
    return SourceAsset(
        asset_id=asset_id,
        kind=kind,
        raw=raw,
        target="images/x.jpg" if kind == "image" else None,
        before="前文",
        after="后文",
        ordinal=1,
        sha256=hashlib.sha256(b"image").hexdigest() if kind == "image" else None,
    )


def test_inventory_assigns_stable_ids_in_document_order(tmp_path):
    (tmp_path / "x.jpg").write_bytes(b"image")
    source = (
        "前文\n"
        "![图](images/x.jpg)\n"
        "中间\n"
        "<table><tr><td>1</td></tr></table>\n"
        "后文\n\n"
        "| A |\n|---|\n| 1 |\n"
    )

    assets = inventory_assets(source, tmp_path)

    assert [(item.asset_id, item.kind) for item in assets] == [
        ("image-001", "image"),
        ("html-table-001", "html_table"),
        ("markdown-table-001", "markdown_table"),
    ]
    assert assets[0].sha256 == hashlib.sha256(b"image").hexdigest()
    assert assets[0].before == "前文"
    assert assets[0].after == "中间"


def test_inventory_accepts_minio_http_image_without_local_copy(tmp_path):
    target = "https://minio.example/assets/book/page-1.jpg"

    assets = inventory_assets(f"图前。\n\n![区域图]({target})\n\n图后。", tmp_path)

    assert len(assets) == 1
    assert assets[0].target == target
    assert assets[0].sha256 is None


def test_inventory_rejects_missing_local_image(tmp_path):
    with pytest.raises(AssetError, match="does not exist"):
        inventory_assets("![](images/missing.jpg)", tmp_path)


def test_strip_missing_image_references_keeps_only_existing_assets(tmp_path):
    (tmp_path / "kept.jpg").write_bytes(b"image")
    source = "前文 ![](images/kept.jpg) 中间 ![](images/missing.jpg) 后文"

    normalized = strip_missing_image_references(source, tmp_path)

    assert "images/kept.jpg" in normalized
    assert "images/missing.jpg" not in normalized
    assert len(inventory_assets(normalized, tmp_path)) == 1


def test_inventory_rejects_unsafe_local_image(tmp_path):
    with pytest.raises(AssetError, match="unsafe"):
        inventory_assets("![](images/../secret.jpg)", tmp_path)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (
            "after",
            "第一段。\n\n正文锚点。\n\n![图](images/x.jpg)\n\n第三段。",
        ),
        (
            "before",
            "第一段。\n\n![图](images/x.jpg)\n\n正文锚点。\n\n第三段。",
        ),
    ],
)
def test_placement_inserts_raw_asset_without_changing_draft(position, expected):
    result = apply_asset_placements(
        [draft()],
        [asset("image-001", "![图](images/x.jpg)")],
        [AssetPlacement("image-001", "a", "正文锚点。", position, "相关图")],
    )

    assert result[0].body == expected
    assert result[0].filename == "a.md"
    assert result[0].frontmatter["source"] == "报告.md"


def test_placement_requires_unique_anchor():
    with pytest.raises(AssetError, match="exactly once"):
        apply_asset_placements(
            [draft(body="相同锚点。\n\n相同锚点。")],
            [asset("image-001", "![图](images/x.jpg)")],
            [AssetPlacement("image-001", "a", "相同锚点。", "after", "相关图")],
        )


def test_placement_rejects_missing_or_duplicate_decisions():
    source_asset = asset("image-001", "![图](images/x.jpg)")
    placement = AssetPlacement("image-001", "a", "正文锚点。", "after", "相关图")

    with pytest.raises(AssetError, match="exactly once"):
        apply_asset_placements([draft()], [source_asset], [])
    with pytest.raises(AssetError, match="exactly once"):
        apply_asset_placements(
            [draft()], [source_asset], [placement, placement]
        )


def test_validate_preservation_requires_exact_counts_and_image_hash(tmp_path):
    image_path = tmp_path / "x.jpg"
    image_path.write_bytes(b"image")
    source_asset = asset("image-001", "![图](images/x.jpg)")
    concepts = apply_asset_placements(
        [draft()],
        [source_asset],
        [AssetPlacement("image-001", "a", "正文锚点。", "after", "相关图")],
    )

    validate_asset_preservation([source_asset], concepts, tmp_path)

    image_path.write_bytes(b"changed")
    with pytest.raises(AssetError, match="checksum"):
        validate_asset_preservation([source_asset], concepts, tmp_path)


def test_validate_preservation_rejects_duplicate_raw_asset(tmp_path):
    (tmp_path / "x.jpg").write_bytes(b"image")
    source_asset = asset("image-001", "![图](images/x.jpg)")
    concepts = apply_asset_placements(
        [draft(body="正文锚点。\n\n![图](images/x.jpg)")],
        [source_asset],
        [AssetPlacement("image-001", "a", "正文锚点。", "after", "相关图")],
    )

    with pytest.raises(AssetError, match="count"):
        validate_asset_preservation([source_asset], concepts, tmp_path)


def test_validate_preservation_counts_only_current_source_delta(tmp_path):
    (tmp_path / "x.jpg").write_bytes(b"image")
    raw = "![图](images/x.jpg)"
    source_asset = asset("image-001", raw)
    baseline = [draft(body=f"既有来源图。\n\n{raw}\n\n正文锚点。")]
    concepts = apply_asset_placements(
        baseline,
        [source_asset],
        [AssetPlacement("image-001", "a", "正文锚点。", "after", "相关图")],
    )

    validate_asset_preservation(
        [source_asset],
        concepts,
        tmp_path,
        baseline=baseline,
    )
