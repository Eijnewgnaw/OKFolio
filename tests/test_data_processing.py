from __future__ import annotations

import json
from pathlib import Path

from okfolio.data_processing import (
    LocalAssetWriter,
    MinerUContentAdapter,
    S3WriterAssetWriter,
    process_mineru_output,
    segment_document,
)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    pdf = tmp_path / "区域报告.pdf"
    pdf.write_bytes(b"%PDF-1.4\nfixture")
    output = tmp_path / "mineru"
    images = output / "images"
    images.mkdir(parents=True)
    (images / "chart.png").write_bytes(b"png-fixture")
    (output / "report_content_list.json").write_text(
        json.dumps(
            [
                {
                    "type": "title",
                    "text": "第一章 发展基础",
                    "text_level": 1,
                    "page_idx": 0,
                    "bbox": [10, 20, 900, 90],
                },
                {
                    "type": "text",
                    "text": "本章分析区域经济的发展基础。",
                    "page_idx": 0,
                    "bbox": [10, 100, 900, 180],
                },
                {
                    "type": "image",
                    "img_path": "images/chart.png",
                    "image_caption": ["图1 区域经济结构"],
                    "page_idx": 1,
                    "bbox": [50, 100, 850, 700],
                },
                {
                    "type": "title",
                    "text": "第二章 政策建议",
                    "text_level": 1,
                    "page_idx": 2,
                    "bbox": [10, 20, 900, 90],
                },
                {
                    "type": "text",
                    "text": "建议强化跨区域协同机制。",
                    "page_idx": 2,
                    "bbox": [10, 100, 900, 180],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return pdf, output


def test_mineru_ir_segmentation_and_asset_lineage(tmp_path: Path) -> None:
    pdf, output = _fixture(tmp_path)
    content_list = MinerUContentAdapter.find_content_list(output)
    document = MinerUContentAdapter().load(
        content_list,
        document_id="article-test",
        source_file=pdf.name,
        source_sha256="abc",
        title=pdf.stem,
    )
    assert document.page_count == 3
    assert document.blocks[1].heading_path == ("第一章 发展基础",)
    assert document.blocks[-1].heading_path == ("第二章 政策建议",)
    segments = segment_document(
        document,
        target_chars=500,
        hard_max_chars=500,
    )
    assert len(segments) == 2
    assert segments[0].page_start == 0
    assert segments[1].page_end == 2


def test_pipeline_writes_agentwiki_inputs(tmp_path: Path) -> None:
    pdf, output = _fixture(tmp_path)
    destination = tmp_path / "processed"
    result = process_mineru_output(
        pdf,
        output,
        destination,
        asset_writer=LocalAssetWriter(destination / "images"),
        target_chars=500,
        hard_max_chars=500,
    )
    assert result.blocks == 5
    assert result.assets == 1
    assert result.segments == 2
    article = Path(result.article_path).read_text(encoding="utf-8")
    assert "document_id:" in article
    assert "images/article-" in article
    assert Path(result.raw_article_path).is_file()
    assert Path(result.normalized_document_ir_path).is_file()
    structure = json.loads(
        Path(result.structure_path).read_text(encoding="utf-8")
    )
    assert structure["status"] == "complete"
    assert result.normalization_status == "complete"
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"


def test_s3writer_adapter_uses_bytes_without_secrets() -> None:
    calls: list[tuple[str, bytes]] = []

    class Writer:
        def write(self, key: str, data: bytes) -> None:
            calls.append((key, data))

    adapter = S3WriterAssetWriter(
        Writer(),
        bucket="knowledge-assets",
        prefix="wiki/images",
    )
    uri = adapter.write("article-1/a.png", b"image", content_type="image/png")
    assert calls == [("wiki/images/article-1/a.png", b"image")]
    assert uri == "s3://knowledge-assets/wiki/images/article-1/a.png"
