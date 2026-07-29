from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

from kmpro_wiki.data_processing import (
    LocalAssetWriter,
    OfficialMinerUPageParser,
    OpenAICompatiblePageParser,
    PageParseResult,
    PageRoleResult,
    S3CompatibleAssetWriter,
    parse_pdf_with_vlm,
    process_mineru_output,
)


class FakeRenderer:
    def page_count(self, pdf: Path) -> int:
        del pdf
        return 305

    def render_page(
        self,
        pdf: Path,
        page_idx: int,
        destination: Path,
        *,
        dpi: int,
    ) -> None:
        del pdf, dpi
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 200), (255, 255, 255)).save(
            destination,
            "JPEG",
        )


class FakeParser:
    model = "mineru-fixture"
    max_tokens = 512

    def __init__(self) -> None:
        self.calls = 0

    def parse_image(
        self,
        image_path: Path,
        *,
        max_tokens: int | None = None,
    ) -> PageParseResult:
        self.calls += 1
        self.asserted_image = image_path
        self.asserted_max_tokens = max_tokens
        return PageParseResult(
            content="| 时间 | 主要内容 |\n|---|---|\n| 5月24日 | 保供稳价 |",
            model=self.model,
            finish_reason="stop",
            elapsed_ms=25,
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
        )


class PDFWorkerTests(unittest.TestCase):
    def test_worker_repairs_corrupt_page_state_and_updates_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "report.pdf"
            pdf.write_bytes(b"%PDF fixture")
            output = root / "mineru"
            results = output / "page-results"
            results.mkdir(parents=True)
            (results / "page-0001.json").write_text(
                "{broken",
                encoding="utf-8",
            )
            parser = FakeParser()

            parse_pdf_with_vlm(
                pdf,
                output,
                parser=parser,  # type: ignore[arg-type]
                renderer=FakeRenderer(),
                page_start=0,
                page_end=1,
            )

            state = json.loads(
                (output / "job.json").read_text(encoding="utf-8")
            )
            record = json.loads(
                (results / "page-0001.json").read_text(encoding="utf-8")
            )
        self.assertEqual(parser.calls, 1)
        self.assertEqual(record["page_idx"], 0)
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["completed_pages"], 1)
        self.assertEqual(len(state["source_sha256"]), 64)

    def test_worker_rejects_mixed_models_in_one_resume_job(self) -> None:
        class OtherParser(FakeParser):
            model = "different-model"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "report.pdf"
            pdf.write_bytes(b"%PDF fixture")
            output = root / "mineru"
            parse_pdf_with_vlm(
                pdf,
                output,
                parser=FakeParser(),  # type: ignore[arg-type]
                renderer=FakeRenderer(),
                page_start=0,
                page_end=1,
            )

            with self.assertRaisesRegex(ValueError, "model changed"):
                parse_pdf_with_vlm(
                    pdf,
                    output,
                    parser=OtherParser(),  # type: ignore[arg-type]
                    renderer=FakeRenderer(),
                    page_start=1,
                    page_end=2,
                )

    def test_official_mineru_uses_two_step_client(self) -> None:
        extracted = [
            {
                "type": "title",
                "bbox": [0.1, 0.1, 0.9, 0.2],
                "angle": 0,
                "content": "专题篇",
            },
            {
                "type": "text",
                "bbox": [0.1, 0.2, 0.9, 0.5],
                "angle": 0,
                "content": "产业协同互动活力不断激活。",
                "merge_prev": False,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            Image.new("RGB", (100, 200), (255, 255, 255)).save(
                image,
                "JPEG",
            )
            with patch(
                "kmpro_wiki.data_processing."
                "mineru_official.MinerUClient"
            ) as client_class:
                client_class.return_value.two_step_extract.return_value = (
                    extracted
                )
                parser = OfficialMinerUPageParser(
                    api_base=(
                        "http://model.local/compatible-mode/v1"
                    ),
                    api_key="test-key",
                    model="mineru-test",
                )
                result = parser.parse_image(image)
        kwargs = client_class.call_args.kwargs
        self.assertEqual(
            kwargs["server_url"],
            "http://model.local/compatible-mode/",
        )
        self.assertEqual(
            kwargs["server_headers"]["Authorization"],
            "Bearer test-key",
        )
        client_class.return_value.two_step_extract.assert_called_once()
        self.assertEqual(result.finish_reason, "two_step_complete")
        self.assertIn("产业协同互动活力不断激活", result.content)
        self.assertEqual(len(result.blocks), 2)

    def test_official_mineru_preserves_empty_layout_as_page_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            Image.new("RGB", (100, 120), "white").save(image)
            with patch(
                "kmpro_wiki.data_processing."
                "mineru_official.MinerUClient"
            ) as client_class:
                client_class.return_value.two_step_extract.return_value = []
                parser = OfficialMinerUPageParser(
                    api_base="http://model.local/v1",
                    api_key="test-key",
                    model="mineru-test",
                )
                result = parser.parse_image(image)

        self.assertEqual(
            result.finish_reason,
            "two_step_empty_page_preserved",
        )
        self.assertEqual(result.blocks[0]["type"], "image")
        self.assertEqual(result.blocks[0]["bbox"], [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(result.request_count, 1)

    def test_worker_persists_structured_page_role_decision(self) -> None:
        class EmptyParser(FakeParser):
            def parse_image(
                self,
                image_path: Path,
                *,
                max_tokens: int | None = None,
            ) -> PageParseResult:
                del image_path, max_tokens
                return PageParseResult(
                    content="",
                    model=self.model,
                    finish_reason="two_step_empty_page_preserved",
                    elapsed_ms=10,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    blocks=(
                        {
                            "type": "image",
                            "bbox": [0.0, 0.0, 1.0, 1.0],
                            "content": None,
                        },
                    ),
                )

        class RoleClassifier:
            def classify(self, image_path: Path) -> PageRoleResult:
                self.image_path = image_path
                return PageRoleResult(
                    role="decorative",
                    confidence=0.96,
                    reason="只有装饰底纹。",
                    model="role-model",
                    elapsed_ms=12,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "report.pdf"
            pdf.write_bytes(b"%PDF fixture")
            output = root / "mineru"
            classifier = RoleClassifier()
            parse_pdf_with_vlm(
                pdf,
                output,
                parser=EmptyParser(),  # type: ignore[arg-type]
                renderer=FakeRenderer(),
                page_start=0,
                page_end=1,
                page_role_classifier=classifier,
            )
            record = json.loads(
                (output / "page-results" / "page-0001.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(record["page_role"], "decorative")
        self.assertEqual(record["page_role_confidence"], 0.96)

    def test_vlm_provider_parses_openai_compatible_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], "mineru-test")
            self.assertEqual(payload["messages"][0]["content"][1]["type"], "image_url")
            return httpx.Response(
                200,
                json={
                    "model": "mineru-test",
                    "choices": [
                        {
                            "message": {"content": "```markdown\n正文\n```"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 2,
                        "total_tokens": 13,
                    },
                },
            )

        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "page.jpg"
            image.write_bytes(b"jpeg")
            parser = OpenAICompatiblePageParser(
                api_base="http://model.local/v1",
                api_key="test-key",
                model="mineru-test",
                transport=httpx.MockTransport(handler),
            )
            result = parser.parse_image(image)
        self.assertEqual(result.content, "正文")
        self.assertEqual(result.total_tokens, 13)

    def test_resumable_worker_and_pipeline_preserve_total_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "report.pdf"
            pdf.write_bytes(b"%PDF fixture")
            parser = FakeParser()
            output = root / "mineru"
            content_path = parse_pdf_with_vlm(
                pdf,
                output,
                parser=parser,  # type: ignore[arg-type]
                renderer=FakeRenderer(),
                page_start=152,
                page_end=153,
            )
            self.assertEqual(parser.calls, 1)
            content = json.loads(content_path.read_text(encoding="utf-8"))
            self.assertEqual(content[0]["type"], "table")
            self.assertEqual(content[0]["page_idx"], 152)

            parse_pdf_with_vlm(
                pdf,
                output,
                parser=parser,  # type: ignore[arg-type]
                renderer=FakeRenderer(),
                page_start=152,
                page_end=153,
            )
            self.assertEqual(parser.calls, 1, "completed page must be resumed")

            destination = root / "processed"
            result = process_mineru_output(
                pdf,
                output,
                destination,
                asset_writer=LocalAssetWriter(destination / "images"),
                target_chars=500,
                hard_max_chars=1000,
            )
            manifest = json.loads(
                Path(result.manifest_path).read_text(encoding="utf-8")
            )
            document = json.loads(
                Path(result.document_ir_path).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["pages"], 305)
            self.assertEqual(document["page_count"], 305)
            self.assertEqual(document["parser"], "mineru-openai-compatible")
            self.assertEqual(manifest["assets"], 1)

    def test_structured_blocks_crop_only_detected_images(self) -> None:
        class StructuredParser(FakeParser):
            def parse_image(
                self,
                image_path: Path,
                *,
                max_tokens: int | None = None,
            ) -> PageParseResult:
                self.calls += 1
                return PageParseResult(
                    content="## 专题篇\n\n下方正文",
                    model=self.model,
                    finish_reason="two_step_complete",
                    elapsed_ms=40,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    blocks=(
                        {
                            "type": "title",
                            "bbox": [0.1, 0.05, 0.9, 0.15],
                            "angle": 0,
                            "content": "专题篇",
                            "merge_prev": False,
                        },
                        {
                            "type": "image",
                            "bbox": [0.2, 0.25, 0.8, 0.65],
                            "angle": 0,
                            "content": None,
                            "merge_prev": False,
                        },
                        {
                            "type": "text",
                            "bbox": [0.1, 0.7, 0.9, 0.9],
                            "angle": 0,
                            "content": "下方正文",
                            "merge_prev": False,
                        },
                    ),
                    request_count=3,
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "report.pdf"
            pdf.write_bytes(b"%PDF fixture")
            output = root / "mineru"
            content_path = parse_pdf_with_vlm(
                pdf,
                output,
                parser=StructuredParser(),  # type: ignore[arg-type]
                renderer=FakeRenderer(),
                page_start=0,
                page_end=1,
            )
            content = json.loads(content_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["type"] for item in content],
                ["title", "image", "text"],
            )
            cropped = output / content[1]["img_path"]
            self.assertTrue(cropped.is_file())
            with Image.open(cropped) as image:
                self.assertEqual(image.size, (60, 80))

    def test_s3_compatible_writer_signs_and_uploads(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, headers={"etag": '"fixture"'})

        writer = S3CompatibleAssetWriter(
            endpoint="http://minio.local:9000",
            access_key="access",
            secret_key="secret",
            bucket="assets",
            prefix="kmpro",
            transport=httpx.MockTransport(handler),
        )
        uri = writer.write(
            "article/page.jpg",
            b"image",
            content_type="image/jpeg",
        )
        self.assertEqual(uri, "s3://assets/kmpro/article/page.jpg")
        self.assertEqual(requests[0].url.path, "/assets/kmpro/article/page.jpg")
        self.assertTrue(
            requests[0].headers["authorization"].startswith("AWS4-HMAC-SHA256 ")
        )
        self.assertEqual(requests[0].headers["content-type"], "image/jpeg")

    def test_s3_writer_corrects_minio_clock_skew_without_host_change(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.path == "/minio/health/live":
                return httpx.Response(
                    200,
                    headers={"Date": "Wed, 29 Jul 2026 01:34:26 GMT"},
                )
            if len([item for item in requests if item.method == "HEAD"]) == 1:
                return httpx.Response(403)
            return httpx.Response(200)

        writer = S3CompatibleAssetWriter(
            endpoint="http://minio.local:9000",
            access_key="access",
            secret_key="secret",
            bucket="assets",
            transport=httpx.MockTransport(handler),
        )

        self.assertTrue(writer.bucket_exists())
        self.assertEqual(
            [item.method for item in requests],
            ["HEAD", "GET", "HEAD"],
        )


if __name__ == "__main__":
    unittest.main()
