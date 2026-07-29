from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from kmpro_wiki.data_processing.mineru import MinerUContentAdapter
from kmpro_wiki.data_processing.pipeline import _document_title
from scripts.process_pdf_corpus import (
    _document_retry_delay,
    _corpus_state,
    _page_metrics,
    _prepare_documents,
    _summary,
)


class PDFCorpusTests(unittest.TestCase):
    def test_prepare_documents_keeps_full_manifest_and_repairs_resume_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdfs = [root / "a.pdf", root / "b.pdf"]
            for path, content in zip(pdfs, (b"a", b"b"), strict=True):
                path.write_bytes(content)
            digest_a = hashlib.sha256(b"a").hexdigest()
            digest_b = hashlib.sha256(b"b").hexdigest()
            jobs = root / "parser-jobs"
            processed = root / "processed"
            sources = root / "normalized-sources"
            page_results = jobs / digest_a[:20] / "page-results"
            page_results.mkdir(parents=True)
            (page_results / "page-0001.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "page_idx": 0,
                        "request_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            destination = processed / digest_a[:20]
            destination.mkdir(parents=True)
            (destination / "manifest.json").write_text(
                json.dumps({"status": "complete"}),
                encoding="utf-8",
            )
            previous = {
                digest_a: {
                    "source_sha256": digest_a,
                    "status": "complete",
                    "expected_pages": 1,
                    "attempts": 2,
                    "error": "stale HTTP 429",
                    "activated_for_agentwiki": True,
                },
                digest_b: {
                    "source_sha256": digest_b,
                    "status": "running",
                    "expected_pages": 2,
                    "attempts": 1,
                },
            }

            entries, documents = _prepare_documents(
                pdfs,
                jobs_dir=jobs,
                processed_dir=processed,
                sources_dir=sources,
                previous=previous,
            )

        self.assertEqual(len(entries), 2)
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0]["status"], "complete")
        self.assertNotIn("error", documents[0])
        self.assertEqual(documents[1]["status"], "pending")
        self.assertEqual(documents[1]["expected_pages"], 2)

    def test_corpus_state_distinguishes_processing_from_activation(self):
        documents = [
            {
                "status": "complete",
                "expected_pages": 1,
                "activated_for_agentwiki": True,
            },
            {
                "status": "complete",
                "expected_pages": 1,
                "activated_for_agentwiki": False,
            },
        ]

        state = _corpus_state(
            status="complete",
            run_id="run",
            started_at="start",
            documents=documents,
        )

        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["activation_status"], "needs_review")
        self.assertEqual(state["totals"]["complete_documents"], 2)
        self.assertEqual(state["totals"]["activated_documents"], 1)

    def test_rate_limit_retry_uses_longer_cooldown(self):
        self.assertEqual(
            _document_retry_delay(
                RuntimeError("HTTP 429 请求过于频繁"),
                2,
            ),
            60,
        )

    def test_document_title_uses_longest_title_on_first_candidate_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            content = Path(temporary) / "book_content_list.json"
            content.write_text(
                json.dumps(
                    [
                        {"type": "title", "text": "目录", "page_idx": 0},
                        {
                            "type": "title",
                            "text": "中国区域经济发展年度报告",
                            "page_idx": 1,
                        },
                        {"type": "title", "text": "总报告", "page_idx": 1},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            document = MinerUContentAdapter().load(
                content,
                document_id="article-1",
                source_file="opaque.pdf",
                source_sha256="abc",
                title="opaque",
            )

            self.assertEqual(
                _document_title(document, "opaque"),
                "中国区域经济发展年度报告",
            )

    def test_corpus_metrics_and_summary_count_durable_pages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "page-results"
            results.mkdir()
            (results / "page-0001.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "request_count": 7,
                        "elapsed_ms": 1500,
                        "attempts": 2,
                    }
                ),
                encoding="utf-8",
            )
            metrics = _page_metrics(root)
        self.assertEqual(
            metrics,
            {
                "completed_pages": 1,
                "model_requests": 7,
                "parser_elapsed_ms": 1500,
                "page_retries": 1,
            },
        )

        totals = _summary(
            [
                {
                    "status": "complete",
                    "expected_pages": 1,
                    "metrics": metrics,
                    "result": {"blocks": 5, "segments": 1, "assets": 2},
                }
            ]
        )
        self.assertEqual(totals["complete_documents"], 1)
        self.assertEqual(totals["completed_pages"], 1)
        self.assertEqual(totals["model_requests"], 7)


if __name__ == "__main__":
    unittest.main()
