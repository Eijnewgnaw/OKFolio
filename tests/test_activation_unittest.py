from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from okfolio.data_processing.activation import (
    activate_article,
)


class ActivationTests(unittest.TestCase):
    def test_ready_article_is_activated_as_a_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            article = root / "normalized-article.md"
            structure = root / "document-structure.json"
            sources = root / "sources"
            article.write_text("# 正文", encoding="utf-8")
            structure.write_text('{"status":"complete"}', encoding="utf-8")

            activated = activate_article(
                article,
                structure,
                sources,
                ready=True,
                article_name="report.md",
            )

            self.assertTrue(activated)
            self.assertEqual(
                (sources / "report.md").read_text(encoding="utf-8"),
                "# 正文",
            )
            self.assertTrue((sources / "report.structure.json").is_file())

    def test_failed_gate_removes_stale_article_and_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = root / "sources"
            sources.mkdir()
            (sources / "report.md").write_text("stale", encoding="utf-8")
            (sources / "report.structure.json").write_text(
                "{}",
                encoding="utf-8",
            )

            activated = activate_article(
                root / "normalized-article.md",
                root / "document-structure.json",
                sources,
                ready=False,
                article_name="report.md",
            )

            self.assertFalse(activated)
            self.assertFalse((sources / "report.md").exists())
            self.assertFalse((sources / "report.structure.json").exists())

    def test_failed_pair_update_never_leaves_stale_markdown_active(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            article = root / "normalized-article.md"
            sources = root / "sources"
            sources.mkdir()
            article.write_text("new", encoding="utf-8")
            (sources / "report.md").write_text("stale", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                activate_article(
                    article,
                    root / "missing-structure.json",
                    sources,
                    ready=True,
                    article_name="report.md",
                )

            self.assertFalse((sources / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
