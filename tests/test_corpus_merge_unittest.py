from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_pdf_corpus import merge


class CorpusMergeTests(unittest.TestCase):
    def test_merge_requires_disjoint_complete_shards_and_activates_articles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = root / "a"
            b = root / "b"
            for data, name, digest, pages in (
                (a, "a.md", "a" * 64, 10),
                (b, "b.md", "b" * 64, 20),
            ):
                sources = data / "normalized-sources"
                sources.mkdir(parents=True)
                (sources / name).write_text(name, encoding="utf-8")
                (sources / f"{Path(name).stem}.structure.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                (data / "corpus-run.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "activation_status": "complete",
                            "documents": [
                                {
                                    "status": "complete",
                                    "source_sha256": digest,
                                    "activated_for_agentwiki": True,
                                }
                            ],
                            "totals": {
                                "documents": 1,
                                "complete_documents": 1,
                                "expected_pages": pages,
                                "completed_pages": pages,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            result = merge(a, b)

            self.assertEqual(result["activated_articles"], 2)
            self.assertEqual(result["totals"]["completed_pages"], 30)
            self.assertTrue((a / "normalized-sources" / "b.md").is_file())
            self.assertTrue(
                (a / "normalized-sources" / "b.structure.json").is_file()
            )
            self.assertTrue((a / "corpus-run-combined.json").is_file())


if __name__ == "__main__":
    unittest.main()
