#!/usr/bin/env python3
"""Resolve only ambiguous visual-only PDF pages through a constrained VLM."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from kmpro_wiki.data_processing.page_role import (
    OpenAICompatiblePageRoleClassifier,
)
from kmpro_wiki.data_processing.vlm import (
    OpenAICompatiblePageParser,
)
from kmpro_wiki.data_processing.structure import (
    document_from_dict,
    normalize_document_structure,
)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify unresolved visual-only pages without rerunning MinerU"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "/app/runtime/data")),
    )
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    page_role_model = (
        os.environ.get("PAGE_ROLE_MODEL")
        or os.environ.get("MINERU_MODEL", "")
    ).strip()
    if not page_role_model:
        parser.error("PAGE_ROLE_MODEL or MINERU_MODEL is required")
    recovery_model = (
        os.environ.get("PAGE_RECOVERY_MODEL") or page_role_model
    ).strip()
    classifier = OpenAICompatiblePageRoleClassifier(
        api_base=os.environ.get("PAGE_ROLE_API_BASE")
        or os.environ.get("MINERU_API_BASE")
        or os.environ.get("LLM_API_BASE", ""),
        api_key=os.environ.get("PAGE_ROLE_API_KEY")
        or os.environ.get("MINERU_API_KEY")
        or os.environ.get("LLM_API_KEY", ""),
        model=page_role_model,
        timeout=float(os.environ.get("PAGE_ROLE_TIMEOUT", "120")),
    )
    recovery_parser = OpenAICompatiblePageParser(
        api_base=os.environ.get("PAGE_RECOVERY_API_BASE")
        or os.environ.get("PAGE_ROLE_API_BASE")
        or os.environ.get("MINERU_API_BASE")
        or os.environ.get("LLM_API_BASE", ""),
        api_key=os.environ.get("PAGE_RECOVERY_API_KEY")
        or os.environ.get("PAGE_ROLE_API_KEY")
        or os.environ.get("MINERU_API_KEY")
        or os.environ.get("LLM_API_KEY", ""),
        model=recovery_model,
        timeout=float(os.environ.get("PAGE_RECOVERY_TIMEOUT", "180")),
        max_tokens=int(os.environ.get("PAGE_RECOVERY_MAX_TOKENS", "4096")),
    )
    decisions: list[dict[str, Any]] = []
    for directory in sorted((data_dir / "processed").iterdir()):
        if not directory.is_dir():
            continue
        document_path = directory / "document-ir.json"
        if not document_path.is_file():
            continue
        document = document_from_dict(
            json.loads(document_path.read_text(encoding="utf-8"))
        )
        job_dir = data_dir / "parser-jobs" / directory.name
        normalization = normalize_document_structure(document, job_dir)
        for page in normalization.pages:
            if not page.needs_review:
                continue
            record_path = (
                job_dir
                / "page-results"
                / f"page-{page.page_number:04d}.json"
            )
            record = json.loads(record_path.read_text(encoding="utf-8"))
            image_path = job_dir / str(record["image_path"])
            result = classifier.classify(image_path)
            record.update(result.to_dict())
            if result.role == "content_retry":
                recovered = recovery_parser.parse_image(image_path)
                if not recovered.content.strip():
                    _write_json(record_path, record)
                    decisions.append(
                        {
                            "document_id": document.document_id,
                            "page_number": page.page_number,
                            **result.to_dict(),
                            "recovery_status": "failed_empty",
                        }
                    )
                    continue
                record.update(
                    {
                        "page_role": "content",
                        "page_role_reason": (
                            f"{result.reason} 已触发忠实页面重解析。"
                        ),
                        "recovery_content": recovered.content,
                        "recovery_model": recovered.model,
                        "recovery_elapsed_ms": recovered.elapsed_ms,
                        "recovery_status": "complete",
                    }
                )
            _write_json(record_path, record)
            decisions.append(
                {
                    "document_id": document.document_id,
                    "page_number": page.page_number,
                    **{
                        key: value
                        for key, value in record.items()
                        if key.startswith("page_role")
                        or key.startswith("recovery_")
                    },
                }
            )
    report = {
        "schema_version": "kmpro.page-role-resolution.v1",
        "decisions": decisions,
        "unresolved": [
            item
            for item in decisions
            if item.get("page_role") == "content_retry"
            or item.get("recovery_status") == "failed_empty"
        ],
    }
    _write_json(data_dir / "page-role-resolution.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not report["unresolved"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
