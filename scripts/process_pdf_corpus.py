#!/usr/bin/env python3
"""Resumable corpus-level PDF processing worker."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from okfolio.data_processing.activation import (
    activate_article,
    deactivate_article,
)
from okfolio.data_processing.mineru_official import (
    OfficialMinerUPageParser,
)
from okfolio.data_processing.pdf_worker import (
    parse_pdf_with_vlm,
)
from okfolio.data_processing.page_role import (
    OpenAICompatiblePageRoleClassifier,
)
from okfolio.data_processing.pipeline import (
    process_mineru_output,
)
from okfolio.data_processing.s3 import (
    S3CompatibleAssetWriter,
)
from okfolio.data_processing.storage import LocalAssetWriter
from okfolio.data_processing.vlm import OpenAICompatiblePageParser
from okfolio.agentwiki.config import (
    openai_model,
    provider_api_key,
    provider_base_url,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _page_metrics(job_dir: Path) -> dict[str, int]:
    pages = requests = parser_elapsed_ms = retries = 0
    for path in sorted((job_dir / "page-results").glob("page-*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if item.get("status") != "complete":
            continue
        pages += 1
        requests += int(item.get("request_count") or 0)
        parser_elapsed_ms += int(item.get("elapsed_ms") or 0)
        retries += max(0, int(item.get("attempts") or 1) - 1)
    return {
        "completed_pages": pages,
        "model_requests": requests,
        "parser_elapsed_ms": parser_elapsed_ms,
        "page_retries": retries,
    }


def _document_retry_delay(error: Exception, attempt: int) -> int:
    message = str(error).lower()
    if "429" in message or "请求过于频繁" in message:
        return min(120, 30 * attempt)
    if "timeout" in message or "timed out" in message:
        return min(60, 10 * attempt)
    return min(30, 2**attempt)


def _summary(documents: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "documents": len(documents),
        "complete_documents": 0,
        "failed_documents": 0,
        "pending_documents": 0,
        "running_documents": 0,
        "activated_documents": 0,
        "needs_review_documents": 0,
        "expected_pages": 0,
        "completed_pages": 0,
        "model_requests": 0,
        "parser_elapsed_ms": 0,
        "page_retries": 0,
        "blocks": 0,
        "segments": 0,
        "assets": 0,
    }
    for item in documents:
        status = str(item.get("status") or "")
        totals["complete_documents"] += int(status == "complete")
        totals["failed_documents"] += int(status == "failed")
        totals["pending_documents"] += int(status == "pending")
        totals["running_documents"] += int(status == "running")
        activated = item.get("activated_for_agentwiki") is True
        totals["activated_documents"] += int(activated)
        totals["needs_review_documents"] += int(
            status == "complete" and not activated
        )
        totals["expected_pages"] += int(item.get("expected_pages") or 0)
        metrics = item.get("metrics") or {}
        result = item.get("result") or {}
        for key in (
            "completed_pages",
            "model_requests",
            "parser_elapsed_ms",
            "page_retries",
        ):
            totals[key] += int(metrics.get(key) or 0)
        for key in ("blocks", "segments", "assets"):
            totals[key] += int(result.get(key) or 0)
    return totals


def _writer(args: argparse.Namespace):
    if args.asset_mode == "minio":
        writer = S3CompatibleAssetWriter(
            endpoint=os.environ["S3_ENDPOINT"],
            access_key=os.environ["S3_ACCESS_KEY"],
            secret_key=os.environ["S3_SECRET_KEY"],
            bucket=os.environ["S3_BUCKET"],
            prefix=os.environ.get("S3_PREFIX", "okfolio"),
            region=os.environ.get("S3_REGION", "us-east-1"),
            verify_tls=os.environ.get("S3_VERIFY_TLS", "true").lower()
            not in {"0", "false", "no"},
            public_base_url=os.environ.get("S3_PUBLIC_BASE_URL", ""),
        )
        if not writer.bucket_exists():
            if not args.ensure_bucket:
                raise RuntimeError(
                    "MinIO bucket does not exist; pass --ensure-bucket to create it"
                )
            writer.create_bucket()
        return writer
    return LocalAssetWriter(args.data_dir / "normalized-sources" / "images")


def _manifest_document(
    pdf: Path,
    *,
    source_sha256: str,
    expected_pages: int,
) -> dict[str, Any]:
    return {
        "source_file": pdf.name,
        "source_sha256": source_sha256,
        "bytes": pdf.stat().st_size,
        "expected_pages": expected_pages,
        "status": "pending",
        "attempts": 0,
    }


def _load_expected_pages(job_dir: Path) -> int:
    job = job_dir / "job.json"
    if not job.is_file():
        return 0
    try:
        return int(json.loads(job.read_text(encoding="utf-8")).get("page_count") or 0)
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _complete_output_exists(
    item: dict[str, Any],
    *,
    job_dir: Path,
    destination: Path,
) -> bool:
    if item.get("status") != "complete":
        return False
    expected_pages = int(item.get("expected_pages") or 0)
    metrics = _page_metrics(job_dir)
    if expected_pages < 1 or metrics["completed_pages"] != expected_pages:
        return False
    manifest = _load_json(destination / "manifest.json")
    return manifest.get("status") == "complete"


def _prepare_documents(
    pdfs: list[Path],
    *,
    jobs_dir: Path,
    processed_dir: Path,
    sources_dir: Path,
    previous: dict[str, dict[str, Any]],
) -> tuple[
    list[tuple[Path, str, Path, Path]],
    list[dict[str, Any]],
]:
    entries: list[tuple[Path, str, Path, Path]] = []
    documents: list[dict[str, Any]] = []
    for pdf in pdfs:
        source_sha256 = _sha256(pdf)
        job_dir = jobs_dir / source_sha256[:20]
        destination = processed_dir / source_sha256[:20]
        expected_pages = _load_expected_pages(job_dir)
        item = copy.deepcopy(
            previous.get(source_sha256)
            or _manifest_document(
                pdf,
                source_sha256=source_sha256,
                expected_pages=expected_pages,
            )
        )
        item.update(
            {
                "source_file": pdf.name,
                "source_sha256": source_sha256,
                "bytes": pdf.stat().st_size,
                "expected_pages": (
                    expected_pages
                    or int(item.get("expected_pages") or 0)
                ),
                "metrics": _page_metrics(job_dir),
            }
        )
        if item.get("status") == "running":
            item["status"] = "pending"
        if item.get("status") == "complete" and not _complete_output_exists(
            item,
            job_dir=job_dir,
            destination=destination,
        ):
            stale_result = item.get("result") or {}
            stale_article = Path(
                str(stale_result.get("article_path") or f"{pdf.stem}.md")
            ).name
            deactivate_article(stale_article, sources_dir)
            item["status"] = "pending"
            item["activated_for_agentwiki"] = False
            item.pop("finished_at", None)
            item.pop("result", None)
        if item.get("status") == "complete":
            item.pop("error", None)
            item.pop("failed_at", None)
        entries.append((pdf, source_sha256, job_dir, destination))
        documents.append(item)
    return entries, documents


def _corpus_state(
    *,
    status: str,
    run_id: str,
    started_at: str,
    documents: list[dict[str, Any]],
    finished_at: str | None = None,
) -> dict[str, Any]:
    totals = _summary(documents)
    if totals["documents"] and totals["activated_documents"] == totals["documents"]:
        activation_status = "complete"
    elif totals["complete_documents"] == totals["documents"]:
        activation_status = "needs_review"
    else:
        activation_status = "pending"
    payload: dict[str, Any] = {
        "schema_version": "okfolio.pdf-corpus-run.v1",
        "run_id": run_id,
        "status": status,
        "activation_status": activation_status,
        "started_at": started_at,
        "updated_at": _now(),
        "documents": documents,
        "totals": totals,
    }
    if finished_at is not None:
        payload["finished_at"] = finished_at
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process every PDF in a directory with resumable MinerU jobs"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "/app/runtime/data")),
    )
    parser.add_argument("--render-dpi", type=int, default=160)
    parser.add_argument("--target-chars", type=int, default=12_000)
    parser.add_argument("--hard-max-chars", type=int, default=24_000)
    parser.add_argument("--max-document-attempts", type=int, default=3)
    parser.add_argument(
        "--mineru-provider",
        choices=("openai-compatible", "mineru-http-client"),
        default=os.environ.get("MINERU_PROVIDER", "openai-compatible"),
    )
    parser.add_argument("--parser-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--asset-mode",
        choices=("local", "minio"),
        default=os.environ.get("DATA_ASSET_MODE", "local"),
    )
    parser.add_argument("--ensure-bucket", action="store_true")
    parser.add_argument("--no-page-assets", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_document_attempts <= 5:
        parser.error("--max-document-attempts must be between 1 and 5")

    input_dir = args.input_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    pdfs = sorted(input_dir.rglob("*.pdf"), key=lambda item: item.name)
    if not pdfs:
        parser.error("input directory contains no PDF files")
    data_dir = args.data_dir
    jobs_dir = data_dir / "parser-jobs"
    processed_dir = data_dir / "processed"
    sources_dir = data_dir / "normalized-sources"
    for directory in (jobs_dir, processed_dir, sources_dir):
        directory.mkdir(parents=True, exist_ok=True)

    mineru_model = (
        os.environ.get("MINERU_MODEL") or openai_model()
    ).strip()
    if not mineru_model:
        parser.error("MINERU_MODEL or OPENAI_MODEL is required")
    if args.mineru_provider == "mineru-http-client":
        page_parser = OfficialMinerUPageParser(
            api_base=provider_base_url("MINERU"),
            api_key=provider_api_key("MINERU"),
            model=mineru_model,
            timeout=float(os.environ.get("MINERU_TIMEOUT", "180")),
            max_concurrency=int(os.environ.get("MINERU_MAX_CONCURRENCY", "8")),
        )
    else:
        page_parser = OpenAICompatiblePageParser(
            api_base=provider_base_url("MINERU"),
            api_key=provider_api_key("MINERU"),
            model=mineru_model,
            timeout=float(os.environ.get("MINERU_TIMEOUT", "180")),
            max_tokens=args.parser_max_tokens,
        )
    page_role_classifier = None
    if os.environ.get("PAGE_ROLE_CLASSIFIER", "off").lower() == "vlm":
        page_role_classifier = OpenAICompatiblePageRoleClassifier(
            api_base=provider_base_url("PAGE_ROLE"),
            api_key=provider_api_key("PAGE_ROLE"),
            model=os.environ.get("PAGE_ROLE_MODEL") or mineru_model,
            timeout=float(os.environ.get("PAGE_ROLE_TIMEOUT", "120")),
        )
    asset_writer = _writer(args)
    manifest_path = data_dir / "corpus-run.json"
    previous: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        saved = _load_json(manifest_path)
        previous = {
            item["source_sha256"]: item
            for item in saved.get("documents", [])
            if isinstance(item, dict) and isinstance(item.get("source_sha256"), str)
        }

    entries, documents = _prepare_documents(
        pdfs,
        jobs_dir=jobs_dir,
        processed_dir=processed_dir,
        sources_dir=sources_dir,
        previous=previous,
    )
    run_id = uuid.uuid4().hex
    run_started_at = _now()
    _write_json(
        manifest_path,
        _corpus_state(
            status="running",
            run_id=run_id,
            started_at=run_started_at,
            documents=documents,
        ),
    )

    for position, ((pdf, source_sha256, job_dir, destination), item) in enumerate(
        zip(entries, documents, strict=True),
        start=1,
    ):
        if _complete_output_exists(
            item,
            job_dir=job_dir,
            destination=destination,
        ):
            print(
                f"corpus.resume document={position}/{len(pdfs)} "
                f"source_sha256={source_sha256[:12]} status=complete",
                flush=True,
            )
            continue
        item["status"] = "running"
        item["started_at"] = item.get("started_at") or _now()
        item.pop("error", None)
        item.pop("failed_at", None)
        item.pop("finished_at", None)
        _write_json(
            manifest_path,
            _corpus_state(
                status="running",
                run_id=run_id,
                started_at=run_started_at,
                documents=documents,
            ),
        )
        for attempt in range(1, args.max_document_attempts + 1):
            item["status"] = "running"
            item.pop("error", None)
            item.pop("failed_at", None)
            item["attempts"] = int(item.get("attempts") or 0) + 1
            _write_json(
                manifest_path,
                _corpus_state(
                    status="running",
                    run_id=run_id,
                    started_at=run_started_at,
                    documents=documents,
                ),
            )
            print(
                f"corpus.document.start document={position}/{len(pdfs)} "
                f"source_sha256={source_sha256[:12]} attempt={attempt}",
                flush=True,
            )
            try:
                parse_pdf_with_vlm(
                    pdf,
                    job_dir,
                    parser=page_parser,
                    render_dpi=args.render_dpi,
                    max_attempts=2,
                    include_page_assets=not args.no_page_assets,
                    page_role_classifier=page_role_classifier,
                )
                result = process_mineru_output(
                    pdf,
                    job_dir,
                    destination,
                    asset_writer=asset_writer,
                    target_chars=args.target_chars,
                    hard_max_chars=args.hard_max_chars,
                )
                activated = activate_article(
                    Path(result.article_path),
                    Path(result.structure_path),
                    sources_dir,
                    ready=result.normalization_status == "complete",
                )
                item.update(
                    {
                        "status": "complete",
                        "finished_at": _now(),
                        "expected_pages": result.pages,
                        "metrics": _page_metrics(job_dir),
                        "result": result.to_dict(),
                        "activated_for_agentwiki": activated,
                    }
                )
                item.pop("error", None)
                item.pop("failed_at", None)
                print(
                    f"corpus.document.done document={position}/{len(pdfs)} "
                    f"source_sha256={source_sha256[:12]} pages={result.pages} "
                    f"blocks={result.blocks} segments={result.segments} "
                    f"assets={result.assets}",
                    flush=True,
                )
                break
            except Exception as error:
                stale_result = item.get("result") or {}
                stale_article = Path(
                    str(
                        stale_result.get("article_path")
                        or f"{pdf.stem}.md"
                    )
                ).name
                deactivate_article(stale_article, sources_dir)
                item.update(
                    {
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "failed_at": _now(),
                        "metrics": _page_metrics(job_dir),
                        "expected_pages": _load_expected_pages(job_dir),
                        "activated_for_agentwiki": False,
                    }
                )
                _write_json(
                    manifest_path,
                    _corpus_state(
                        status="running",
                        run_id=run_id,
                        started_at=run_started_at,
                        documents=documents,
                    ),
                )
                print(
                    f"corpus.document.retry document={position}/{len(pdfs)} "
                    f"source_sha256={source_sha256[:12]} attempt={attempt} "
                    f"error={type(error).__name__}",
                    flush=True,
                )
                if attempt < args.max_document_attempts:
                    time.sleep(_document_retry_delay(error, attempt))
        _write_json(
            manifest_path,
            _corpus_state(
                status="running",
                run_id=run_id,
                started_at=run_started_at,
                documents=documents,
            ),
        )

    totals = _summary(documents)
    status = "complete" if totals["failed_documents"] == 0 else "failed"
    _write_json(
        manifest_path,
        _corpus_state(
            status=status,
            run_id=run_id,
            started_at=run_started_at,
            documents=documents,
            finished_at=_now(),
        ),
    )
    print(
        "corpus.done "
        + " ".join(f"{key}={value}" for key, value in totals.items()),
        flush=True,
    )
    return 0 if status == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
