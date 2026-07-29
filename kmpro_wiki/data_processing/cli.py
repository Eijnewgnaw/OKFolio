"""CLI for MinerU execution, IR normalization and AgentWiki activation."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from pathlib import Path

from .activation import activate_article
from .mineru_official import OfficialMinerUPageParser
from .pdf_worker import parse_pdf_with_vlm
from .page_role import OpenAICompatiblePageRoleClassifier
from .pipeline import process_mineru_output
from .s3 import S3CompatibleAssetWriter
from .storage import LocalAssetWriter, S3WriterAssetWriter
from .vlm import OpenAICompatiblePageParser


def _run_mineru(
    pdf: Path,
    output: Path,
    *,
    backend: str,
    command: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [command, "-p", str(pdf), "-o", str(output), "-b", backend],
        check=True,
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
        description="Convert one PDF into KMPro Article/Segment IR"
    )
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--mineru-output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--backend", choices=("pipeline", "hybrid", "vlm"), default="pipeline")
    parser.add_argument("--skip-mineru", action="store_true")
    parser.add_argument(
        "--mineru-provider",
        choices=("cli", "mineru-http-client", "openai-compatible"),
        default=os.environ.get("MINERU_PROVIDER", "cli"),
    )
    parser.add_argument("--mineru-command", default=os.environ.get("MINERU_COMMAND", "mineru"))
    parser.add_argument("--page-start", type=int, default=1)
    parser.add_argument("--page-end", type=int)
    parser.add_argument("--render-dpi", type=int, default=160)
    parser.add_argument("--parser-max-tokens", type=int, default=4096)
    parser.add_argument("--parser-max-attempts", type=int, default=2)
    parser.add_argument("--no-page-assets", action="store_true")
    parser.add_argument("--target-chars", type=int, default=12_000)
    parser.add_argument("--hard-max-chars", type=int, default=24_000)
    parser.add_argument("--activate-dir", type=Path)
    parser.add_argument(
        "--asset-mode",
        choices=("local", "s3writer", "minio"),
        default=os.environ.get("DATA_ASSET_MODE", "local"),
    )
    args = parser.parse_args()
    if not args.skip_mineru:
        if args.mineru_provider in {
            "mineru-http-client",
            "openai-compatible",
        }:
            api_base = os.environ.get("MINERU_API_BASE") or os.environ.get(
                "LLM_API_BASE", ""
            )
            api_key = os.environ.get("MINERU_API_KEY") or os.environ.get(
                "LLM_API_KEY", ""
            )
            model = os.environ.get("MINERU_MODEL", "").strip()
            if not model:
                parser.error(
                    "MINERU_MODEL is required for HTTP MinerU providers"
                )
            timeout = float(os.environ.get("MINERU_TIMEOUT", "180"))
            if args.mineru_provider == "mineru-http-client":
                page_parser = OfficialMinerUPageParser(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    timeout=timeout,
                    max_concurrency=int(
                        os.environ.get("MINERU_MAX_CONCURRENCY", "8")
                    ),
                )
            else:
                page_parser = OpenAICompatiblePageParser(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    timeout=timeout,
                    max_tokens=args.parser_max_tokens,
                )
            page_role_classifier = None
            if os.environ.get("PAGE_ROLE_CLASSIFIER", "off").lower() == "vlm":
                page_role_classifier = OpenAICompatiblePageRoleClassifier(
                    api_base=os.environ.get("PAGE_ROLE_API_BASE")
                    or api_base,
                    api_key=os.environ.get("PAGE_ROLE_API_KEY")
                    or api_key,
                    model=os.environ.get("PAGE_ROLE_MODEL") or model,
                    timeout=float(
                        os.environ.get("PAGE_ROLE_TIMEOUT", "120")
                    ),
                )
            parse_pdf_with_vlm(
                args.pdf,
                args.mineru_output,
                parser=page_parser,
                page_start=args.page_start - 1,
                page_end=args.page_end,
                render_dpi=args.render_dpi,
                max_attempts=args.parser_max_attempts,
                include_page_assets=not args.no_page_assets,
                page_role_classifier=page_role_classifier,
            )
        else:
            _run_mineru(
                args.pdf,
                args.mineru_output,
                backend=args.backend,
                command=args.mineru_command,
            )
    if args.asset_mode == "s3writer":
        writer = S3WriterAssetWriter.from_factory(
            os.environ["S3_WRITER_FACTORY"],
            bucket=os.environ["S3_BUCKET"],
            prefix=os.environ.get("S3_PREFIX", "okfolio"),
        )
    elif args.asset_mode == "minio":
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
    else:
        writer = LocalAssetWriter(args.destination / "images")
    result = process_mineru_output(
        args.pdf,
        args.mineru_output,
        args.destination,
        asset_writer=writer,
        target_chars=args.target_chars,
        hard_max_chars=args.hard_max_chars,
    )
    output = result.to_dict()
    if args.activate_dir is not None:
        activated = activate_article(
            Path(result.article_path),
            Path(result.structure_path),
            args.activate_dir,
            ready=result.normalization_status == "complete",
        )
        output["activated_for_agentwiki"] = activated
        manifest_path = Path(result.manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["activated_for_agentwiki"] = activated
        _write_json(manifest_path, manifest)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return (
        3
        if args.activate_dir is not None
        and output.get("activated_for_agentwiki") is not True
        else 0
    )
