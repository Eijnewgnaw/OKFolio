"""Deterministic service layer exposed by the OKFolio MCP server.

The MCP adapter is intentionally thin.  This module owns path confinement,
write gates, background jobs, corpus operations, release publication and
knowledge lookup so the same behavior can be tested without an MCP client.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from kmpro_wiki.agentwiki.okf import (
    OKFValidationError,
    parse_concept_markdown,
)


SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SAFE_IMAGE_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,255}\.(?:png|jpe?g|gif|webp|svg)$",
    re.IGNORECASE,
)
IMAGE_TAG_RE = re.compile(
    r"^[a-z0-9]+(?:[._/-][a-z0-9]+)*(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?$"
)


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_id(value: str, *, label: str) -> str:
    candidate = value.strip()
    if not SAFE_ID_RE.fullmatch(candidate):
        raise ValueError(
            f"{label} must contain only letters, numbers, dot, underscore or dash"
        )
    return candidate


def _safe_markdown_name(value: str) -> str:
    candidate = Path(value.strip()).name
    if candidate != value.strip() or not candidate.lower().endswith(".md"):
        raise ValueError("filename must be one safe .md basename")
    if candidate in {".md", "..md"} or "\x00" in candidate:
        raise ValueError("unsafe Markdown filename")
    return candidate


def _safe_pdf_name(value: str) -> str:
    candidate = Path(value.strip()).name
    if candidate != value.strip() or not candidate.lower().endswith(".pdf"):
        raise ValueError("filename must be one safe .pdf basename")
    if candidate in {".pdf", "..pdf"} or "\x00" in candidate:
        raise ValueError("unsafe PDF filename")
    return candidate


def _within(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


@dataclass(frozen=True)
class MCPConfig:
    project_root: Path
    data_dir: Path
    prompts_dir: Path
    releases_dir: Path
    active_release_dir: Path | None
    enable_writes: bool
    enable_docker: bool
    max_document_bytes: int = 8 * 1024 * 1024
    max_asset_bytes: int = 24 * 1024 * 1024
    max_result_chars: int = 200_000
    source_dir_override: Path | None = None

    @property
    def sources_dir(self) -> Path:
        return self.source_dir_override or self.data_dir / "sources"

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "MCPConfig":
        values = os.environ if environ is None else environ
        project_root = Path(
            values.get(
                "OKFOLIO_PROJECT_ROOT",
                str(Path(__file__).resolve().parents[3]),
            )
        ).resolve()
        active = values.get("MCP_RELEASE_DIR", "").strip()
        return cls(
            project_root=project_root,
            data_dir=Path(
                values.get("DATA_DIR", str(project_root / "runtime" / "data"))
            ).resolve(),
            prompts_dir=Path(
                values.get("PROMPTS_DIR", str(project_root / "prompts"))
            ).resolve(),
            releases_dir=Path(
                values.get(
                    "RELEASES_DIR",
                    str(project_root / "artifacts" / "releases"),
                )
            ).resolve(),
            active_release_dir=Path(active).resolve() if active else None,
            enable_writes=_as_bool(values.get("MCP_ENABLE_WRITES", "false")),
            enable_docker=_as_bool(values.get("MCP_ENABLE_DOCKER", "false")),
            source_dir_override=(
                Path(values["SOURCES_DIR"]).resolve()
                if values.get("SOURCES_DIR", "").strip()
                else None
            ),
            max_document_bytes=int(
                values.get("MCP_MAX_DOCUMENT_BYTES", str(8 * 1024 * 1024))
            ),
            max_asset_bytes=int(
                values.get("MCP_MAX_ASSET_BYTES", str(24 * 1024 * 1024))
            ),
            max_result_chars=int(
                values.get("MCP_MAX_RESULT_CHARS", "200000")
            ),
        )


@dataclass(frozen=True)
class ReleaseView:
    label: str
    root: Path
    data_dir: Path

    @property
    def wiki_dir(self) -> Path:
        return self.data_dir / "wiki"

    @property
    def provenance_dir(self) -> Path:
        return self.data_dir / "provenance"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"


class JobManager:
    """Launch allowlisted command sequences through a persistent job record."""

    def __init__(self, config: MCPConfig):
        self.config = config
        self.jobs_dir = config.data_dir / ".mcp" / "jobs"

    def start(
        self,
        kind: str,
        commands: list[list[str]],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_kind = _safe_id(kind, label="job kind")
        if not commands or any(not command for command in commands):
            raise ValueError("at least one non-empty command is required")
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        job_id = (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            + f"-{safe_kind}-{uuid.uuid4().hex[:8]}"
        )
        record_path = self.jobs_dir / f"{job_id}.json"
        log_path = self.jobs_dir / f"{job_id}.log"
        record = {
            "job_id": job_id,
            "kind": safe_kind,
            "status": "queued",
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "commands": commands,
            "metadata": dict(metadata or {}),
            "log_path": str(log_path),
            "cwd": str(self.config.project_root),
            "environment": {
                "DATA_DIR": str(self.config.data_dir),
                "PROMPTS_DIR": str(self.config.prompts_dir),
                "PYTHONPATH": str(self.config.project_root),
            },
        }
        _write_json_atomic(record_path, record)
        runner = (
            self.config.project_root
            / "kmpro_wiki"
            / "mcp"
            / "job_runner.py"
        )
        process = subprocess.Popen(
            [sys.executable, str(runner), str(record_path)],
            cwd=self.config.project_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # The helper process records its own PID.  Do not rewrite the queued
        # record here: it may already have advanced it to ``running``.
        del process
        return self.get(job_id, include_log=False)

    def get(self, job_id: str, *, include_log: bool = True) -> dict[str, Any]:
        safe = _safe_id(job_id, label="job_id")
        path = self.jobs_dir / f"{safe}.json"
        if not path.is_file():
            raise FileNotFoundError(f"unknown job: {safe}")
        record = _load_json(path, {})
        record.pop("environment", None)
        if include_log:
            log_path = self.jobs_dir / f"{safe}.log"
            record["log_tail"] = self._tail(log_path)
        return record

    def list(self, *, limit: int = 30) -> list[dict[str, Any]]:
        bounded = max(1, min(limit, 200))
        if not self.jobs_dir.is_dir():
            return []
        paths = sorted(
            self.jobs_dir.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        return [
            self.get(path.stem, include_log=False) for path in paths[:bounded]
        ]

    def log(self, job_id: str, *, max_chars: int = 20_000) -> dict[str, Any]:
        record = self.get(job_id, include_log=False)
        path = self.jobs_dir / f"{record['job_id']}.log"
        return {
            "job_id": record["job_id"],
            "status": record["status"],
            "log": self._tail(path, max_chars=max_chars),
        }

    @staticmethod
    def _tail(path: Path, *, max_chars: int = 20_000) -> str:
        if not path.is_file():
            return ""
        content = path.read_text(encoding="utf-8", errors="replace")
        return content[-max(1, min(max_chars, 200_000)) :]


class WikiMCPService:
    """Complete OKFolio capability surface with explicit side-effect gates."""

    def __init__(self, config: MCPConfig | None = None):
        self.config = config or MCPConfig.from_env()
        self.jobs = JobManager(self.config)

    def capabilities(self) -> dict[str, Any]:
        return {
            "name": "OKFolio",
            "version": self._version(),
            "provenance_model": "Article -> ConceptRef -> Concept",
            "pipeline": [
                "pdf_ingest",
                "mineru_parse",
                "document_ir",
                "article_segmentation",
                "agent_discovery",
                "refinement",
                "cross_document_grouping",
                "concept_compile",
                "quality_recompile",
                "relation_judgement",
                "acceptance_audit",
                "release_publish",
                "graph_and_site",
                "container_package",
            ],
            "query_capabilities": [
                "list_pdfs",
                "get_processed_document",
                "search_concepts",
                "get_concept",
                "trace_concept",
                "get_article",
                "get_graph",
            ],
            "write_capabilities": [
                "ingest_markdown",
                "ingest_asset",
                "sync_inbox",
                "start_pdf_processing",
                "start_incremental_compile",
                "start_agent_compile",
                "start_relation_judgement",
                "audit_agent_run",
                "report_agent_run",
                "publish_release",
                "audit_release",
                "build_release_image",
                "package_release",
            ],
            "writes_enabled": self.config.enable_writes,
            "docker_enabled": self.config.enable_docker,
            "long_running_operations": "background jobs",
        }

    def system_status(self) -> dict[str, Any]:
        try:
            release = self.resolve_release("")
        except FileNotFoundError:
            release = None
        manifest = self._release_manifest(release) if release else {}
        runs = []
        runs_dir = self.config.data_dir / "agent-runs"
        if runs_dir.is_dir():
            for path in sorted(runs_dir.iterdir()):
                if not path.is_dir():
                    continue
                item = _load_json(path / "manifest.json", {})
                runs.append(
                    {
                        "run_id": path.name,
                        "status": item.get("status", "unknown"),
                        "articles": item.get("articles"),
                        "refs": item.get("refs"),
                        "concepts": item.get("concepts"),
                    }
                )
        return {
            "version": self._version(),
            "writes_enabled": self.config.enable_writes,
            "docker_enabled": self.config.enable_docker,
            "paths": {
                "data": str(self.config.data_dir),
                "prompts": str(self.config.prompts_dir),
                "releases": str(self.config.releases_dir),
            },
            "sources": len(list(self.config.sources_dir.glob("*.md"))),
            "inbox_markdown": len(
                list((self.config.data_dir / "inbox").glob("*.md"))
            ),
            "inbox_pdf": len(
                list((self.config.data_dir / "inbox").glob("*.pdf"))
            ),
            "processed_documents": len(
                list((self.config.data_dir / "processed").glob("*/manifest.json"))
            ),
            "active_release": {
                "label": release.label if release else None,
                "path": str(release.root) if release else None,
                "manifest": manifest,
            },
            "runs": runs,
            "jobs": self.jobs.list(limit=20),
        }

    def list_sources(self) -> dict[str, Any]:
        sources_dir = self.config.sources_dir
        sources = []
        for path in sorted(sources_dir.glob("*.md")):
            content = path.read_text(encoding="utf-8", errors="replace")
            heading = next(
                (
                    line.lstrip("#").strip()
                    for line in content.splitlines()
                    if line.startswith("#")
                ),
                path.stem,
            )
            sources.append(
                {
                    "filename": path.name,
                    "title": heading,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        return {"count": len(sources), "sources": sources}

    def list_pdfs(self) -> dict[str, Any]:
        inbox = self.config.data_dir / "inbox"
        documents = []
        for path in sorted(inbox.glob("*.pdf")):
            parsed = self.config.data_dir / "mineru-output" / path.stem
            processed = self.config.data_dir / "processed" / path.stem
            documents.append(
                {
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "mineru_output_ready": any(
                        parsed.rglob("*_content_list.json")
                    )
                    if parsed.is_dir()
                    else False,
                    "processed": (processed / "manifest.json").is_file(),
                }
            )
        return {"count": len(documents), "pdfs": documents}

    def ingest_markdown(
        self,
        filename: str,
        content: str,
        *,
        replace: bool = False,
        activate: bool = True,
    ) -> dict[str, Any]:
        self._require_writes("ingest_markdown")
        safe_name = _safe_markdown_name(filename)
        encoded = content.encode("utf-8")
        if not content.strip():
            raise ValueError("Markdown content must not be empty")
        if len(encoded) > self.config.max_document_bytes:
            raise ValueError("Markdown document exceeds MCP_MAX_DOCUMENT_BYTES")
        target = self.config.data_dir / "inbox" / safe_name
        if target.exists() and not replace:
            raise FileExistsError(
                f"{safe_name} already exists; set replace=true to update it"
            )
        _write_text_atomic(target, content)
        result = {
            "filename": safe_name,
            "bytes": len(encoded),
            "sha256": _sha256(target),
            "inbox_path": str(target),
            "activated": False,
        }
        if activate:
            sync = self.sync_inbox()
            result["activated"] = safe_name in sync["updated"]
            result["source_path"] = str(
                self.config.sources_dir / safe_name
            )
        return result

    def ingest_asset(
        self,
        filename: str,
        content_base64: str,
        *,
        replace: bool = False,
        activate: bool = True,
    ) -> dict[str, Any]:
        self._require_writes("ingest_asset")
        candidate = filename.strip().replace("\\", "/")
        if (
            not SAFE_IMAGE_RE.fullmatch(candidate)
            or candidate.startswith("/")
            or ".." in Path(candidate).parts
        ):
            raise ValueError("asset filename must be a safe relative image path")
        try:
            content = base64.b64decode(content_base64, validate=True)
        except ValueError as error:
            raise ValueError("content_base64 is not valid base64") from error
        if not content or len(content) > self.config.max_asset_bytes:
            raise ValueError("asset is empty or exceeds MCP_MAX_ASSET_BYTES")
        target = self.config.data_dir / "inbox" / "images" / candidate
        _within(target, self.config.data_dir / "inbox" / "images")
        if target.exists() and not replace:
            raise FileExistsError(
                f"{candidate} already exists; set replace=true to update it"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, target)
        result = {
            "filename": candidate,
            "bytes": len(content),
            "sha256": _sha256(target),
            "activated": False,
        }
        if activate:
            sync = self.sync_inbox()
            result["activated"] = candidate in sync["updated_images"]
            result["source_path"] = str(
                self.config.sources_dir / "images" / candidate
            )
        return result

    def sync_inbox(self) -> dict[str, Any]:
        self._require_writes("sync_inbox")
        inbox = self.config.data_dir / "inbox"
        sources = self.config.sources_dir
        inbox.mkdir(parents=True, exist_ok=True)
        sources.mkdir(parents=True, exist_ok=True)
        updated: list[str] = []
        unchanged: list[str] = []
        for source in sorted(inbox.glob("*.md")):
            target = sources / source.name
            if target.is_file() and _sha256(source) == _sha256(target):
                unchanged.append(source.name)
                continue
            self._copy_atomic(source, target)
            updated.append(source.name)
        updated_images: list[str] = []
        image_root = inbox / "images"
        if image_root.is_dir():
            for source in sorted(path for path in image_root.rglob("*") if path.is_file()):
                relative = source.relative_to(image_root)
                target = sources / "images" / relative
                if target.is_file() and _sha256(source) == _sha256(target):
                    continue
                self._copy_atomic(source, target)
                updated_images.append(relative.as_posix())
        return {
            "updated": updated,
            "unchanged": unchanged,
            "updated_images": updated_images,
            "deferred_pdfs": [path.name for path in sorted(inbox.glob("*.pdf"))],
        }

    def start_pdf_processing(
        self,
        filename: str,
        *,
        backend: str = "pipeline",
        target_chars: int = 12_000,
        hard_max_chars: int = 24_000,
        reuse_mineru_output: bool = True,
    ) -> dict[str, Any]:
        self._require_writes("start_pdf_processing")
        safe_name = _safe_pdf_name(filename)
        if backend not in {"pipeline", "hybrid", "vlm"}:
            raise ValueError("backend must be pipeline, hybrid or vlm")
        if target_chars < 500 or hard_max_chars < target_chars:
            raise ValueError("invalid segmentation character limits")
        pdf = self.config.data_dir / "inbox" / safe_name
        _within(pdf, self.config.data_dir / "inbox")
        if not pdf.is_file():
            raise FileNotFoundError(f"PDF is not available in inbox: {safe_name}")
        mineru_output = self.config.data_dir / "mineru-output" / pdf.stem
        destination = self.config.data_dir / "processed" / pdf.stem
        command = [
            sys.executable,
            str(self.config.project_root / "scripts/process_pdf.py"),
            "--pdf",
            str(pdf),
            "--mineru-output",
            str(mineru_output),
            "--destination",
            str(destination),
            "--backend",
            backend,
            "--target-chars",
            str(target_chars),
            "--hard-max-chars",
            str(hard_max_chars),
            "--activate-dir",
            str(self.config.sources_dir),
        ]
        has_parsed_output = (
            reuse_mineru_output
            and mineru_output.is_dir()
            and (
                any(mineru_output.rglob("*_content_list.json"))
                or any(mineru_output.rglob("*_content_list_v2.json"))
            )
        )
        if has_parsed_output:
            command.append("--skip-mineru")
        return self.jobs.start(
            "pdf-processing",
            [command],
            metadata={
                "filename": safe_name,
                "backend": backend,
                "reuse_mineru_output": has_parsed_output,
                "destination": str(destination),
            },
        )

    def get_processed_document(self, document_name: str) -> dict[str, Any]:
        safe = _safe_id(document_name, label="document_name")
        root = self.config.data_dir / "processed" / safe
        _within(root, self.config.data_dir / "processed")
        manifest = _load_json(root / "manifest.json")
        if not isinstance(manifest, dict):
            raise FileNotFoundError(f"processed document is unavailable: {safe}")
        segments = _load_json(root / "segments.json", {})
        assets = _load_json(root / "asset-manifest.json", {})
        return {
            "document_name": safe,
            "manifest": manifest,
            "segment_count": len(segments.get("segments", [])),
            "asset_count": len(assets.get("assets", [])),
            "article_path": manifest.get("article_path"),
        }

    def start_incremental_compile(
        self,
        *,
        sync_inbox: bool = True,
    ) -> dict[str, Any]:
        self._require_writes("start_incremental_compile")
        self._require_llm(require_key=False)
        if sync_inbox:
            self.sync_inbox()
        command = ["bash", str(self.config.project_root / "scripts/process_inbox.sh")]
        return self.jobs.start(
            "incremental-compile",
            [command],
            metadata={"mode": "A-D incremental compiler"},
        )

    def start_agent_compile(
        self,
        run_id: str,
        *,
        resume: bool = False,
        sync_inbox: bool = True,
        quality_threshold: float = 0.82,
        max_recompile_attempts: int = 2,
        max_component_refs: int = 24,
        max_component_chars: int = 42_000,
    ) -> dict[str, Any]:
        self._require_writes("start_agent_compile")
        self._require_llm(require_key=False)
        safe_run = _safe_id(run_id, label="run_id")
        if not 0.0 <= quality_threshold <= 1.0:
            raise ValueError("quality_threshold must be between 0 and 1")
        if not 0 <= max_recompile_attempts <= 5:
            raise ValueError("max_recompile_attempts must be between 0 and 5")
        if not 2 <= max_component_refs <= 100:
            raise ValueError("max_component_refs must be between 2 and 100")
        if not 8_000 <= max_component_chars <= 500_000:
            raise ValueError("max_component_chars must be between 8000 and 500000")
        if sync_inbox:
            self.sync_inbox()
        command = [
            sys.executable,
            str(self.config.project_root / "scripts/agent_compile.py"),
            "--run-id",
            safe_run,
            "--quality-threshold",
            str(quality_threshold),
            "--max-recompile-attempts",
            str(max_recompile_attempts),
            "--max-component-refs",
            str(max_component_refs),
            "--max-component-chars",
            str(max_component_chars),
        ]
        if resume:
            command.append("--resume")
        return self.jobs.start(
            "agent-compile",
            [command],
            metadata={
                "run_id": safe_run,
                "resume": resume,
                "quality_threshold": quality_threshold,
                "max_recompile_attempts": max_recompile_attempts,
                "max_component_refs": max_component_refs,
                "max_component_chars": max_component_chars,
            },
        )

    def start_relation_judgement(
        self,
        run_id: str,
        *,
        resume: bool = False,
        batch_size: int = 16,
    ) -> dict[str, Any]:
        self._require_writes("start_relation_judgement")
        self._require_llm(require_key=False)
        safe_run = _safe_id(run_id, label="run_id")
        self._run_dir(safe_run, must_exist=True)
        if not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be between 1 and 64")
        command = [
            sys.executable,
            str(self.config.project_root / "scripts/judge_agent_relations.py"),
            str(self._run_dir(safe_run, must_exist=True)),
            "--batch-size",
            str(batch_size),
        ]
        if resume:
            command.append("--resume")
        return self.jobs.start(
            "relation-judgement",
            [command],
            metadata={"run_id": safe_run, "resume": resume},
        )

    def audit_agent_run(self, run_id: str) -> dict[str, Any]:
        self._require_writes("audit_agent_run")
        from kmpro_wiki.agentwiki.audit_run import run_audit

        run_dir = self._run_dir(run_id, must_exist=True)
        result = run_audit(run_dir, self.config.sources_dir)
        _write_json_atomic(run_dir / "acceptance.json", result)
        return result

    def report_agent_run(self, run_id: str) -> dict[str, Any]:
        self._require_writes("report_agent_run")
        from kmpro_wiki.agentwiki.report import (
            render_markdown,
            summarize,
        )

        run_dir = self._run_dir(run_id, must_exist=True)
        log = self.config.data_dir / "agent-runs" / f"{run_dir.name}.log"
        metrics = summarize(run_dir, log if log.is_file() else None)
        _write_json_atomic(run_dir / "experiment-metrics.json", metrics)
        _write_text_atomic(
            run_dir / "experiment-report.md",
            render_markdown(metrics),
        )
        return metrics

    def publish_release(
        self,
        run_id: str,
        release_name: str,
        *,
        version: str,
        replace: bool = False,
        confirm_replace: str = "",
    ) -> dict[str, Any]:
        self._require_writes("publish_release")
        safe_run = _safe_id(run_id, label="run_id")
        safe_release = _safe_id(release_name, label="release_name")
        safe_version = _safe_id(version, label="version")
        run_dir = self._run_dir(safe_run, must_exist=True)
        release_dir = self.config.releases_dir / safe_release
        if release_dir.exists() and (
            not replace or confirm_replace != safe_release
        ):
            raise FileExistsError(
                "release exists; set replace=true and confirm_replace to the "
                "exact release_name"
            )
        command = [
            sys.executable,
            str(self.config.project_root / "scripts/publish_agent_release.py"),
            str(run_dir),
            "--sources-dir",
            str(self.config.sources_dir),
            "--release-dir",
            str(release_dir),
            "--version",
            safe_version,
        ]
        if replace:
            command.append("--replace")
        return self.jobs.start(
            "publish-release",
            [command],
            metadata={
                "run_id": safe_run,
                "release_name": safe_release,
                "version": safe_version,
            },
        )

    def audit_release(self, release_name: str = "") -> dict[str, Any]:
        self._require_writes("audit_release")
        from kmpro_wiki.agentwiki.audit_release import audit_release

        release = self.resolve_release(release_name)
        if release.root == self.config.data_dir:
            raise ValueError("live data is not a formal release directory")
        result = audit_release(release.root)
        _write_json_atomic(release.root / "acceptance.json", result)
        return result

    def build_release_image(
        self,
        release_name: str,
        *,
        image_tag: str,
        export_tar: bool = True,
    ) -> dict[str, Any]:
        self._require_writes("build_release_image")
        if not self.config.enable_docker:
            raise PermissionError(
                "Docker operations are disabled; set MCP_ENABLE_DOCKER=true"
            )
        if not IMAGE_TAG_RE.fullmatch(image_tag):
            raise ValueError("image_tag is not a safe Docker image tag")
        release = self.resolve_release(release_name)
        if release.root == self.config.data_dir:
            raise ValueError("build_release_image requires a formal release")
        relative = release.root.resolve().relative_to(
            self.config.project_root.resolve()
        )
        acceptance = _load_json(release.root / "acceptance.json", {})
        if acceptance.get("status") != "pass":
            raise ValueError("release must pass audit before image build")
        dockerfile = self.config.project_root / "Dockerfile"
        if not dockerfile.is_file():
            raise FileNotFoundError("release image Dockerfile is unavailable")
        commands = [
            [
                "docker",
                "buildx",
                "build",
                "--platform",
                "linux/amd64",
                "--load",
                "--target",
                "release",
                "-f",
                str(dockerfile),
                "-t",
                image_tag,
                "--build-arg",
                f"RELEASE_PATH={relative.as_posix()}",
                ".",
            ]
        ]
        image_tar = None
        if export_tar:
            image_dir = release.root / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            safe_tar = re.sub(r"[^A-Za-z0-9_.-]+", "-", image_tag) + ".tar"
            image_tar = image_dir / safe_tar
            commands.append(
                ["docker", "save", "-o", str(image_tar), image_tag]
            )
        return self.jobs.start(
            "build-image",
            commands,
            metadata={
                "release_name": release.label,
                "image_tag": image_tag,
                "platform": "linux/amd64",
                "image_tar": str(image_tar) if image_tar else None,
            },
        )

    def package_release(
        self,
        release_name: str,
        *,
        archive_name: str = "",
    ) -> dict[str, Any]:
        self._require_writes("package_release")
        release = self.resolve_release(release_name)
        if release.root == self.config.data_dir:
            raise ValueError("package_release requires a formal release")
        archive = (
            _safe_id(archive_name, label="archive_name")
            if archive_name
            else f"{release.label}.tar.gz"
        )
        if not archive.endswith(".tar.gz"):
            archive += ".tar.gz"
        command = [
            sys.executable,
            str(self.config.project_root / "scripts/package_release.py"),
            str(release.root),
            "--archive",
            str(self.config.releases_dir / archive),
        ]
        return self.jobs.start(
            "package-release",
            [command],
            metadata={
                "release_name": release.label,
                "archive": str(self.config.releases_dir / archive),
            },
        )

    def list_jobs(self, *, limit: int = 30) -> dict[str, Any]:
        jobs = self.jobs.list(limit=limit)
        return {"count": len(jobs), "jobs": jobs}

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self.jobs.get(job_id)

    def get_job_log(
        self,
        job_id: str,
        *,
        max_chars: int = 20_000,
    ) -> dict[str, Any]:
        return self.jobs.log(job_id, max_chars=max_chars)

    def search_concepts(
        self,
        query: str = "",
        *,
        concept_type: str = "",
        source: str = "",
        limit: int = 20,
        release_name: str = "",
    ) -> dict[str, Any]:
        release = self.resolve_release(release_name)
        bounded = max(1, min(limit, 100))
        query_text = query.strip().casefold()
        query_terms = [item for item in re.split(r"\s+", query_text) if item]
        matches = []
        for path in sorted((release.wiki_dir / "concepts").glob("*.md")):
            try:
                document = parse_concept_markdown(
                    path.name, path.read_text(encoding="utf-8")
                )
            except OKFValidationError:
                continue
            metadata = document.frontmatter
            if concept_type and metadata.get("type") != concept_type:
                continue
            if source and source.casefold() not in str(
                metadata.get("source", "")
            ).casefold():
                continue
            fields = {
                "title": str(metadata.get("title", "")),
                "description": str(metadata.get("description", "")),
                "source": str(metadata.get("source", "")),
                "body": document.body,
            }
            lowered = {key: value.casefold() for key, value in fields.items()}
            if query_terms and not all(
                any(term in value for value in lowered.values())
                for term in query_terms
            ):
                continue
            score = sum(
                8 * lowered["title"].count(term)
                + 4 * lowered["description"].count(term)
                + 2 * lowered["source"].count(term)
                + lowered["body"].count(term)
                for term in query_terms
            )
            matches.append(
                {
                    "concept_id": path.stem,
                    "type": metadata.get("type"),
                    "title": fields["title"],
                    "description": fields["description"],
                    "source": fields["source"],
                    "articles": metadata.get("articles", []),
                    "concept_refs": metadata.get("concept_refs", []),
                    "relation_count": metadata.get("relation_count", 0),
                    "score": score,
                }
            )
        matches.sort(key=lambda item: (-item["score"], item["title"]))
        return {
            "release": release.label,
            "query": query,
            "total_matches": len(matches),
            "results": matches[:bounded],
        }

    def get_concept(
        self,
        concept_id: str,
        *,
        release_name: str = "",
    ) -> dict[str, Any]:
        release = self.resolve_release(release_name)
        safe = _safe_id(concept_id.removesuffix(".md"), label="concept_id")
        path = release.wiki_dir / "concepts" / f"{safe}.md"
        if not path.is_file():
            raise FileNotFoundError(f"unknown Concept: {safe}")
        content = path.read_text(encoding="utf-8")
        document = parse_concept_markdown(path.name, content)
        return {
            "release": release.label,
            "concept_id": safe,
            "frontmatter": document.frontmatter,
            "body": document.body,
            "markdown": self._bounded(content),
        }

    def trace_concept(
        self,
        concept_id: str,
        *,
        include_evidence: bool = True,
        release_name: str = "",
    ) -> dict[str, Any]:
        release = self.resolve_release(release_name)
        concept = self.get_concept(concept_id, release_name=release_name)
        safe = concept["concept_id"]
        refs_payload = _load_json(release.provenance_dir / "refs.json", {})
        groups_payload = _load_json(
            release.provenance_dir / "groups.json", {}
        )
        relations_payload = _load_json(
            release.provenance_dir / "relations.json", {}
        )
        refs_by_id = {
            item["ref_id"]: item for item in refs_payload.get("refs", [])
        }
        groups = groups_payload.get("groups", [])
        ref_to_group = {
            ref_id: item["group_id"]
            for item in groups
            for ref_id in item.get("ref_ids", [])
        }
        group_titles = {
            item["group_id"]: item.get("title", item["group_id"])
            for item in groups
        }
        ref_ids = concept["frontmatter"].get("concept_refs", [])
        refs = []
        for ref_id in ref_ids:
            item = dict(refs_by_id.get(ref_id, {"ref_id": ref_id}))
            if not include_evidence:
                item.pop("evidence", None)
            refs.append(item)
        articles = []
        for article_id in concept["frontmatter"].get("articles", []):
            article_path = (
                release.wiki_dir / "articles" / f"{article_id}.md"
            )
            if not article_path.is_file():
                articles.append({"article_id": article_id})
                continue
            article = parse_concept_markdown(
                article_path.name,
                article_path.read_text(encoding="utf-8"),
            )
            articles.append(
                {
                    "article_id": article_id,
                    "title": article.frontmatter.get("title"),
                    "source": article.frontmatter.get("source"),
                }
            )
        related: dict[str, dict[str, Any]] = {}
        for judgement in relations_payload.get("judgements", []):
            if judgement.get("decision") != "related":
                continue
            left_group = ref_to_group.get(judgement.get("left_ref_id", ""))
            right_group = ref_to_group.get(judgement.get("right_ref_id", ""))
            if safe not in {left_group, right_group}:
                continue
            target = right_group if left_group == safe else left_group
            if not target or target == safe:
                continue
            entry = related.setdefault(
                target,
                {
                    "concept_id": target,
                    "title": group_titles.get(target, target),
                    "evidence": [],
                },
            )
            entry["evidence"].append(
                {
                    "left_ref_id": judgement.get("left_ref_id"),
                    "right_ref_id": judgement.get("right_ref_id"),
                    "reason": judgement.get("reason"),
                }
            )
        return {
            "release": release.label,
            "concept_id": safe,
            "title": concept["frontmatter"].get("title"),
            "concept_refs": refs,
            "articles": articles,
            "related_concepts": sorted(
                related.values(), key=lambda item: item["title"]
            ),
            "provenance": "Concept -> ConceptRef -> Article",
        }

    def get_article(
        self,
        article_id: str,
        *,
        release_name: str = "",
    ) -> dict[str, Any]:
        release = self.resolve_release(release_name)
        safe = _safe_id(article_id.removesuffix(".md"), label="article_id")
        path = release.wiki_dir / "articles" / f"{safe}.md"
        if not path.is_file():
            raise FileNotFoundError(f"unknown Article: {safe}")
        content = path.read_text(encoding="utf-8")
        document = parse_concept_markdown(path.name, content)
        return {
            "release": release.label,
            "article_id": safe,
            "frontmatter": document.frontmatter,
            "body": document.body,
            "markdown": self._bounded(content),
        }

    def graph_info(self, release_name: str = "") -> dict[str, Any]:
        release = self.resolve_release(release_name)
        path = release.outputs_dir / "graph.html"
        if not path.is_file():
            path = release.wiki_dir / "graph.html"
        if not path.is_file():
            raise FileNotFoundError("graph.html is not available")
        return {
            "release": release.label,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "views": [
                "三维关系编织网",
                "三维知识球",
                "全部概念查阅",
            ],
        }

    def graph_html(self, release_name: str = "") -> str:
        info = self.graph_info(release_name)
        return Path(info["path"]).read_text(encoding="utf-8")

    def concept_markdown_resource(self, concept_id: str) -> str:
        return self.get_concept(concept_id)["markdown"]

    def article_markdown_resource(self, article_id: str) -> str:
        return self.get_article(article_id)["markdown"]

    def resolve_release(self, release_name: str) -> ReleaseView:
        candidate = release_name.strip()
        if candidate == "live":
            return self._live_view()
        if candidate:
            safe = _safe_id(candidate, label="release_name")
            root = self.config.releases_dir / safe
            return self._formal_view(safe, root)
        if self.config.active_release_dir is not None:
            return self._formal_view(
                self.config.active_release_dir.name,
                self.config.active_release_dir,
            )
        if self.config.releases_dir.is_dir():
            candidates = [
                path
                for path in self.config.releases_dir.iterdir()
                if path.is_dir()
                and (path / "data" / "wiki" / "concepts").is_dir()
                and (path / "release-manifest.json").is_file()
            ]
            if candidates:
                root = max(
                    candidates,
                    key=lambda path: (
                        path.stat().st_mtime,
                        path.name,
                    ),
                )
                return self._formal_view(root.name, root)
        return self._live_view()

    def _formal_view(self, label: str, root: Path) -> ReleaseView:
        if not (root / "data" / "wiki" / "concepts").is_dir():
            raise FileNotFoundError(f"release is not available: {label}")
        return ReleaseView(label=label, root=root, data_dir=root / "data")

    def _live_view(self) -> ReleaseView:
        if not (self.config.data_dir / "wiki" / "concepts").is_dir():
            raise FileNotFoundError("no active Wiki Bundle is available")
        embedded_manifest = (
            _load_json(
                self.config.project_root / "release-manifest.json", {}
            )
            if self.config.data_dir
            == (self.config.project_root / "data").resolve()
            else {}
        )
        label = (
            embedded_manifest.get("version", "live")
            if isinstance(embedded_manifest, dict)
            else "live"
        )
        return ReleaseView(
            label=str(label),
            root=self.config.data_dir,
            data_dir=self.config.data_dir,
        )

    def _release_manifest(self, release: ReleaseView) -> dict[str, Any]:
        if release.root == self.config.data_dir:
            embedded = (
                _load_json(
                    self.config.project_root / "release-manifest.json", {}
                )
                if self.config.data_dir
                == (self.config.project_root / "data").resolve()
                else {}
            )
            if isinstance(embedded, dict) and embedded:
                return embedded
            manifest = _load_json(
                release.provenance_dir / "manifest.json", {}
            )
            return manifest if isinstance(manifest, dict) else {}
        manifest = _load_json(release.root / "release-manifest.json", {})
        return manifest if isinstance(manifest, dict) else {}

    def _run_dir(self, run_id: str, *, must_exist: bool) -> Path:
        safe = _safe_id(run_id, label="run_id")
        path = self.config.data_dir / "agent-runs" / safe
        _within(path, self.config.data_dir / "agent-runs")
        if must_exist and not path.is_dir():
            raise FileNotFoundError(f"unknown Agent run: {safe}")
        return path

    def _require_writes(self, operation: str) -> None:
        if not self.config.enable_writes:
            raise PermissionError(
                f"{operation} is disabled; set MCP_ENABLE_WRITES=true"
            )

    @staticmethod
    def _require_llm(*, require_key: bool = True) -> None:
        required = ["OPENAI_MODEL"]
        if require_key:
            required.append("OPENAI_API_KEY")
        missing = [
            name
            for name in required
            if not os.environ.get(name)
        ]
        if missing:
            raise ValueError(
                "missing model configuration: " + ", ".join(missing)
            )

    @staticmethod
    def _copy_atomic(source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)

    def _bounded(self, content: str) -> str:
        if len(content) <= self.config.max_result_chars:
            return content
        return (
            content[: self.config.max_result_chars]
            + "\n\n[内容因 MCP_MAX_RESULT_CHARS 限制而截断]"
        )

    def _version(self) -> str:
        path = self.config.project_root / "VERSION"
        return (
            path.read_text(encoding="utf-8").strip()
            if path.is_file()
            else "unknown"
        )
