"""Atomic activation gate between PDF normalization and AgentWiki."""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path


def _article_name(value: str) -> str:
    candidate = Path(value).name
    if (
        candidate != value
        or not candidate.lower().endswith(".md")
        or candidate in {".md", "..md"}
    ):
        raise ValueError("activation requires one safe Markdown basename")
    return candidate


def activate_article(
    article_path: Path,
    structure_path: Path,
    sources_dir: Path,
    *,
    ready: bool,
    article_name: str | None = None,
) -> bool:
    """Publish one normalized Article only after its structure gate passes.

    The structure manifest is copied first and the Markdown file last.  The
    Markdown file therefore acts as the activation marker consumed by
    AgentWiki.  A failed gate removes any stale prior activation.
    """
    article = article_path.resolve()
    structure = structure_path.resolve()
    target_root = sources_dir.resolve()
    safe_name = _article_name(article_name or article.name)
    target_article = target_root / safe_name
    target_structure = target_root / f"{Path(safe_name).stem}.structure.json"
    target_root.mkdir(parents=True, exist_ok=True)

    if not ready:
        deactivate_article(safe_name, target_root)
        return False

    token = uuid.uuid4().hex
    staged_article = target_root / f".{safe_name}.{token}.tmp"
    staged_structure = target_root / (
        f".{Path(safe_name).stem}.structure.json.{token}.tmp"
    )
    try:
        if not article.is_file():
            raise FileNotFoundError(
                f"activation source is unavailable: {article}"
            )
        if not structure.is_file():
            raise FileNotFoundError(
                f"activation source is unavailable: {structure}"
            )
        shutil.copy2(article, staged_article)
        shutil.copy2(structure, staged_structure)
        # Remove the Markdown activation marker before committing either new
        # file.  A failed update can therefore never leave stale content active.
        target_article.unlink(missing_ok=True)
        os.replace(staged_structure, target_structure)
        os.replace(staged_article, target_article)
    except BaseException:
        target_article.unlink(missing_ok=True)
        raise
    finally:
        staged_article.unlink(missing_ok=True)
        staged_structure.unlink(missing_ok=True)
    return True


def deactivate_article(article_name: str, sources_dir: Path) -> None:
    """Remove both activation markers for one normalized Article."""
    safe_name = _article_name(article_name)
    target_root = sources_dir.resolve()
    (target_root / safe_name).unlink(missing_ok=True)
    (target_root / f"{Path(safe_name).stem}.structure.json").unlink(
        missing_ok=True
    )
