from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_collection(path: Path | None, key: str) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get(key, []) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError(f"{path.name} must contain a {key} list")
    return [item for item in values if isinstance(item, dict)]


def audit_concept_rag_inputs(
    *,
    source_dir: Path,
    refs_path: Path | None,
    concepts_path: Path | None,
) -> dict[str, Any]:
    """Check that a Concept-as-Chunk comparison covers the entire corpus."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    source_names = sorted(path.name for path in source_dir.glob("*.md"))
    source_set = set(source_names)
    refs = _load_collection(refs_path, "refs")
    concepts = _load_collection(concepts_path, "concepts")

    ref_ids = {
        str(item.get("ref_id", "")).strip()
        for item in refs
        if str(item.get("ref_id", "")).strip()
    }
    ref_sources = {
        Path(str(item.get("source", item.get("source_file", "")))).name
        for item in refs
        if str(item.get("source", item.get("source_file", ""))).strip()
    }
    concept_sources = {
        Path(str(source)).name
        for concept in concepts
        for source in concept.get("sources", [])
        if str(source).strip()
    }
    concept_ref_ids = {
        str(ref_id).strip()
        for concept in concepts
        for ref_id in concept.get("ref_ids", [])
        if str(ref_id).strip()
    }
    multi_source_concepts = sum(
        1
        for concept in concepts
        if len({Path(str(item)).name for item in concept.get("sources", [])}) > 1
    )

    uncovered_ref_sources = sorted(source_set - ref_sources)
    uncovered_concept_sources = sorted(source_set - concept_sources)
    orphan_ref_sources = sorted(ref_sources - source_set)
    orphan_concept_sources = sorted(concept_sources - source_set)
    dangling_concept_ref_ids = sorted(concept_ref_ids - ref_ids)
    ready = bool(source_names) and not any(
        (
            uncovered_ref_sources,
            uncovered_concept_sources,
            orphan_ref_sources,
            orphan_concept_sources,
            dangling_concept_ref_ids,
        )
    )

    return {
        "schema": "okfolio.concept-rag-readiness.v1",
        "status": "ready" if ready else "incomplete",
        "ready_for_full_comparison": ready,
        "source_articles": len(source_names),
        "concept_refs": len(refs),
        "ref_source_articles": len(ref_sources & source_set),
        "concepts": len(concepts),
        "concept_source_articles": len(concept_sources & source_set),
        "multi_source_concepts": multi_source_concepts,
        "uncovered_ref_sources": uncovered_ref_sources,
        "uncovered_concept_sources": uncovered_concept_sources,
        "orphan_ref_sources": orphan_ref_sources,
        "orphan_concept_sources": orphan_concept_sources,
        "dangling_concept_ref_ids": dangling_concept_ref_ids,
    }
