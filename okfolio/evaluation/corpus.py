"""Framework-neutral corpus builders for controlled RAG comparisons.

The three builders deliberately share the same evidence identities:

``T0``
    Block-preserving, fixed-character chunks.
``T1``
    Heading-aware child chunks whose generation context expands to a bounded
    semantic parent.
``C1``
    Audited AgentWiki Concepts, mapped back to their source blocks through
    ConceptRef provenance.

No embedding, retriever, model, or third-party RAG framework is imported here.
That keeps corpus construction deterministic and makes it possible to compare
different retrieval stacks against exactly the same source evidence atoms.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from okfolio.agentwiki.okf import parse_concept_markdown

from .contracts import EvidenceAtomId, RetrievedUnit


Arm = Literal["T0", "T1", "C1"]
TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class EvidenceAtom:
    """Canonical source evidence represented by one eligible structure block."""

    atom_id: EvidenceAtomId
    article_id: str
    source_file: str
    page: int
    block_id: str
    block_type: str
    heading_path: tuple[str, ...]
    text: str
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["atom_id"] = str(self.atom_id)
        payload["heading_path"] = list(self.heading_path)
        return payload


@dataclass(frozen=True)
class CorpusBuild:
    arm: Arm
    units: tuple[RetrievedUnit, ...]
    evidence_atoms: tuple[EvidenceAtom, ...]

    @property
    def article_ids(self) -> tuple[str, ...]:
        return tuple(sorted({atom.article_id for atom in self.evidence_atoms}))

    def audit(self) -> dict[str, Any]:
        catalog = {atom.atom_id for atom in self.evidence_atoms}
        represented = {
            atom_id
            for unit in self.units
            for atom_id in unit.context_evidence_atom_ids
        }
        retrieval_atoms = {
            atom_id
            for unit in self.units
            for atom_id in unit.retrieval_evidence_atom_ids
        }
        unknown = sorted(str(item) for item in (represented | retrieval_atoms) - catalog)
        missing = sorted(str(item) for item in catalog - represented)
        represented_articles = {
            article_id for unit in self.units for article_id in unit.article_ids
        }
        missing_articles = sorted(set(self.article_ids) - represented_articles)
        oversized = sum(bool(unit.metadata.get("oversized")) for unit in self.units)
        full_atom_coverage_required = self.arm in {"T0", "T1"}
        passed = (
            bool(self.units)
            and not unknown
            and not missing_articles
            and (not full_atom_coverage_required or not missing)
        )
        return {
            "schema": "okfolio.rag-corpus-audit.v1",
            "arm": self.arm,
            "articles": len(self.article_ids),
            "units": len(self.units),
            "evidence_atoms": len(catalog),
            "represented_evidence_atoms": len(represented),
            "retrieval_evidence_atoms": len(retrieval_atoms),
            "missing_context_evidence_atom_count": len(missing),
            "missing_context_evidence_atoms_sample": missing[:20],
            "unknown_evidence_atom_count": len(unknown),
            "unknown_evidence_atoms_sample": unknown[:20],
            "missing_articles": missing_articles,
            "oversized_units": oversized,
            "status": "pass" if passed else "fail",
        }


@dataclass(frozen=True)
class ContextSelection:
    units: tuple[RetrievedUnit, ...]
    context_text: str
    token_count: int
    token_budget: int
    skipped_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class _DocumentEvidence:
    article_id: str
    source_file: str
    atoms: tuple[EvidenceAtom, ...]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _locator(block_id: str) -> str:
    return block_id.removeprefix("blk-")


def _load_documents(structures_dir: Path) -> tuple[_DocumentEvidence, ...]:
    if not structures_dir.is_dir():
        raise FileNotFoundError(f"structure directory does not exist: {structures_dir}")
    paths = sorted(structures_dir.glob("*.structure.json"))
    if not paths:
        raise ValueError(f"no *.structure.json files found in {structures_dir}")

    documents: list[_DocumentEvidence] = []
    seen_atoms: set[EvidenceAtomId] = set()
    seen_articles: set[str] = set()
    for path in paths:
        payload = _read_json(path)
        # Accept both the current schema id and the legacy "kmpro" id so
        # previously persisted structure files keep loading unchanged.
        if payload.get("schema_version") not in (
            "okfolio.document-structure.v1",
            "kmpro.document-structure.v1",
        ):
            raise ValueError(f"unsupported structure schema: {path}")
        if payload.get("status") != "complete":
            raise ValueError(f"document structure is not complete: {path}")
        article_id = str(payload.get("document_id", "")).strip()
        if not article_id:
            raise ValueError(f"missing document_id: {path}")
        if article_id in seen_articles:
            raise ValueError(f"duplicate document_id: {article_id}")
        seen_articles.add(article_id)
        source_file = f"{path.name.removesuffix('.structure.json')}.md"

        atoms: list[EvidenceAtom] = []
        for raw in payload.get("blocks", []):
            if not isinstance(raw, dict) or not raw.get("evidence_eligible"):
                continue
            text = str(raw.get("content", "")).strip()
            block_id = str(raw.get("block_id", "")).strip()
            page = int(raw.get("page_number") or (int(raw.get("page_idx", -1)) + 1))
            if not text or not block_id or page < 1:
                continue
            atom_id = EvidenceAtomId(
                article_id=article_id,
                page=page,
                atom_kind="block",
                locator_id=_locator(block_id),
            )
            if atom_id in seen_atoms:
                raise ValueError(f"duplicate evidence atom: {atom_id}")
            seen_atoms.add(atom_id)
            atoms.append(
                EvidenceAtom(
                    atom_id=atom_id,
                    article_id=article_id,
                    source_file=source_file,
                    page=page,
                    block_id=block_id,
                    block_type=str(raw.get("block_type", "text")),
                    heading_path=tuple(str(item) for item in raw.get("heading_path") or ()),
                    text=text,
                    content_hash=str(raw.get("content_hash", "")),
                )
            )
        if not atoms:
            raise ValueError(f"no eligible evidence blocks: {path}")
        documents.append(_DocumentEvidence(article_id, source_file, tuple(atoms)))
    return tuple(documents)


def _render_atoms(atoms: Sequence[EvidenceAtom], *, heading: Sequence[str] = ()) -> str:
    prefix = f"{' > '.join(heading)}\n\n" if heading else ""
    return prefix + "\n\n".join(atom.text for atom in atoms)


def _heading_budget(max_chars: int, heading: Sequence[str]) -> int:
    """Reserve room for the heading prefix inside a fixed character budget."""

    prefix_chars = len(" > ".join(heading)) + 2 if heading else 0
    return max(1, max_chars - prefix_chars)


def _pack_atoms(
    atoms: Sequence[EvidenceAtom],
    *,
    max_chars: int,
) -> tuple[tuple[EvidenceAtom, ...], ...]:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    groups: list[tuple[EvidenceAtom, ...]] = []
    current: list[EvidenceAtom] = []
    current_chars = 0
    for atom in atoms:
        added = len(atom.text) + (2 if current else 0)
        if current and current_chars + added > max_chars:
            groups.append(tuple(current))
            current = []
            current_chars = 0
            added = len(atom.text)
        current.append(atom)
        current_chars += added
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def build_t0_fixed_chunks(
    structures_dir: Path,
    *,
    max_chars: int = 1_200,
) -> CorpusBuild:
    """Build a transparent fixed-character baseline without splitting atoms.

    Chunk boundaries are placed at canonical block boundaries.  A source block
    longer than ``max_chars`` is kept intact and marked ``oversized`` rather
    than silently pretending that a partial block equals the whole gold atom.
    """

    documents = _load_documents(structures_dir)
    units: list[RetrievedUnit] = []
    for document in documents:
        for ordinal, atoms in enumerate(
            _pack_atoms(document.atoms, max_chars=max_chars), start=1
        ):
            text = _render_atoms(atoms)
            atom_ids = tuple(atom.atom_id for atom in atoms)
            pages = [atom.page for atom in atoms]
            units.append(
                RetrievedUnit(
                    unit_id=f"t0-{document.article_id}-{ordinal:05d}",
                    arm="T0",
                    retrieval_text=text,
                    context_text=text,
                    retrieval_evidence_atom_ids=atom_ids,
                    context_evidence_atom_ids=atom_ids,
                    article_ids=(document.article_id,),
                    metadata={
                        "source_file": document.source_file,
                        "page_start": min(pages),
                        "page_end": max(pages),
                        "max_chars": max_chars,
                        "oversized": len(text) > max_chars,
                        "context_id": f"t0-{document.article_id}-{ordinal:05d}",
                    },
                )
            )
    atoms = tuple(atom for document in documents for atom in document.atoms)
    result = CorpusBuild("T0", tuple(units), atoms)
    if result.audit()["status"] != "pass":
        raise ValueError("T0 corpus failed provenance audit")
    return result


def _heading_groups(atoms: Sequence[EvidenceAtom]) -> tuple[tuple[EvidenceAtom, ...], ...]:
    groups: list[tuple[EvidenceAtom, ...]] = []
    current: list[EvidenceAtom] = []
    current_path: tuple[str, ...] | None = None
    for atom in atoms:
        path = atom.heading_path
        if current and path != current_path:
            groups.append(tuple(current))
            current = []
        current.append(atom)
        current_path = path
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def build_t1_parent_child(
    structures_dir: Path,
    *,
    child_max_chars: int = 600,
    parent_max_chars: int = 4_800,
) -> CorpusBuild:
    """Build heading-aware children with bounded semantic parent contexts."""

    if parent_max_chars < child_max_chars:
        raise ValueError("parent_max_chars must be >= child_max_chars")
    documents = _load_documents(structures_dir)
    units: list[RetrievedUnit] = []
    for document in documents:
        parent_ordinal = 0
        for section_atoms in _heading_groups(document.atoms):
            section_heading = section_atoms[0].heading_path
            for parent_atoms in _pack_atoms(
                section_atoms,
                max_chars=_heading_budget(parent_max_chars, section_heading),
            ):
                parent_ordinal += 1
                heading = parent_atoms[0].heading_path
                parent_text = _render_atoms(parent_atoms, heading=heading)
                parent_atom_ids = tuple(atom.atom_id for atom in parent_atoms)
                parent_id = f"t1-parent-{document.article_id}-{parent_ordinal:05d}"
                for child_ordinal, child_atoms in enumerate(
                    _pack_atoms(
                        parent_atoms,
                        max_chars=_heading_budget(child_max_chars, heading),
                    ),
                    start=1,
                ):
                    child_text = _render_atoms(child_atoms, heading=heading)
                    child_ids = tuple(atom.atom_id for atom in child_atoms)
                    pages = [atom.page for atom in child_atoms]
                    units.append(
                        RetrievedUnit(
                            unit_id=f"{parent_id}-child-{child_ordinal:03d}",
                            arm="T1",
                            retrieval_text=child_text,
                            context_text=parent_text,
                            retrieval_evidence_atom_ids=child_ids,
                            context_evidence_atom_ids=parent_atom_ids,
                            article_ids=(document.article_id,),
                            metadata={
                                "source_file": document.source_file,
                                "heading_path": list(heading),
                                "page_start": min(pages),
                                "page_end": max(pages),
                                "child_max_chars": child_max_chars,
                                "parent_max_chars": parent_max_chars,
                                "oversized": len(child_text) > child_max_chars,
                                "context_oversized": len(parent_text) > parent_max_chars,
                                "context_id": parent_id,
                                "parent_id": parent_id,
                            },
                        )
                    )
    atoms = tuple(atom for document in documents for atom in document.atoms)
    result = CorpusBuild("T1", tuple(units), atoms)
    if result.audit()["status"] != "pass":
        raise ValueError("T1 corpus failed provenance audit")
    return result


def _by_block_id(
    documents: Sequence[_DocumentEvidence],
) -> dict[tuple[str, str], EvidenceAtom]:
    return {
        (atom.article_id, atom.block_id): atom
        for document in documents
        for atom in document.atoms
    }


def build_c1_audited_concepts(
    *,
    run_dir: Path,
    structures_dir: Path,
) -> CorpusBuild:
    """Import only a completed and independently accepted AgentWiki run."""

    manifest = _read_json(run_dir / "manifest.json")
    acceptance = _read_json(run_dir / "acceptance.json")
    if manifest.get("status") != "complete":
        raise ValueError("C1 requires a complete AgentWiki run")
    if acceptance.get("status") != "pass":
        raise ValueError("C1 requires acceptance.json with status=pass")

    refs_payload = _read_json(run_dir / "refs.json")
    concepts_payload = _read_json(run_dir / "concepts.json")
    refs = refs_payload.get("refs", [])
    concepts = concepts_payload.get("concepts", [])
    if not isinstance(refs, list) or not isinstance(concepts, list):
        raise ValueError("invalid refs.json or concepts.json")
    by_ref = {str(item.get("ref_id")): item for item in refs if isinstance(item, dict)}
    if len(by_ref) != len(refs):
        raise ValueError("C1 refs must have unique ref_id values")

    documents = _load_documents(structures_dir)
    block_catalog = _by_block_id(documents)
    units: list[RetrievedUnit] = []
    assigned_refs: set[str] = set()
    for item in concepts:
        if not isinstance(item, dict) or item.get("status") != "publishable":
            raise ValueError("C1 cannot import non-publishable concepts")
        concept_id = str(item.get("group_id", "")).strip()
        ref_ids = tuple(str(value) for value in item.get("ref_ids") or ())
        if not concept_id or not ref_ids:
            raise ValueError("C1 concept must contain group_id and ref_ids")
        if assigned_refs.intersection(ref_ids):
            raise ValueError("ConceptRef cannot be assigned to multiple Concepts")
        assigned_refs.update(ref_ids)
        unknown_refs = set(ref_ids) - set(by_ref)
        if unknown_refs:
            raise ValueError(f"unknown ConceptRef: {sorted(unknown_refs)[0]}")

        markdown_path = run_dir / "concepts" / f"{concept_id}.md"
        document = parse_concept_markdown(
            markdown_path.name,
            markdown_path.read_text(encoding="utf-8"),
        )
        if tuple(document.frontmatter.get("concept_refs") or ()) != ref_ids:
            raise ValueError(f"ConceptRef mismatch in {markdown_path.name}")

        atom_ids: list[EvidenceAtomId] = []
        article_ids: set[str] = set()
        source_files: set[str] = set()
        for ref_id in ref_ids:
            ref = by_ref[ref_id]
            article_id = str(ref.get("article_id", "")).strip()
            source_file = str(ref.get("source", "")).strip()
            block_ids = tuple(str(value) for value in ref.get("evidence_block_ids") or ())
            if not article_id or not block_ids:
                raise ValueError(f"ConceptRef lacks block provenance: {ref_id}")
            article_ids.add(article_id)
            if source_file:
                source_files.add(Path(source_file).name)
            for block_id in block_ids:
                atom = block_catalog.get((article_id, block_id))
                if atom is None:
                    raise ValueError(
                        f"ConceptRef points to unknown or ineligible block: {ref_id}/{block_id}"
                    )
                atom_ids.append(atom.atom_id)
        unique_atom_ids = tuple(dict.fromkeys(atom_ids))
        title = str(document.frontmatter.get("title", item.get("title", ""))).strip()
        description = str(
            document.frontmatter.get("description", item.get("description", ""))
        ).strip()
        text = "\n\n".join(value for value in (title, description, document.body.strip()) if value)
        if not text:
            raise ValueError(f"empty Concept content: {concept_id}")
        units.append(
            RetrievedUnit(
                unit_id=f"c1-{concept_id}",
                arm="C1",
                retrieval_text=text,
                context_text=text,
                retrieval_evidence_atom_ids=unique_atom_ids,
                context_evidence_atom_ids=unique_atom_ids,
                article_ids=tuple(sorted(article_ids)),
                metadata={
                    "concept_id": concept_id,
                    "concept_ref_ids": list(ref_ids),
                    "source_files": sorted(source_files),
                    "quality_score": document.frontmatter.get("agent_quality_score"),
                    "context_id": f"c1-{concept_id}",
                },
            )
        )

    if assigned_refs != set(by_ref):
        missing = sorted(set(by_ref) - assigned_refs)
        raise ValueError(f"C1 does not cover every accepted ConceptRef: {missing[0]}")
    atoms = tuple(atom for document in documents for atom in document.atoms)
    result = CorpusBuild("C1", tuple(units), atoms)
    audit = result.audit()
    if audit["status"] != "pass":
        raise ValueError("C1 corpus failed provenance or article-coverage audit")
    return result


def select_context_by_token_budget(
    ranked_units: Iterable[RetrievedUnit],
    *,
    token_budget: int,
    count_tokens: TokenCounter,
    separator: str = "\n\n---\n\n",
) -> ContextSelection:
    """Greedily select ranked contexts while enforcing the actual token budget.

    Token counts are recomputed on the complete joined prompt context.  This is
    slightly more expensive than adding per-unit counts, but remains correct
    for tokenizers whose boundary merges make token counts non-additive.
    Repeated Parent-Child hits are deduplicated by ``metadata.context_id``.
    """

    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    selected: list[RetrievedUnit] = []
    texts: list[str] = []
    skipped: list[str] = []
    seen_contexts: set[str] = set()
    token_count = 0
    for unit in ranked_units:
        context_id = str(unit.metadata.get("context_id") or _stable_id(
            "context", unit.context_text, *(str(atom) for atom in unit.context_evidence_atom_ids)
        ))
        if context_id in seen_contexts:
            continue
        seen_contexts.add(context_id)
        candidate_text = separator.join((*texts, unit.context_text))
        candidate_tokens = int(count_tokens(candidate_text))
        if candidate_tokens < 0:
            raise ValueError("count_tokens returned a negative value")
        if candidate_tokens > token_budget:
            skipped.append(unit.unit_id)
            continue
        selected.append(unit)
        texts.append(unit.context_text)
        token_count = candidate_tokens
    return ContextSelection(
        units=tuple(selected),
        context_text=separator.join(texts),
        token_count=token_count,
        token_budget=token_budget,
        skipped_unit_ids=tuple(skipped),
    )
