"""Checkpointed, provider-neutral runner for the three-arm RAG experiment.

The runner owns experiment invariants and durable artifacts.  Search and text
generation stay behind :class:`ExperimentBackend`, so this module never loads a
model, contacts an endpoint, or imports an optional retrieval framework.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from .contracts import (
    AnswerPrediction,
    Citation,
    EvidenceAtomId,
    GoldQuestion,
    RetrievedUnit,
)
from .corpus import (
    Arm,
    CorpusBuild,
    EvidenceAtom,
    build_c1_audited_concepts,
    build_t0_fixed_chunks,
    build_t1_parent_child,
)
from .generation import (
    AnswerContext,
    AnswerGenerationInput,
    GenerationResult,
    parse_structured_answer,
    structured_answer_prediction,
)
from .gold import load_gold_jsonl
from .retrieval import (
    BackendConfig,
    HybridRetrievalHarness,
    RetrievalConfig,
    Retriever,
    Reranker,
    ThreeArmRetrievalHarness,
)
from .scoring import (
    joint_success,
    mean_defined,
    score_answer,
    score_provisional_answer,
    score_retrieval,
)
from .stats import paired_bootstrap_delta


Stage = Literal["index", "retrieve", "generate", "score"]
HydeMode = Literal["off", "ablation"]
SemanticAlignmentStatus = Literal[
    "provisional_structured", "human_reviewed", "independent_judge"
]
ARMS: tuple[Arm, ...] = ("T0", "T1", "C1")


class ExperimentStateError(RuntimeError):
    """Raised when a run cannot be resumed without violating its lock."""


@dataclass(frozen=True)
class CorpusConfig:
    t0_max_chars: int = 1_200
    t1_child_max_chars: int = 600
    t1_parent_max_chars: int = 4_800

    def __post_init__(self) -> None:
        if min(self.t0_max_chars, self.t1_child_max_chars) < 1:
            raise ValueError("corpus character budgets must be positive")
        if self.t1_parent_max_chars < self.t1_child_max_chars:
            raise ValueError("T1 parent budget must be >= child budget")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    structures_dir: Path
    c1_run_dir: Path
    gold_path: Path
    retrieval: RetrievalConfig
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    hyde_mode: HydeMode = "off"
    bootstrap_samples: int = 10_000
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 0
    adapter_factory: str = ""
    adapter_options: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "okfolio.rag-experiment-config.v1"

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if self.hyde_mode not in {"off", "ablation"}:
            raise ValueError("HyDE must be 'off' or the explicit 'ablation' mode")
        if self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if not 0 < self.bootstrap_confidence < 1:
            raise ValueError("bootstrap_confidence must be between zero and one")
        json.dumps(self.adapter_options, ensure_ascii=False, sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "experiment_id": self.experiment_id,
            "structures_dir": str(self.structures_dir.resolve()),
            "c1_run_dir": str(self.c1_run_dir.resolve()),
            "gold_path": str(self.gold_path.resolve()),
            "corpus": asdict(self.corpus),
            "retrieval": self.retrieval.to_dict(),
            "hyde_mode": self.hyde_mode,
            "bootstrap": {
                "samples": self.bootstrap_samples,
                "confidence": self.bootstrap_confidence,
                "seed": self.bootstrap_seed,
            },
            "adapter": {
                "factory": self.adapter_factory,
                "options": dict(self.adapter_options),
            },
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class GeneratedAnswer:
    """Provider output plus deterministic fact/citation alignment.

    A production adapter may obtain ``prediction`` from human labels or a
    separately calibrated structured judge.  The runner never infers facts
    from prose with hidden heuristics.
    """

    text: str
    prediction: AnswerPrediction
    semantic_alignment_status: SemanticAlignmentStatus
    usage: Mapping[str, int | None] = field(default_factory=dict)
    timing: Mapping[str, float | None] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("generated answer text must be non-empty")
        if self.semantic_alignment_status not in {
            "provisional_structured",
            "human_reviewed",
            "independent_judge",
        }:
            raise ValueError("unknown semantic_alignment_status")


class ExperimentBackend(Protocol):
    """Minimal plugin surface; implementations own all provider dependencies."""

    def count_tokens(self, text: str) -> int: ...

    def prepare_index(self, corpus: CorpusBuild, index_dir: Path) -> None: ...

    def create_bm25(self, corpus: CorpusBuild, index_dir: Path) -> Retriever: ...

    def create_dense(self, corpus: CorpusBuild, index_dir: Path) -> Retriever: ...

    def create_reranker(self, corpus: CorpusBuild) -> Reranker: ...

    def generate(
        self,
        *,
        gold: GoldQuestion,
        arm: Arm,
        request: AnswerGenerationInput,
    ) -> GeneratedAnswer: ...

    def generate_hyde(self, query: str) -> str: ...


class AnswerTextGenerator(Protocol):
    """The provider-neutral subset exposed by generation.py clients."""

    def generate_answer(
        self, request: AnswerGenerationInput, *, stream: bool = True
    ) -> GenerationResult: ...


class PredictionAligner(Protocol):
    """Explicitly align generated prose to reviewed fact/evidence labels."""

    def align(
        self,
        *,
        gold: GoldQuestion,
        arm: Arm,
        request: AnswerGenerationInput,
        result: GenerationResult,
    ) -> AnswerPrediction: ...


@dataclass(frozen=True)
class AlignedGenerationPipeline:
    """Compose ``generation.py`` with a visible, replaceable alignment step."""

    generator: AnswerTextGenerator
    aligner: PredictionAligner
    semantic_alignment_status: Literal["human_reviewed", "independent_judge"]
    stream: bool = True

    def generate(
        self,
        *,
        gold: GoldQuestion,
        arm: Arm,
        request: AnswerGenerationInput,
    ) -> GeneratedAnswer:
        result = self.generator.generate_answer(request, stream=self.stream)
        structured = parse_structured_answer(result.text, request)
        contract_prediction = structured_answer_prediction(
            gold.question_id, structured
        )
        aligned_facts = self.aligner.align(
            gold=gold,
            arm=arm,
            request=request,
            result=replace(result, text=structured.answer),
        )
        if aligned_facts.question_id != gold.question_id:
            raise ValueError("prediction aligner returned the wrong question_id")
        prediction = AnswerPrediction(
            question_id=gold.question_id,
            predicted_answerable=contract_prediction.predicted_answerable,
            matched_required_fact_ids=aligned_facts.matched_required_fact_ids,
            asserted_forbidden_fact_ids=aligned_facts.asserted_forbidden_fact_ids,
            citations=contract_prediction.citations,
        )
        return GeneratedAnswer(
            text=structured.answer,
            prediction=prediction,
            semantic_alignment_status=self.semantic_alignment_status,
            usage=asdict(result.usage),
            timing=asdict(result.timing),
            metadata={
                "finish_reason": result.finish_reason,
                "stream_events": result.stream_events,
                "alignment": "explicit-plugin",
                "structured_answer": structured.to_dict(),
            },
        )


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical(payload) + b"\n")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical(payload) + b"\n"
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.exists():
        return ()
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ExperimentStateError(f"{path}:{number}: trace row must be an object")
        rows.append(value)
    return tuple(rows)


def _unique_rows(path: Path, key_fields: Sequence[str]) -> dict[tuple[str, ...], dict[str, Any]]:
    indexed: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if not all(key):
            raise ExperimentStateError(f"{path}: trace row lacks {key_fields}")
        if key in indexed:
            raise ExperimentStateError(f"{path}: duplicate trace key {key}")
        indexed[key] = row
    return indexed


def _serialize_unit(unit: RetrievedUnit) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "arm": unit.arm,
        "retrieval_text": unit.retrieval_text,
        "context_text": unit.context_text,
        "retrieval_evidence_atom_ids": [str(item) for item in unit.retrieval_evidence_atom_ids],
        "context_evidence_atom_ids": [str(item) for item in unit.context_evidence_atom_ids],
        "article_ids": list(unit.article_ids),
        "score": unit.score,
        "metadata": dict(unit.metadata),
    }


def _deserialize_unit(payload: Mapping[str, Any]) -> RetrievedUnit:
    return RetrievedUnit(
        unit_id=str(payload["unit_id"]),
        arm=str(payload["arm"]),
        retrieval_text=str(payload["retrieval_text"]),
        context_text=str(payload["context_text"]),
        retrieval_evidence_atom_ids=tuple(
            EvidenceAtomId.parse(str(value))
            for value in payload["retrieval_evidence_atom_ids"]
        ),
        context_evidence_atom_ids=tuple(
            EvidenceAtomId.parse(str(value))
            for value in payload["context_evidence_atom_ids"]
        ),
        article_ids=tuple(str(value) for value in payload["article_ids"]),
        score=(float(payload["score"]) if payload.get("score") is not None else None),
        metadata=dict(payload.get("metadata") or {}),
    )


def _serialize_corpus(corpus: CorpusBuild) -> dict[str, Any]:
    return {
        "schema": "okfolio.rag-corpus-snapshot.v1",
        "arm": corpus.arm,
        "audit": corpus.audit(),
        "units": [_serialize_unit(unit) for unit in corpus.units],
        "evidence_atoms": [atom.to_dict() for atom in corpus.evidence_atoms],
    }


def _deserialize_corpus(payload: Mapping[str, Any]) -> CorpusBuild:
    atoms = tuple(
        EvidenceAtom(
            atom_id=EvidenceAtomId.parse(str(item["atom_id"])),
            article_id=str(item["article_id"]),
            source_file=str(item["source_file"]),
            page=int(item["page"]),
            block_id=str(item["block_id"]),
            block_type=str(item["block_type"]),
            heading_path=tuple(str(value) for value in item.get("heading_path") or ()),
            text=str(item["text"]),
            content_hash=str(item.get("content_hash", "")),
        )
        for item in payload["evidence_atoms"]
    )
    corpus = CorpusBuild(
        arm=cast(Arm, str(payload["arm"])),
        units=tuple(_deserialize_unit(item) for item in payload["units"]),
        evidence_atoms=atoms,
    )
    if corpus.audit()["status"] != "pass":
        raise ExperimentStateError(f"stored {corpus.arm} corpus failed audit")
    return corpus


def _prediction_to_dict(prediction: AnswerPrediction) -> dict[str, Any]:
    return {
        "question_id": prediction.question_id,
        "predicted_answerable": prediction.predicted_answerable,
        "matched_required_fact_ids": list(prediction.matched_required_fact_ids),
        "asserted_forbidden_fact_ids": list(prediction.asserted_forbidden_fact_ids),
        "citations": [str(item.evidence_atom_id) for item in prediction.citations],
    }


def _prediction_from_dict(payload: Mapping[str, Any]) -> AnswerPrediction:
    return AnswerPrediction(
        question_id=str(payload["question_id"]),
        predicted_answerable=bool(payload["predicted_answerable"]),
        matched_required_fact_ids=tuple(str(value) for value in payload.get("matched_required_fact_ids") or ()),
        asserted_forbidden_fact_ids=tuple(str(value) for value in payload.get("asserted_forbidden_fact_ids") or ()),
        citations=tuple(
            Citation(EvidenceAtomId.parse(str(value)))
            for value in payload.get("citations") or ()
        ),
    )


class ThreeArmExperimentRunner:
    """Run index, retrieve, generate and score as idempotent stages."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        output_dir: Path,
        backend: ExperimentBackend | None = None,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.backend = backend
        self.lock_path = output_dir / "experiment.lock.json"
        self.manifest_path = output_dir / "manifest.json"
        self.corpora_dir = output_dir / "corpora"
        self.index_dir = output_dir / "indices"
        self.traces_dir = output_dir / "traces"

    def _input_lock(self) -> dict[str, Any]:
        structures = sorted(self.config.structures_dir.glob("*.structure.json"))
        if not structures:
            raise ExperimentStateError(
                f"no structure sidecars in {self.config.structures_dir}"
            )
        c1_manifest_path = self.config.c1_run_dir / "manifest.json"
        c1_acceptance_path = self.config.c1_run_dir / "acceptance.json"
        if not c1_manifest_path.is_file():
            raise ExperimentStateError(
                f"C1 is not ready: missing {c1_manifest_path}"
            )
        manifest = _read_json(c1_manifest_path)
        if manifest.get("status") != "complete":
            raise ExperimentStateError(
                "C1 is not ready: AgentWiki manifest status is "
                f"{manifest.get('status')!r}; expected 'complete'"
            )
        if not c1_acceptance_path.is_file():
            raise ExperimentStateError(
                f"C1 is not ready: missing {c1_acceptance_path}"
            )
        acceptance = _read_json(c1_acceptance_path)
        if acceptance.get("status") != "pass":
            raise ExperimentStateError(
                "C1 is not ready: acceptance status is "
                f"{acceptance.get('status')!r}; expected 'pass'"
            )
        c1_files = [
            c1_manifest_path,
            c1_acceptance_path,
            self.config.c1_run_dir / "refs.json",
            self.config.c1_run_dir / "concepts.json",
            *sorted((self.config.c1_run_dir / "concepts").glob("*.md")),
        ]
        for path in (*structures, *c1_files, self.config.gold_path):
            if not path.is_file():
                raise ExperimentStateError(
                    f"experiment input is not ready: missing {path}"
                )
        return {
            "schema": "okfolio.rag-experiment-lock.v1",
            "config": self.config.to_dict(),
            "config_fingerprint": self.config.fingerprint,
            "inputs": {
                "gold": _file_sha256(self.config.gold_path),
                "structures": {
                    path.name: _file_sha256(path) for path in structures
                },
                "c1": {
                    str(path.relative_to(self.config.c1_run_dir)): _file_sha256(path)
                    for path in c1_files
                },
            },
        }

    def readiness(self, *, build_corpora: bool = True) -> dict[str, Any]:
        lock = self._input_lock()
        gold = load_gold_jsonl(self.config.gold_path)
        report: dict[str, Any] = {
            "schema": "okfolio.rag-experiment-readiness.v1",
            "status": "ready",
            "experiment_id": self.config.experiment_id,
            "config_fingerprint": self.config.fingerprint,
            "input_fingerprint": _fingerprint(lock["inputs"]),
            "questions": len(gold),
            "hyde_mode": self.config.hyde_mode,
            "hyde_policy": "disabled-by-default; ablation-only",
            "writes_performed": False,
            "backend_loaded": self.backend is not None,
        }
        if build_corpora:
            corpora = self._build_corpora()
            report["corpora"] = {
                arm: corpus.audit() for arm, corpus in corpora.items()
            }
        return report

    def _build_corpora(self) -> dict[Arm, CorpusBuild]:
        corpus_config = self.config.corpus
        return {
            "T0": build_t0_fixed_chunks(
                self.config.structures_dir, max_chars=corpus_config.t0_max_chars
            ),
            "T1": build_t1_parent_child(
                self.config.structures_dir,
                child_max_chars=corpus_config.t1_child_max_chars,
                parent_max_chars=corpus_config.t1_parent_max_chars,
            ),
            "C1": build_c1_audited_concepts(
                run_dir=self.config.c1_run_dir,
                structures_dir=self.config.structures_dir,
            ),
        }

    def _lock(self) -> dict[str, Any]:
        candidate = self._input_lock()
        candidate["lock_fingerprint"] = _fingerprint(candidate)
        if self.lock_path.exists():
            current = _read_json(self.lock_path)
            if current != candidate:
                raise ExperimentStateError(
                    "experiment lock changed; use a new output directory"
                )
        else:
            _atomic_json(self.lock_path, candidate)
        if not self.manifest_path.exists():
            _atomic_json(
                self.manifest_path,
                {
                    "schema": "okfolio.rag-experiment-manifest.v1",
                    "experiment_id": self.config.experiment_id,
                    "lock_fingerprint": candidate["lock_fingerprint"],
                    "stages": {stage: {"status": "pending"} for stage in ("index", "retrieve", "generate", "score")},
                },
            )
        return candidate

    def _set_stage(self, stage: Stage, status: str, **details: Any) -> None:
        manifest = _read_json(self.manifest_path)
        manifest["stages"][stage] = {"status": status, **details}
        _atomic_json(self.manifest_path, manifest)

    def _require_backend(self) -> ExperimentBackend:
        if self.backend is None:
            raise ExperimentStateError(
                "this stage requires an adapter factory; readiness does not"
            )
        return self.backend

    def _require_stage(self, stage: Stage) -> None:
        manifest = _read_json(self.manifest_path)
        if manifest["stages"][stage]["status"] != "complete":
            raise ExperimentStateError(f"required stage is not complete: {stage}")

    def _load_corpora(self) -> dict[Arm, CorpusBuild]:
        corpora: dict[Arm, CorpusBuild] = {}
        for arm in ARMS:
            path = self.corpora_dir / f"{arm}.json"
            if not path.is_file():
                raise ExperimentStateError(f"missing corpus snapshot: {path}")
            corpora[arm] = _deserialize_corpus(_read_json(path))
        return corpora

    def index(self) -> dict[str, Any]:
        backend = self._require_backend()
        lock = self._lock()
        if _read_json(self.manifest_path)["stages"]["index"]["status"] == "complete":
            return _read_json(self.manifest_path)["stages"]["index"]
        self._set_stage("index", "running")
        corpora = self._build_corpora()
        audits: dict[str, Any] = {}
        progress_path = self.index_dir / "progress.json"
        progress = (
            _read_json(progress_path)
            if progress_path.exists()
            else {
                "schema": "okfolio.rag-index-progress.v1",
                "lock_fingerprint": lock["lock_fingerprint"],
                "arms": {},
            }
        )
        if progress.get("lock_fingerprint") != lock["lock_fingerprint"]:
            raise ExperimentStateError("index progress belongs to another lock")
        for arm in ARMS:
            snapshot = _serialize_corpus(corpora[arm])
            snapshot["lock_fingerprint"] = lock["lock_fingerprint"]
            snapshot_fingerprint = _fingerprint(snapshot)
            snapshot_path = self.corpora_dir / f"{arm}.json"
            arm_progress = progress["arms"].get(arm) or {}
            if (
                arm_progress.get("status") == "complete"
                and arm_progress.get("corpus_fingerprint") == snapshot_fingerprint
                and snapshot_path.is_file()
                and _fingerprint(_read_json(snapshot_path)) == snapshot_fingerprint
            ):
                audits[arm] = corpora[arm].audit()
                continue
            _atomic_json(snapshot_path, snapshot)
            arm_index = self.index_dir / arm
            arm_index.mkdir(parents=True, exist_ok=True)
            backend.prepare_index(corpora[arm], arm_index)
            audits[arm] = corpora[arm].audit()
            progress["arms"][arm] = {
                "status": "complete",
                "corpus_fingerprint": snapshot_fingerprint,
                "audit": audits[arm],
            }
            _atomic_json(progress_path, progress)
        self._set_stage("index", "complete", corpora=audits)
        return _read_json(self.manifest_path)["stages"]["index"]

    def _harness(self, corpora: Mapping[Arm, CorpusBuild]) -> ThreeArmRetrievalHarness:
        backend = self._require_backend()
        arms: dict[Arm, HybridRetrievalHarness] = {}
        for arm in ARMS:
            corpus = corpora[arm]
            index_dir = self.index_dir / arm
            arms[arm] = HybridRetrievalHarness(
                corpus=corpus,
                bm25=backend.create_bm25(corpus, index_dir),
                dense=backend.create_dense(corpus, index_dir),
                reranker=backend.create_reranker(corpus),
                config=self.config.retrieval,
                count_tokens=backend.count_tokens,
            )
        return ThreeArmRetrievalHarness(arms)

    def retrieve(self) -> dict[str, Any]:
        self._lock()
        self._require_stage("index")
        backend = self._require_backend()
        corpora = self._load_corpora()
        harness = self._harness(corpora)
        questions = load_gold_jsonl(self.config.gold_path)
        path = self.traces_dir / "retrieval.jsonl"
        completed = _unique_rows(path, ("question_id", "arm"))
        self._set_stage("retrieve", "running", records=len(completed))
        for gold in questions:
            missing = [arm for arm in ARMS if (gold.question_id, arm) not in completed]
            if not missing:
                continue
            retrieval_query = gold.question
            hyde_text: str | None = None
            if self.config.hyde_mode == "ablation":
                hyde_text = backend.generate_hyde(gold.question).strip()
                if not hyde_text:
                    raise ValueError("HyDE adapter returned empty expansion")
                retrieval_query = f"{gold.question}\n\n{hyde_text}"
            result = harness.retrieve(
                question_id=gold.question_id, query=retrieval_query
            )
            for arm in missing:
                payload = result.arms[arm].to_dict()
                payload.update(
                    {
                        "experiment_id": self.config.experiment_id,
                        "question": gold.question,
                        "original_query": gold.question,
                        "retrieval_query": retrieval_query,
                        "hyde_mode": self.config.hyde_mode,
                        "hyde_text": hyde_text,
                    }
                )
                _append_jsonl(path, payload)
                completed[(gold.question_id, arm)] = payload
        expected = len(questions) * len(ARMS)
        if len(completed) != expected:
            raise ExperimentStateError("retrieval trace is incomplete")
        self._set_stage("retrieve", "complete", records=expected)
        return _read_json(self.manifest_path)["stages"]["retrieve"]

    @staticmethod
    def _selected_contexts(
        trace: Mapping[str, Any], catalog: Mapping[str, RetrievedUnit]
    ) -> tuple[AnswerContext, ...]:
        contexts: list[AnswerContext] = []
        for item in trace.get("selected_contexts") or ():
            unit_id = str(item["unit_id"])
            unit = catalog.get(unit_id)
            if unit is None:
                raise ExperimentStateError(f"trace references unknown unit: {unit_id}")
            pages = tuple(sorted({atom.page for atom in unit.context_evidence_atom_ids}))
            contexts.append(
                AnswerContext(
                    context_id=str(unit.metadata.get("context_id") or unit.unit_id),
                    text=unit.context_text,
                    title=str(unit.metadata.get("concept_id") or ""),
                    source_id=",".join(unit.article_ids),
                    page_numbers=pages,
                    evidence_ids=tuple(str(atom) for atom in unit.context_evidence_atom_ids),
                )
            )
        if not contexts:
            raise ExperimentStateError("retrieval selected no generation context")
        return tuple(contexts)

    def generate(self) -> dict[str, Any]:
        self._lock()
        self._require_stage("retrieve")
        backend = self._require_backend()
        corpora = self._load_corpora()
        catalogs = {
            arm: {unit.unit_id: unit for unit in corpora[arm].units} for arm in ARMS
        }
        questions = {item.question_id: item for item in load_gold_jsonl(self.config.gold_path)}
        retrieval = _unique_rows(self.traces_dir / "retrieval.jsonl", ("question_id", "arm"))
        path = self.traces_dir / "generation.jsonl"
        completed = _unique_rows(path, ("question_id", "arm"))
        self._set_stage("generate", "running", records=len(completed))
        for key in sorted(retrieval):
            if key in completed:
                continue
            question_id, arm_value = key
            arm = cast(Arm, arm_value)
            gold = questions[question_id]
            contexts = self._selected_contexts(retrieval[key], catalogs[arm])
            request = AnswerGenerationInput(question=gold.question, contexts=contexts)
            result = backend.generate(gold=gold, arm=arm, request=request)
            if result.prediction.question_id != question_id:
                raise ValueError("generator prediction question_id mismatch")
            payload = {
                "schema": "okfolio.rag-generation-trace.v1",
                "experiment_id": self.config.experiment_id,
                "question_id": question_id,
                "arm": arm,
                "context_ids": [item.context_id for item in contexts],
                "answer": result.text,
                "prediction": _prediction_to_dict(result.prediction),
                "semantic_alignment_status": result.semantic_alignment_status,
                "usage": dict(result.usage),
                "timing": dict(result.timing),
                "metadata": dict(result.metadata),
            }
            _append_jsonl(path, payload)
            completed[key] = payload
        if set(completed) != set(retrieval):
            raise ExperimentStateError("generation trace is incomplete")
        self._set_stage("generate", "complete", records=len(completed))
        return _read_json(self.manifest_path)["stages"]["generate"]

    def score(self) -> dict[str, Any]:
        self._lock()
        self._require_stage("generate")
        corpora = self._load_corpora()
        catalogs = {
            arm: {unit.unit_id: unit for unit in corpora[arm].units} for arm in ARMS
        }
        universes = {
            arm: tuple(atom.atom_id for atom in corpora[arm].evidence_atoms) for arm in ARMS
        }
        questions = {item.question_id: item for item in load_gold_jsonl(self.config.gold_path)}
        retrieval = _unique_rows(self.traces_dir / "retrieval.jsonl", ("question_id", "arm"))
        generations = _unique_rows(self.traces_dir / "generation.jsonl", ("question_id", "arm"))
        if set(retrieval) != set(generations):
            raise ExperimentStateError("retrieval and generation trace keys differ")
        path = self.traces_dir / "scores.jsonl"
        completed = _unique_rows(path, ("question_id", "arm"))
        self._set_stage("score", "running", records=len(completed))
        for key in sorted(retrieval):
            if key in completed:
                continue
            question_id, arm_value = key
            arm = cast(Arm, arm_value)
            trace = retrieval[key]
            ranked: list[RetrievedUnit] = []
            for item in trace.get("ranked_units") or ():
                unit = catalogs[arm].get(str(item["unit_id"]))
                if unit is None:
                    raise ExperimentStateError("retrieval trace references unknown unit")
                ranked.append(replace(unit, score=float(item["score"])))
            gold = questions[question_id]
            retrieval_score = score_retrieval(
                gold,
                ranked,
                k=self.config.retrieval.rerank_top_k,
                provenance_view="context",
            )
            prediction = _prediction_from_dict(generations[key]["prediction"])
            generation_metadata = generations[key].get("metadata") or {}
            if not isinstance(generation_metadata, Mapping):
                raise ExperimentStateError("generation metadata must be an object")
            alignment_value = generations[key].get("semantic_alignment_status")
            if not isinstance(alignment_value, str) or not alignment_value:
                raise ExperimentStateError(
                    "generation trace lacks explicit semantic_alignment_status"
                )
            alignment_status = alignment_value
            if alignment_status in {"human_reviewed", "independent_judge"}:
                answer_score = score_answer(
                    gold,
                    prediction,
                    evidence_universe=universes[arm],
                    semantic_alignment_status=alignment_status,
                )
            elif alignment_status in {
                "provisional_structured",
            }:
                answer_score = score_provisional_answer(
                    gold,
                    prediction,
                    evidence_universe=universes[arm],
                    semantic_alignment_status=alignment_status,
                )
            else:
                raise ExperimentStateError(
                    "unknown semantic alignment status in generation trace: "
                    f"{alignment_status!r}"
                )
            end_to_end_success = joint_success(retrieval_score, answer_score)
            payload = {
                "schema": "okfolio.rag-score-trace.v1",
                "experiment_id": self.config.experiment_id,
                "question_id": question_id,
                "arm": arm,
                "retrieval": asdict(retrieval_score),
                "answer": asdict(answer_score),
                "joint_success": end_to_end_success,
            }
            _append_jsonl(path, payload)
            completed[key] = payload
        summary = self._summarize(completed)
        _atomic_json(self.output_dir / "summary.json", summary)
        self._set_stage("score", "complete", records=len(completed), summary="summary.json")
        return summary

    def _summarize(
        self, scores: Mapping[tuple[str, ...], Mapping[str, Any]]
    ) -> dict[str, Any]:
        by_arm: dict[Arm, list[Mapping[str, Any]]] = {arm: [] for arm in ARMS}
        for (_, arm_value), row in scores.items():
            by_arm[cast(Arm, arm_value)].append(row)
        metrics: dict[str, Any] = {}
        paired_values: dict[str, dict[Arm, dict[str, float]]] = {
            name: {arm: {} for arm in ARMS}
            for name in ("evidence_recall", "answer_correct", "joint_success")
        }
        for arm in ARMS:
            rows = by_arm[arm]
            for row in rows:
                retrieval = row["retrieval"]
                answer = row["answer"]
                question_id = str(row["question_id"])
                if retrieval["evidence_recall"] is not None:
                    paired_values["evidence_recall"][arm][question_id] = float(retrieval["evidence_recall"])
                if answer["answer_correct"] is not None:
                    paired_values["answer_correct"][arm][question_id] = float(
                        bool(answer["answer_correct"])
                    )
                if row["joint_success"] is not None:
                    paired_values["joint_success"][arm][question_id] = float(
                        bool(row["joint_success"])
                    )
            # Refusal depends only on the two answerability booleans; construct
            # its aggregate directly to avoid coupling summary to text judging.
            true_refusals = sum(not r["answer"]["gold_answerable"] and not r["answer"]["predicted_answerable"] for r in rows)
            false_refusals = sum(r["answer"]["gold_answerable"] and not r["answer"]["predicted_answerable"] for r in rows)
            missed_refusals = sum(not r["answer"]["gold_answerable"] and r["answer"]["predicted_answerable"] for r in rows)
            answerable = sum(bool(r["answer"]["gold_answerable"]) for r in rows)
            unanswerable = len(rows) - answerable
            refusal_precision = true_refusals / (true_refusals + false_refusals) if true_refusals + false_refusals else 0.0
            refusal_recall = true_refusals / (true_refusals + missed_refusals) if true_refusals + missed_refusals else 0.0
            refusal_f1 = 2 * refusal_precision * refusal_recall / (refusal_precision + refusal_recall) if refusal_precision + refusal_recall else 0.0
            alignment_counts: dict[str, int] = {}
            for row in rows:
                status = str(row["answer"]["semantic_alignment_status"])
                alignment_counts[status] = alignment_counts.get(status, 0) + 1
            semantic_rows = sum(
                row["answer"]["answer_correct"] is not None for row in rows
            )
            metrics[arm] = {
                "questions": len(rows),
                "evidence_recall": mean_defined(r["retrieval"]["evidence_recall"] for r in rows),
                "complete_evidence_set_recall": mean_defined(r["retrieval"]["complete_evidence_set_recall"] for r in rows),
                "mrr": mean_defined(r["retrieval"]["mrr"] for r in rows),
                "ndcg": mean_defined(r["retrieval"]["ndcg"] for r in rows),
                "context_precision": mean_defined(r["retrieval"]["context_precision"] for r in rows),
                "answer_accuracy": mean_defined(
                    float(r["answer"]["answer_correct"])
                    if r["answer"]["answer_correct"] is not None
                    else None
                    for r in rows
                ),
                "joint_success_rate": mean_defined(
                    float(r["joint_success"])
                    if r["joint_success"] is not None
                    else None
                    for r in rows
                ),
                "semantic_scoring": {
                    "status": (
                        "complete" if semantic_rows == len(rows) else "provisional"
                    ),
                    "scored_rows": semantic_rows,
                    "pending_rows": len(rows) - semantic_rows,
                    "alignment_status_counts": alignment_counts,
                },
                "refusal": {
                    "precision": refusal_precision,
                    "recall": refusal_recall,
                    "f1": refusal_f1,
                    "over_refusal_rate": false_refusals / answerable if answerable else 0.0,
                    "unsupported_answer_rate": missed_refusals / unanswerable if unanswerable else 0.0,
                },
            }
        comparisons: dict[str, Any] = {}
        for metric, arm_values in paired_values.items():
            comparisons[metric] = {}
            for candidate in ("T1", "C1"):
                common = sorted(set(arm_values["T0"]) & set(arm_values[candidate]))
                baseline = {qid: arm_values["T0"][qid] for qid in common}
                proposed = {qid: arm_values[candidate][qid] for qid in common}
                if not common:
                    comparisons[metric][f"{candidate}-T0"] = None
                    continue
                comparisons[metric][f"{candidate}-T0"] = asdict(
                    paired_bootstrap_delta(
                        baseline,
                        proposed,
                        samples=self.config.bootstrap_samples,
                        confidence=self.config.bootstrap_confidence,
                        seed=self.config.bootstrap_seed,
                    )
                )
        return {
            "schema": "okfolio.rag-experiment-summary.v1",
            "experiment_id": self.config.experiment_id,
            "config_fingerprint": self.config.fingerprint,
            "hyde_mode": self.config.hyde_mode,
            "metrics": metrics,
            "paired_bootstrap": comparisons,
        }

    def run(self, stage: Literal["index", "retrieve", "generate", "score", "all"]) -> Any:
        if stage == "all":
            self.index()
            self.retrieve()
            self.generate()
            return self.score()
        return getattr(self, stage)()


def _backend_config(payload: Mapping[str, Any]) -> BackendConfig:
    return BackendConfig(
        provider=str(payload["provider"]),
        model=str(payload["model"]),
        revision=str(payload["revision"]),
        parameters=dict(payload.get("parameters") or {}),
    )


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load a JSON config, resolving data paths relative to the config file."""

    payload = _read_json(path)
    base = path.resolve().parent
    retrieval = payload["retrieval"]
    corpus = payload.get("corpus") or {}
    bootstrap = payload.get("bootstrap") or {}
    adapter = payload.get("adapter") or {}

    def resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else base / candidate

    return ExperimentConfig(
        experiment_id=str(payload["experiment_id"]),
        structures_dir=resolve(str(payload["structures_dir"])),
        c1_run_dir=resolve(str(payload["c1_run_dir"])),
        gold_path=resolve(str(payload["gold_path"])),
        retrieval=RetrievalConfig(
            bm25=_backend_config(retrieval["bm25"]),
            dense=_backend_config(retrieval["dense"]),
            reranker=_backend_config(retrieval["reranker"]),
            bm25_top_k=int(retrieval.get("bm25_top_k", 50)),
            dense_top_k=int(retrieval.get("dense_top_k", 50)),
            fusion_top_k=int(retrieval.get("fusion_top_k", 50)),
            rerank_top_k=int(retrieval.get("rerank_top_k", 20)),
            context_token_budget=int(retrieval.get("context_token_budget", 8192)),
            rrf_k=int(retrieval.get("rrf_k", 60)),
            rrf_bm25_weight=float(retrieval.get("rrf_bm25_weight", 1.0)),
            rrf_dense_weight=float(retrieval.get("rrf_dense_weight", 1.0)),
            context_separator=str(retrieval.get("context_separator", "\n\n---\n\n")),
        ),
        corpus=CorpusConfig(
            t0_max_chars=int(corpus.get("t0_max_chars", 1200)),
            t1_child_max_chars=int(corpus.get("t1_child_max_chars", 600)),
            t1_parent_max_chars=int(corpus.get("t1_parent_max_chars", 4800)),
        ),
        hyde_mode=cast(HydeMode, payload.get("hyde_mode", "off")),
        bootstrap_samples=int(bootstrap.get("samples", 10_000)),
        bootstrap_confidence=float(bootstrap.get("confidence", 0.95)),
        bootstrap_seed=int(bootstrap.get("seed", 0)),
        adapter_factory=str(adapter.get("factory", "")),
        adapter_options=dict(adapter.get("options") or {}),
    )


def load_backend(config: ExperimentConfig, output_dir: Path) -> ExperimentBackend:
    """Load an explicit ``module:function`` adapter factory."""

    reference = config.adapter_factory.strip()
    if ":" not in reference:
        raise ValueError("adapter.factory must use module:function syntax")
    module_name, function_name = reference.split(":", 1)
    factory = getattr(importlib.import_module(module_name), function_name)
    backend = factory(dict(config.adapter_options), output_dir)
    return cast(ExperimentBackend, backend)


__all__ = [
    "AlignedGenerationPipeline",
    "AnswerTextGenerator",
    "CorpusConfig",
    "ExperimentBackend",
    "ExperimentConfig",
    "ExperimentStateError",
    "GeneratedAnswer",
    "PredictionAligner",
    "SemanticAlignmentStatus",
    "ThreeArmExperimentRunner",
    "load_backend",
    "load_experiment_config",
]
