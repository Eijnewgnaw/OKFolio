"""Local BGE backend for the checkpointed three-arm RAG experiment.

The module is an adapter factory, not a model installer.  Model directories,
revisions, devices, and the OpenAI-compatible generation endpoint are supplied
through ``adapter.options`` or environment variables.  Optional heavyweight
dependencies are imported only when their stage is first used.

Document embeddings are persisted as a float32 NumPy matrix.  This keeps the
T0/T1/C1 comparison framework-neutral and avoids loading BGE-M3 merely to read
an existing index.  Query vectors are cached in memory and shared across arms.
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from string import Template
from typing import Any, Literal, Protocol, cast

from .contracts import AnswerPrediction, GoldQuestion, RetrievedUnit
from .corpus import Arm, CorpusBuild
from .experiment_runner import GeneratedAnswer, PredictionAligner
from .generation import (
    AnswerGenerationInput,
    OpenAICompatibleRAGClient,
    RAGGenerationConfig,
    parse_structured_answer,
    structured_answer_prediction,
)
from .retrieval import SearchHit
from .retrieval_adapters import BgeM3EmbeddingAdapter, BgeRerankerAdapter, BM25Retriever


class LocalBackendConfigurationError(ValueError):
    """Raised before a local backend can silently use an unfrozen setting."""


class AlignmentRequiredError(RuntimeError):
    """Legacy error retained for import compatibility.

    Generation now has a built-in contract-only path. Final semantic fact
    alignment is optional and must come from a human or independent judge.
    """


class JiebaLike(Protocol):
    def cut(
        self, sentence: str, cut_all: bool = False, HMM: bool = True
    ) -> Sequence[str]: ...


DenseLoader = Callable[[Path, Mapping[str, Any]], Any]
RerankerLoader = Callable[[Path, Mapping[str, Any]], Any]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
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


def _option(
    options: Mapping[str, Any],
    key: str,
    *,
    environ: Mapping[str, str],
    env_name: str,
    default: Any = None,
) -> Any:
    value = options.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        value = environ.get(env_name, default)
    if isinstance(value, str):
        value = Template(value).safe_substitute(environ).strip()
    return value


def _positive_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise LocalBackendConfigurationError(f"{name} must be an integer") from error
    if parsed < 1:
        raise LocalBackendConfigurationError(f"{name} must be positive")
    return parsed


def _bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _local_model_path(value: Any, *, name: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise LocalBackendConfigurationError(
            f"{name} is required in adapter.options or its documented environment variable"
        )
    if "$" in text:
        raise LocalBackendConfigurationError(f"{name} contains an unresolved variable")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise LocalBackendConfigurationError(f"{name} must be an absolute local path")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


class FrozenJiebaTokenizer:
    """Chinese tokenizer whose dictionary identity and HMM policy are auditable."""

    def __init__(self, tokenizer: JiebaLike, *, dictionary_sha256: str) -> None:
        if not dictionary_sha256.strip():
            raise ValueError("jieba dictionary SHA256 must be non-empty")
        self._tokenizer = tokenizer
        self.dictionary_sha256 = dictionary_sha256

    def __call__(self, text: str) -> tuple[str, ...]:
        # HMM must stay disabled: otherwise unseen-word segmentation can vary
        # independently from the frozen dictionary used by this experiment.
        return tuple(
            token.strip().casefold()
            for token in self._tokenizer.cut(text, cut_all=False, HMM=False)
            if token.strip()
        )


def build_frozen_jieba_tokenizer(
    dictionary_path: Path | None = None,
) -> FrozenJiebaTokenizer:
    """Create an isolated jieba tokenizer and hash the exact dictionary bytes."""

    try:
        import jieba
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "jieba is required for the local BGE backend; install requirements-rag.lock"
        ) from error

    if dictionary_path is None:
        dictionary_path = Path(jieba.__file__).resolve().parent / "dict.txt"
    dictionary_path = dictionary_path.expanduser().resolve()
    if not dictionary_path.is_file():
        raise FileNotFoundError(dictionary_path)
    tokenizer = jieba.Tokenizer(str(dictionary_path))
    tokenizer.initialize()
    return FrozenJiebaTokenizer(
        tokenizer, dictionary_sha256=_sha256_file(dictionary_path)
    )


def _default_dense_loader(path: Path, settings: Mapping[str, Any]) -> Any:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "FlagEmbedding is required for BGE-M3; install requirements-rag.lock"
        ) from error
    return BGEM3FlagModel(
        str(path),
        normalize_embeddings=True,
        use_fp16=bool(settings["use_fp16"]),
        devices=str(settings["device"]),
        batch_size=int(settings["batch_size"]),
        query_max_length=int(settings["query_max_length"]),
        passage_max_length=int(settings["passage_max_length"]),
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
        trust_remote_code=bool(settings["trust_remote_code"]),
        local_files_only=True,
    )


def _default_reranker_loader(path: Path, settings: Mapping[str, Any]) -> Any:
    try:
        from FlagEmbedding import FlagReranker
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "FlagEmbedding is required for BGE reranking; install requirements-rag.lock"
        ) from error
    return FlagReranker(
        str(path),
        use_fp16=bool(settings["use_fp16"]),
        devices=str(settings["device"]),
        batch_size=int(settings["batch_size"]),
        query_max_length=int(settings["query_max_length"]),
        max_length=int(settings["max_length"]),
        normalize=bool(settings["normalize"]),
        trust_remote_code=bool(settings["trust_remote_code"]),
        local_files_only=True,
    )


def _release_accelerator_cache() -> None:
    """Best-effort cache release without importing torch until a model was used."""

    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional outside RAG runs
        return
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class CachedNumpyDenseRetriever:
    """Cosine retriever backed by an immutable float32 NumPy matrix."""

    def __init__(
        self,
        units: Sequence[RetrievedUnit],
        matrix_path: Path,
        *,
        encode_query: Callable[[str], Any],
        numpy_module: Any | None = None,
    ) -> None:
        if not units:
            raise ValueError("dense corpus must be non-empty")
        np = numpy_module or _import_numpy()
        matrix = np.load(matrix_path, mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape[0] != len(units) or matrix.shape[1] < 1:
            raise ValueError("cached dense matrix does not match the corpus")
        self.units = tuple(units)
        self._np = np
        self._matrix = matrix
        self._encode_query = encode_query
        self._norms = np.linalg.norm(matrix, axis=1)

    def search(self, query: str, *, limit: int) -> tuple[SearchHit, ...]:
        if limit < 1:
            raise ValueError("dense limit must be positive")
        vector = self._np.asarray(self._encode_query(query), dtype=self._np.float32)
        if vector.ndim != 1 or vector.shape[0] != self._matrix.shape[1]:
            raise ValueError("query and document vector dimensions differ")
        query_norm = float(self._np.linalg.norm(vector))
        if query_norm == 0:
            scores = self._np.zeros(len(self.units), dtype=self._np.float32)
        else:
            denominators = self._norms * query_norm
            numerators = self._matrix @ vector
            scores = self._np.divide(
                numerators,
                denominators,
                out=self._np.zeros_like(numerators, dtype=self._np.float32),
                where=denominators != 0,
            )
        hits = (
            SearchHit(unit.unit_id, float(score))
            for unit, score in zip(self.units, scores, strict=True)
        )
        return tuple(sorted(hits, key=lambda item: (-item.score, item.unit_id))[:limit])


class LazyBgeReranker:
    """Load one shared reranker only when the first candidate list is scored."""

    def __init__(self, owner: "LocalBGERAGBackend") -> None:
        self._owner = owner

    def score(
        self, query: str, candidates: Sequence[RetrievedUnit]
    ) -> tuple[float, ...]:
        return self._owner._score_candidates(query, candidates)


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "NumPy is required for local dense indices; install requirements-rag.lock"
        ) from error
    return np


class LocalBGERAGBackend:
    """BM25 + BGE-M3 + BGE reranker + OpenAI-compatible generation backend."""

    def __init__(
        self,
        options: Mapping[str, Any],
        output_dir: Path,
        *,
        environ: Mapping[str, str] | None = None,
        dense_loader: DenseLoader | None = None,
        reranker_loader: RerankerLoader | None = None,
        jieba_tokenizer: FrozenJiebaTokenizer | None = None,
        generation_client: OpenAICompatibleRAGClient | Any | None = None,
        aligner: PredictionAligner | None = None,
        numpy_module: Any | None = None,
    ) -> None:
        self.options = dict(options)
        self.output_dir = output_dir
        self.environ = dict(os.environ if environ is None else environ)
        self._dense_loader = dense_loader or _default_dense_loader
        self._reranker_loader = reranker_loader or _default_reranker_loader
        self._jieba = jieba_tokenizer
        self._generation_client = generation_client
        self._aligner = aligner
        self._np = numpy_module
        self._dense_model: Any | None = None
        self._reranker_model: Any | None = None
        self._query_cache: dict[str, Any] = {}
        self._tokenizer: Any | None = None

        self.device = str(
            _option(
                self.options,
                "device",
                environ=self.environ,
                env_name="RAG_BGE_DEVICE",
                default="cpu",
            )
        )
        if self.device not in {"cpu", "mps", "cuda"} and not self.device.startswith(
            "cuda:"
        ):
            raise LocalBackendConfigurationError(
                "device must be cpu, mps, cuda, or cuda:<index>"
            )
        default_batch = 4 if self.device == "mps" else 16
        self.dense_batch_size = _positive_int(
            _option(
                self.options,
                "dense_batch_size",
                environ=self.environ,
                env_name="RAG_BGE_DENSE_BATCH_SIZE",
                default=default_batch,
            ),
            name="dense_batch_size",
        )
        self.reranker_batch_size = _positive_int(
            _option(
                self.options,
                "reranker_batch_size",
                environ=self.environ,
                env_name="RAG_BGE_RERANKER_BATCH_SIZE",
                default=2 if self.device == "mps" else 8,
            ),
            name="reranker_batch_size",
        )
        self.dense_cache_block_size = _positive_int(
            _option(
                self.options,
                "dense_cache_block_size",
                environ=self.environ,
                env_name="RAG_BGE_DENSE_CACHE_BLOCK_SIZE",
                default=max(64, self.dense_batch_size * 16),
            ),
            name="dense_cache_block_size",
        )
        self.dense_revision = str(
            _option(
                self.options,
                "dense_revision",
                environ=self.environ,
                env_name="RAG_BGE_M3_REVISION",
                default="unversioned",
            )
        )
        self.reranker_revision = str(
            _option(
                self.options,
                "reranker_revision",
                environ=self.environ,
                env_name="RAG_BGE_RERANKER_REVISION",
                default="unversioned",
            )
        )
        self._dictionary_path_text = str(
            _option(
                self.options,
                "jieba_dictionary_path",
                environ=self.environ,
                env_name="RAG_JIEBA_DICTIONARY_PATH",
                default="",
            )
        )
        self.token_count_mode = str(self.options.get("token_count_mode", "codepoints"))
        if self.token_count_mode not in {"codepoints", "hf_local"}:
            raise LocalBackendConfigurationError(
                "token_count_mode must be codepoints or hf_local"
            )

    def _dense_path(self) -> Path:
        return _local_model_path(
            _option(
                self.options,
                "dense_model_path",
                environ=self.environ,
                env_name="RAG_BGE_M3_MODEL_PATH",
            ),
            name="dense_model_path",
        )

    def _reranker_path(self) -> Path:
        return _local_model_path(
            _option(
                self.options,
                "reranker_model_path",
                environ=self.environ,
                env_name="RAG_BGE_RERANKER_MODEL_PATH",
            ),
            name="reranker_model_path",
        )

    def _dense_settings(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "batch_size": self.dense_batch_size,
            "use_fp16": _bool(self.options.get("dense_use_fp16"), default=False),
            "query_max_length": _positive_int(
                self.options.get("dense_query_max_length", 512),
                name="dense_query_max_length",
            ),
            "passage_max_length": _positive_int(
                self.options.get("dense_passage_max_length", 8192),
                name="dense_passage_max_length",
            ),
            "trust_remote_code": _bool(
                self.options.get("trust_remote_code"), default=False
            ),
        }

    def _reranker_settings(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "batch_size": self.reranker_batch_size,
            "use_fp16": _bool(self.options.get("reranker_use_fp16"), default=False),
            "query_max_length": _positive_int(
                self.options.get("reranker_query_max_length", 512),
                name="reranker_query_max_length",
            ),
            "max_length": _positive_int(
                self.options.get("reranker_max_length", 1024),
                name="reranker_max_length",
            ),
            "normalize": _bool(self.options.get("reranker_normalize"), default=True),
            "trust_remote_code": _bool(
                self.options.get("trust_remote_code"), default=False
            ),
        }

    def _get_jieba(self) -> FrozenJiebaTokenizer:
        if self._jieba is None:
            dictionary = (
                Path(self._dictionary_path_text)
                if self._dictionary_path_text
                else None
            )
            self._jieba = build_frozen_jieba_tokenizer(dictionary)
        return self._jieba

    def _get_dense_model(self) -> Any:
        if self._dense_model is None:
            self._dense_model = self._dense_loader(
                self._dense_path(), self._dense_settings()
            )
        return self._dense_model

    def _get_reranker_model(self) -> Any:
        if self._reranker_model is None:
            self._reranker_model = self._reranker_loader(
                self._reranker_path(), self._reranker_settings()
            )
        return self._reranker_model

    def _dense_adapter(self) -> BgeM3EmbeddingAdapter:
        return BgeM3EmbeddingAdapter(
            self._get_dense_model(),
            batch_size=self.dense_batch_size,
            max_length=self._dense_settings()["passage_max_length"],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

    def _cache_identity(self, corpus: CorpusBuild) -> dict[str, Any]:
        return {
            "schema": "okfolio.rag-dense-cache-identity.v1",
            "arm": corpus.arm,
            "unit_ids": [unit.unit_id for unit in corpus.units],
            "texts_sha256": _fingerprint(
                [unit.retrieval_text for unit in corpus.units]
            ),
            "model_revision": self.dense_revision,
            "settings": self._dense_settings(),
            "cache_block_size": self.dense_cache_block_size,
        }

    def _dense_cache_valid(self, corpus: CorpusBuild, index_dir: Path) -> bool:
        matrix_path = index_dir / "dense_embeddings.npy"
        metadata_path = index_dir / "dense_embeddings.meta.json"
        if not matrix_path.is_file() or not metadata_path.is_file():
            return False
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("identity") != self._cache_identity(corpus):
                return False
            np = self._np or _import_numpy()
            matrix = np.load(matrix_path, mmap_mode="r")
            return bool(
                matrix.ndim == 2
                and matrix.shape[0] == len(corpus.units)
                and matrix.shape[1] > 0
                and str(matrix.dtype) == "float32"
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def prepare_index(self, corpus: CorpusBuild, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = self._get_jieba()
        _atomic_json(
            index_dir / "bm25.meta.json",
            {
                "schema": "okfolio.rag-bm25-index.v1",
                "tokenizer": "jieba",
                "hmm": False,
                "dictionary_sha256": tokenizer.dictionary_sha256,
                "unit_ids_sha256": _fingerprint([u.unit_id for u in corpus.units]),
            },
        )
        if self._dense_cache_valid(corpus, index_dir):
            return
        np = self._np or _import_numpy()
        matrix_path = index_dir / "dense_embeddings.npy"
        temporary = index_dir / "dense_embeddings.npy.tmp"
        try:
            texts = [unit.retrieval_text for unit in corpus.units]
            blocks = []
            adapter = self._dense_adapter()
            for start in range(0, len(texts), self.dense_cache_block_size):
                # The shared adapter validates every bounded block. Keeping its
                # tuple conversion block-sized avoids expanding the complete
                # corpus into Python floats before writing a compact matrix.
                vectors = adapter.encode_documents(
                    texts[start : start + self.dense_cache_block_size]
                )
                blocks.append(np.asarray(vectors, dtype=np.float32))
            matrix = np.concatenate(blocks, axis=0)
            if matrix.ndim != 2 or matrix.shape[0] != len(corpus.units):
                raise ValueError("BGE-M3 document matrix does not match the corpus")
            with temporary.open("wb") as handle:
                np.save(handle, matrix, allow_pickle=False)
            os.replace(temporary, matrix_path)
            _atomic_json(
                index_dir / "dense_embeddings.meta.json",
                {
                    "schema": "okfolio.rag-dense-cache.v1",
                    "identity": self._cache_identity(corpus),
                    "shape": list(matrix.shape),
                    "dtype": str(matrix.dtype),
                },
            )
        finally:
            if temporary.exists():
                temporary.unlink()
            # Index construction and online retrieval are separate phases.
            self.release_dense_model()

    def create_bm25(self, corpus: CorpusBuild, index_dir: Path) -> BM25Retriever:
        metadata_path = index_dir / "bm25.meta.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        tokenizer = self._get_jieba()
        if metadata.get("dictionary_sha256") != tokenizer.dictionary_sha256:
            raise ValueError("jieba dictionary changed after index construction")
        return BM25Retriever(corpus.units, tokenizer=tokenizer)

    def _encode_query(self, query: str) -> Any:
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached
        adapter = BgeM3EmbeddingAdapter(
            self._get_dense_model(),
            batch_size=self.dense_batch_size,
            max_length=self._dense_settings()["query_max_length"],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        encoded = adapter.encode_queries([query])
        if len(encoded) != 1:
            raise ValueError("BGE-M3 query encoder must return one vector")
        np = self._np or _import_numpy()
        vector = np.asarray(encoded[0], dtype=np.float32)
        self._query_cache[query] = vector
        return vector

    def create_dense(
        self, corpus: CorpusBuild, index_dir: Path
    ) -> CachedNumpyDenseRetriever:
        if not self._dense_cache_valid(corpus, index_dir):
            raise ValueError("dense cache is absent or does not match the corpus")
        return CachedNumpyDenseRetriever(
            corpus.units,
            index_dir / "dense_embeddings.npy",
            encode_query=self._encode_query,
            numpy_module=self._np,
        )

    def create_reranker(self, corpus: CorpusBuild) -> LazyBgeReranker:
        del corpus
        return LazyBgeReranker(self)

    def _score_candidates(
        self, query: str, candidates: Sequence[RetrievedUnit]
    ) -> tuple[float, ...]:
        if not candidates:
            return ()
        adapter = BgeRerankerAdapter(
            self._get_reranker_model(),
            batch_size=self.reranker_batch_size,
            max_length=self._reranker_settings()["max_length"],
            normalize=self._reranker_settings()["normalize"],
        )
        return adapter.score(query, candidates)

    def count_tokens(self, text: str) -> int:
        if self.token_count_mode == "codepoints":
            # Conservative and provider-independent for Chinese text.  The
            # choice is frozen in adapter.options, hence in the run lock.
            return max(1, len(text))
        if self._tokenizer is None:
            path = _local_model_path(
                _option(
                    self.options,
                    "tokenizer_model_path",
                    environ=self.environ,
                    env_name="RAG_TOKENIZER_MODEL_PATH",
                ),
                name="tokenizer_model_path",
            )
            try:
                from transformers import AutoTokenizer
            except ImportError as error:  # pragma: no cover
                raise RuntimeError("transformers is required for hf_local token counting") from error
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(path), trust_remote_code=False, local_files_only=True
            )
        encoded = self._tokenizer.encode(text, add_special_tokens=False)
        return max(1, len(encoded))

    def _get_generation_client(self) -> OpenAICompatibleRAGClient:
        if self._generation_client is not None:
            return cast(OpenAICompatibleRAGClient, self._generation_client)
        section = self.options.get("generation") or {}
        if not isinstance(section, Mapping):
            raise LocalBackendConfigurationError("generation must be an object")
        if section.get("api_key"):
            raise LocalBackendConfigurationError(
                "do not persist generation.api_key in config; use RAG_LLM_API_KEY"
            )
        overlay = dict(self.environ)
        for key, env_name in (
            ("base_url", "RAG_LLM_BASE_URL"),
            ("model", "RAG_LLM_MODEL"),
            ("timeout_seconds", "RAG_LLM_TIMEOUT_SECONDS"),
            ("max_output_tokens", "RAG_LLM_MAX_OUTPUT_TOKENS"),
            ("temperature", "RAG_LLM_TEMPERATURE"),
            ("include_stream_usage", "RAG_LLM_INCLUDE_STREAM_USAGE"),
            ("response_format", "RAG_LLM_RESPONSE_FORMAT"),
        ):
            if key in section:
                overlay[env_name] = Template(str(section[key])).safe_substitute(
                    self.environ
                )
        self._generation_client = OpenAICompatibleRAGClient(
            RAGGenerationConfig.from_env(overlay)
        )
        return self._generation_client

    def _get_aligner(self) -> PredictionAligner | None:
        if self._aligner is not None:
            return self._aligner
        reference = str(self.options.get("prediction_aligner_factory", "")).strip()
        if not reference:
            return None
        if ":" not in reference:
            raise LocalBackendConfigurationError(
                "prediction_aligner_factory must use module:function syntax"
            )
        module_name, function_name = reference.split(":", 1)
        factory = getattr(importlib.import_module(module_name), function_name)
        aligner_options = self.options.get("prediction_aligner_options") or {}
        if not isinstance(aligner_options, Mapping):
            raise LocalBackendConfigurationError(
                "prediction_aligner_options must be an object"
            )
        self._aligner = factory(dict(aligner_options), self.output_dir)
        return self._aligner

    def _external_alignment_status(
        self,
    ) -> Literal["human_reviewed", "independent_judge"]:
        status = str(self.options.get("prediction_alignment_status", "")).strip()
        allowed = {"human_reviewed", "independent_judge"}
        if status not in allowed:
            raise LocalBackendConfigurationError(
                "an external prediction aligner must declare "
                "prediction_alignment_status as human_reviewed or "
                "independent_judge"
            )
        return cast(Literal["human_reviewed", "independent_judge"], status)

    def generate(
        self,
        *,
        gold: GoldQuestion,
        arm: Arm,
        request: AnswerGenerationInput,
    ) -> GeneratedAnswer:
        aligner = self._get_aligner()
        external_alignment_status = (
            self._external_alignment_status() if aligner is not None else None
        )
        self.release_retrieval_models()
        result = self._get_generation_client().generate_answer(request, stream=True)
        structured = parse_structured_answer(result.text, request)
        contract_prediction = structured_answer_prediction(
            gold.question_id, structured
        )
        if aligner is None:
            prediction = contract_prediction
            alignment_status = "provisional_structured"
            alignment_method = "contract-and-provenance-only"
        else:
            normalized_result = replace(result, text=structured.answer)
            aligned_facts = aligner.align(
                gold=gold, arm=arm, request=request, result=normalized_result
            )
            if not isinstance(aligned_facts, AnswerPrediction):
                raise TypeError("prediction aligner must return AnswerPrediction")
            if aligned_facts.question_id != gold.question_id:
                raise ValueError("prediction aligner returned the wrong question_id")
            # The external reviewer supplies only semantic fact decisions.
            # Refusal and citations remain exactly those parsed and mapped by
            # the provider-neutral answer contract.
            prediction = AnswerPrediction(
                question_id=gold.question_id,
                predicted_answerable=contract_prediction.predicted_answerable,
                matched_required_fact_ids=aligned_facts.matched_required_fact_ids,
                asserted_forbidden_fact_ids=aligned_facts.asserted_forbidden_fact_ids,
                citations=contract_prediction.citations,
            )
            assert external_alignment_status is not None
            alignment_status = external_alignment_status
            alignment_method = "explicit-plugin"
        return GeneratedAnswer(
            text=structured.answer,
            prediction=prediction,
            semantic_alignment_status=alignment_status,
            usage=asdict(result.usage),
            timing=asdict(result.timing),
            metadata={
                "finish_reason": result.finish_reason,
                "stream_events": result.stream_events,
                "alignment": alignment_method,
                "structured_answer": structured.to_dict(),
            },
        )

    def generate_hyde(self, query: str) -> str:
        result = self._get_generation_client().generate_hyde(
            query, language="zh-CN", stream=False
        )
        return result.text

    def release_dense_model(self) -> None:
        self._dense_model = None
        _release_accelerator_cache()

    def release_retrieval_models(self) -> None:
        self._dense_model = None
        self._reranker_model = None
        self._query_cache.clear()
        _release_accelerator_cache()


def create_backend(
    options: Mapping[str, Any], output_dir: Path
) -> LocalBGERAGBackend:
    """Factory referenced by ``scripts/run_rag_experiment.py`` configs."""

    return LocalBGERAGBackend(options, output_dir)


__all__ = [
    "AlignmentRequiredError",
    "CachedNumpyDenseRetriever",
    "FrozenJiebaTokenizer",
    "LazyBgeReranker",
    "LocalBGERAGBackend",
    "LocalBackendConfigurationError",
    "build_frozen_jieba_tokenizer",
    "create_backend",
]
