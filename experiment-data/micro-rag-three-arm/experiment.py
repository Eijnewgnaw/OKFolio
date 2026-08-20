#!/usr/bin/env python3
"""Directional micro-experiment: three-arm retrieval quality.

C1 (accepted AgentWiki Concepts) vs T0 (fixed-length chunks) vs T1
(heading-aware Parent-Child chunks), over 140 accepted v6 Claim-Review
checkpoint groups from the OKFolio Concept-Compiler experiment repo.

Retrieval chain (NO rerank - LM Studio has no rerank endpoint, verified 404):
    BM25(top_k=50, jieba) + dense(top_k=50, bge-m3-mlx via LM Studio) + RRF(fusion_top_k=50, rrf_k=60)

Read-only on the repo: no repo file is ever written.  Outputs:
    <out-dir>/result.json     (overwritten on every run; idempotent)
    console Markdown tables + summary

Paths and knobs are CLI arguments / environment variables with defaults that
match the original archive run:
  --repo         (or $OKFOLIO_REPO)  default ~/OKFolio-Concept-Compiler-Experiment-20260810
  --run-dir                          default <repo>/experiment-data/runs/public10-...-v6-nothinking-20260810
  --structures-dir                   default <repo>/data/normalized-sources
  --out-dir      (or $MICRO_RAG_OUT_DIR) default /tmp/micro-rag
  --emb-url / --emb-model / --emb-dim / --poll-attempts (or EMB_URL/EMB_MODEL/EMB_POLL_ATTEMPTS)

Run `python experiment.py --help` for the full list.

Reuse of repo implementations (not reinvented):
  - okfolio.evaluation.corpus.build_t0_fixed_chunks / build_t1_parent_child
    (T0 = block-preserving fixed 1200-char chunks; T1 = heading-aware
     parent<=4800 / child<=600 char).  Article set A == all 10 articles
     (verified: every gold block id resolves into the 10 structures, 0 dups),
     so no article-level filtering is required and the builders are used
     directly on data/normalized-sources.
  - okfolio.evaluation.retrieval_adapters.BM25Retriever (jieba tokenizer
    injected) and InMemoryDenseRetriever (cosine, exact).
  - okfolio.evaluation.retrieval.reciprocal_rank_fusion (RRF, rrf_k=60).

Note: okfolio.evaluation.build_c1_audited_concepts was NOT reusable for C1:
it requires run_dir manifest/acceptance/refs/concepts.json + concepts/*.md,
which do not exist in this run directory (only checkpoints/ + progress jsons).
C1 units are therefore built directly from the checkpoints per the task spec:
unit_id=f"c1-{group_id}", retrieval_text=draft.body, gold = source_blocks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import types
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

EMB_DIM = 1024

BM25_TOP_K = 50
DENSE_TOP_K = 50
FUSION_TOP_K = 50
RRF_K = 60
T0_MAX_CHARS = 1200
T1_CHILD_MAX_CHARS = 600
T1_PARENT_MAX_CHARS = 4800

BATCH_DOCS = 128
BATCH_QUERIES = 64
REQUEST_TIMEOUT = 300.0
MODEL_POLL_SECS = 15
DENSE_SLOW_THRESHOLD = 20.0 * 60.0  # if projected dense indexing > 20 min, fall back

DEFAULT_RUN_DIR = (
    "experiment-data/runs/public10-claim-review-formal-qwen3p6-remote-v6-nothinking-20260810"
)


@dataclass
class Config:
    """Resolved runtime paths and knobs (all overridable, defaults preserved)."""

    repo: Path
    run_dir: Path
    checkpoint_dir: Path
    structures_dir: Path
    out_dir: Path
    result_path: Path
    emb_url: str
    emb_model: str
    emb_dim: int
    poll_attempts: int
    batch_docs: int
    batch_queries: int


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="OKFolio three-arm retrieval-quality micro-experiment (C1 vs T0 vs T1)"
    )
    ap.add_argument(
        "--repo",
        default=os.environ.get(
            "OKFOLIO_REPO", "~/OKFolio-Concept-Compiler-Experiment-20260810"
        ),
        help="repo root (default: ~/OKFolio-Concept-Compiler-Experiment-20260810 or $OKFOLIO_REPO)",
    )
    ap.add_argument(
        "--run-dir",
        default=None,
        help=f"v6 claim-review run dir (default: <repo>/{DEFAULT_RUN_DIR})",
    )
    ap.add_argument(
        "--structures-dir",
        default=None,
        help="normalized-source structures dir (default: <repo>/data/normalized-sources)",
    )
    ap.add_argument(
        "--out-dir",
        default=os.environ.get("MICRO_RAG_OUT_DIR", "/tmp/micro-rag"),
        help="output dir for result.json (default: /tmp/micro-rag or $MICRO_RAG_OUT_DIR)",
    )
    ap.add_argument(
        "--emb-url",
        default=os.environ.get("EMB_URL", "http://localhost:1234/v1/embeddings"),
        help="OpenAI-compatible embeddings endpoint (default: LM Studio localhost:1234)",
    )
    ap.add_argument(
        "--emb-model",
        default=os.environ.get("EMB_MODEL", "bge-m3-mlx"),
        help="embedding model id (default: bge-m3-mlx)",
    )
    ap.add_argument("--emb-dim", type=int, default=EMB_DIM, help="expected embedding dim")
    ap.add_argument(
        "--poll-attempts",
        type=int,
        default=int(os.environ.get("EMB_POLL_ATTEMPTS", "20")),
        help="embedding readiness polls, 15s apart (default: 20 = 5 min)",
    )
    ap.add_argument("--batch-docs", type=int, default=BATCH_DOCS)
    ap.add_argument("--batch-queries", type=int, default=BATCH_QUERIES)
    return ap.parse_args(argv)


def make_config(args) -> Config:
    repo = Path(args.repo).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser() if args.run_dir else repo / DEFAULT_RUN_DIR
    structures_dir = (
        Path(args.structures_dir).expanduser()
        if args.structures_dir
        else repo / "data" / "normalized-sources"
    )
    out_dir = Path(args.out_dir).expanduser()
    return Config(
        repo=repo,
        run_dir=run_dir,
        checkpoint_dir=run_dir / "checkpoints",
        structures_dir=structures_dir,
        out_dir=out_dir,
        result_path=out_dir / "result.json",
        emb_url=args.emb_url,
        emb_model=args.emb_model,
        emb_dim=args.emb_dim,
        poll_attempts=args.poll_attempts,
        batch_docs=args.batch_docs,
        batch_queries=args.batch_queries,
    )


CFG = make_config(parse_args())

# --------------------------------------------------------------------------
# Minimal package shells: load only the four needed submodules so the heavy
# okfolio.evaluation/__init__ (httpx/openai/jsonschema/...) is never executed.
# --------------------------------------------------------------------------
def _load_repo_modules(repo: Path):
    okfolio = types.ModuleType("okfolio")
    okfolio.__path__ = [str(repo / "okfolio")]
    sys.modules["okfolio"] = okfolio
    ev = types.ModuleType("okfolio.evaluation")
    ev.__path__ = [str(repo / "okfolio" / "evaluation")]
    sys.modules["okfolio.evaluation"] = ev

    def load(name, rel):
        spec_path = repo / "okfolio" / "evaluation" / rel
        spec = __import__("importlib.util").util.spec_from_file_location(name, spec_path)
        mod = __import__("importlib.util").util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    contracts = load("okfolio.evaluation.contracts", "contracts.py")
    corpus = load("okfolio.evaluation.corpus", "corpus.py")
    retrieval = load("okfolio.evaluation.retrieval", "retrieval.py")
    adapters = load("okfolio.evaluation.retrieval_adapters", "retrieval_adapters.py")
    return contracts, corpus, retrieval, adapters


contracts, corpus_mod, retrieval_mod, adapters_mod = _load_repo_modules(CFG.repo)
EvidenceAtomId = contracts.EvidenceAtomId
RetrievedUnit = contracts.RetrievedUnit
CorpusBuild = corpus_mod.CorpusBuild
BM25Retriever = adapters_mod.BM25Retriever
InMemoryDenseRetriever = adapters_mod.InMemoryDenseRetriever
reciprocal_rank_fusion = retrieval_mod.reciprocal_rank_fusion

import jieba  # noqa: E402  (venv-micro)

# --------------------------------------------------------------------------
# LM Studio embedding client (OpenAI-compatible)
# --------------------------------------------------------------------------
class LMStudioEmbedding:
    """Embedding client for LM Studio's /v1/embeddings endpoint.

    Exposes encode_documents/encode_queries so the repo's InMemoryDenseRetriever
    can consume it unchanged.  Memoizes by text (queries are asked 3x, once per
    arm).  Tracks actual embedding call counts for the report.
    """

    def __init__(self, url=None, model=None):
        self.url = url or CFG.emb_url
        self.model = model or CFG.emb_model
        self._cache: dict[str, tuple[float, ...]] = {}
        self.texts_embedded = 0  # texts actually sent to the server
        self.http_requests = 0
        self.dim = None
        self._ensure_loaded()

    # -- startup: wait until the model is actually loaded ------------------
    def _ensure_loaded(self):
        attempts = 0
        while True:
            try:
                vec = self._embed_once(["测试"])
                if self.dim is None:
                    self.dim = len(vec[0])
                print(f"[lmstudio] model '{self.model}' ready, dim={self.dim}", flush=True)
                return
            except ModelNotLoadedError:
                attempts += 1
                if attempts > CFG.poll_attempts:
                    raise RuntimeError(
                        "LM Studio embedding endpoint never loaded a model within "
                        f"{CFG.poll_attempts * MODEL_POLL_SECS}s"
                    )
                print(
                    f"[lmstudio] model not loaded yet ({attempts}/{CFG.poll_attempts}), "
                    f"retrying in {MODEL_POLL_SECS}s...",
                    flush=True,
                )
                time.sleep(MODEL_POLL_SECS)
            except Exception as exc:  # noqa: BLE001 - surface real errors
                attempts += 1
                if attempts > 5:
                    raise RuntimeError(f"embedding endpoint unreachable: {exc}") from exc
                print(f"[lmstudio] transient error: {exc}; retrying in 5s", flush=True)
                time.sleep(5)

    def _embed_once(self, texts):
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            self.http_requests += 1
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            if "no models loaded" in raw.lower():
                raise ModelNotLoadedError(raw) from exc
            raise RuntimeError(f"LM Studio HTTP {exc.code}: {raw}") from exc
        if "error" in body:
            if "no models loaded" in str(body["error"]).lower():
                raise ModelNotLoadedError(str(body["error"]))
            raise RuntimeError(f"LM Studio error: {body['error']}")
        data = body.get("data") or []
        vectors = [item["embedding"] for item in sorted(data, key=lambda d: d.get("index", 0))]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"embedding count mismatch: asked {len(texts)}, got {len(vectors)}"
            )
        return [tuple(float(v) for v in vec) for vec in vectors]

    def embed(self, texts):
        """Batch-embed texts with retry/backoff; memoized by text."""
        missing = [(i, t) for i, t in enumerate(texts) if t not in self._cache]
        if missing:
            batch = []
            batch_of = {}
            for i, t in missing:
                batch.append(t)
                batch_of[len(batch) - 1] = i
            vectors_all = []
            for start in range(0, len(batch), CFG.batch_docs):
                chunk = batch[start : start + CFG.batch_docs]
                for attempt in range(5):
                    try:
                        vectors_all.extend(self._embed_once(chunk))
                        break
                    except ModelNotLoadedError:
                        time.sleep(MODEL_POLL_SECS)
                    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                        if attempt == 4:
                            raise RuntimeError(f"embedding failed after retries: {exc}") from exc
                        time.sleep(2 ** (attempt + 1))
            for j, t in enumerate(batch):
                self._cache[t] = tuple(vectors_all[j])
            self.texts_embedded += len(batch)
        return [self._cache[t] for t in texts]

    def encode_documents(self, texts):
        return self.embed(list(texts))

    def encode_queries(self, texts):
        return self.embed(list(texts))


class ModelNotLoadedError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_accepted_groups():
    """Return list of dicts: group_id, question, body, gold_block_ids."""
    groups = []
    files = sorted(CFG.checkpoint_dir.glob("*.json"))
    for path in files:
        cp = json.loads(path.read_text(encoding="utf-8"))
        if cp.get("decision") != "pass":
            continue
        gold = set()
        for ep in cp.get("evidence_provenance") or []:
            for sb in ep.get("source_blocks") or []:
                bid = str(sb.get("block_id", "")).strip()
                if bid:
                    gold.add(bid)
        question = str(cp.get("contract", {}).get("canonical_question", "")).strip()
        if not question:
            question = str(cp.get("draft", {}).get("title", "")).strip()
        groups.append(
            {
                "group_id": str(cp["group_id"]),
                "question": question,
                "body": str(cp.get("draft", {}).get("body", "")),
                "gold": gold,
            }
        )
    return groups


def build_block_map():
    """block_id -> (article_id, block dict) across all 10 structures."""
    mapping = {}
    dup = 0
    for path in sorted(CFG.structures_dir.glob("*.structure.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        aid = str(payload.get("document_id", ""))
        for blk in payload.get("blocks", []):
            bid = str(blk.get("block_id", "")).strip()
            if not bid:
                continue
            if bid in mapping:
                dup += 1
            else:
                mapping[bid] = (aid, blk)
    return mapping, dup


def build_c1_corpus(groups, block_map):
    """C1 units per task spec: unit_id=c1-{group_id}, text=draft.body."""
    units = []
    skipped = []
    for g in groups:
        atom_ids = []
        articles = set()
        pages = []
        for bid in sorted(g["gold"]):
            if bid not in block_map:
                skipped.append((g["group_id"], bid))
                continue
            aid, blk = block_map[bid]
            page = int(blk.get("page_number") or 1)
            atom = EvidenceAtomId(
                article_id=aid, page=page, atom_kind="block",
                locator_id=bid.removeprefix("blk-"),
            )
            if atom not in atom_ids:
                atom_ids.append(atom)
            articles.add(aid)
            pages.append(page)
        text = g["body"].strip()
        if not text:
            skipped.append((g["group_id"], "<empty body>"))
            continue
        units.append(
            RetrievedUnit(
                unit_id=f"c1-{g['group_id']}",
                arm="C1",
                retrieval_text=text,
                context_text=text,
                retrieval_evidence_atom_ids=tuple(atom_ids),
                context_evidence_atom_ids=tuple(atom_ids),
                article_ids=tuple(sorted(articles)),
                metadata={"group_id": g["group_id"]},
            )
        )
    return units, skipped


def atom_to_block_id(atom):
    return f"blk-{atom.locator_id}"


def unit_block_coverage(corpus: CorpusBuild):
    """unit_id -> frozenset(block_id) covered by the unit's retrieval atoms."""
    atom_block = {}
    for atom in corpus.evidence_atoms:
        atom_block[atom.atom_id] = atom.block_id
    coverage = {}
    for unit in corpus.units:
        cov = {atom_block[a] for a in unit.retrieval_evidence_atom_ids if a in atom_block}
        coverage[unit.unit_id] = frozenset(cov)
    return coverage


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
RECALL_CUTS = (5, 10, 20, 50)
EFF_METRIC = "recall10_per_1k_tokens"


def evaluate_arm(units, coverage, questions, fused_by_qid, label):
    """Compute per-question metrics for one arm over the fused top-50 ranks.

    Retrieval metrics: recall@5/10/20/50, MRR, nDCG@50, all derived from the
    same fused top-50 ranking (RRF, rrf_k=60).  Budget-efficiency metric:
    ``recall10_per_1k_tokens`` = recall@10 / (avg_retrieved_text_tokens / 1000),
    where avg_retrieved_text_tokens is the mean ``len(text)/2`` Chinese-token
    approximation over the top-50 fused units (per question).
    """
    cov_list = [coverage[u.unit_id] for u in units]  # aligned with corpus units
    rows = {}
    for q in questions:
        qid = q["group_id"]
        gold = q["gold"]
        n_gold = len(gold)
        fused = fused_by_qid[qid]  # list of (rank, unit) sorted by rank
        top = fused[:FUSION_TOP_K]
        if n_gold == 0:
            rows[qid] = {
                **{f"recall@{k}": None for k in RECALL_CUTS},
                "mrr": None, "ndcg@50": None, EFF_METRIC: None,
                "gold_block_count": 0, "retrieved_units": len(top),
                "avg_retrieved_text_tokens": 0.0,
            }
            continue
        covered = {k: set() for k in RECALL_CUTS}
        mrr = 0.0
        dcg = 0.0
        for rank, unit in top:
            hit = coverage[unit.unit_id] & gold
            for k in RECALL_CUTS:
                if rank <= k:
                    covered[k] |= hit
            if mrr == 0.0 and hit:
                mrr = 1.0 / rank
            if hit:
                dcg += 1.0 / math.log2(rank + 1)
        # R = number of corpus units covering >=1 gold block (for IDCG)
        R = 0
        for cov in cov_list:
            if cov & gold:
                R += 1
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(R, FUSION_TOP_K) + 1)) if R else 0.0
        ndcg = dcg / idcg if idcg > 0 else 0.0
        avg_tokens = (
            sum(len(u.retrieval_text) / 2.0 for _, u in top) / len(top) if top else 0.0
        )
        recall10 = len(covered[10]) / n_gold
        eff = recall10 / (avg_tokens / 1000.0) if avg_tokens > 0 else None
        rows[qid] = {
            **{f"recall@{k}": len(covered[k]) / n_gold for k in RECALL_CUTS},
            "mrr": mrr,
            "ndcg@50": ndcg,
            EFF_METRIC: eff,
            "gold_block_count": n_gold,
            "retrieved_units": len(top),
            "avg_retrieved_text_tokens": round(avg_tokens, 2),
        }
    return rows


def macro_mean(rows, key):
    vals = [r[key] for r in rows.values() if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    CFG.out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print("== OKFolio three-arm micro-experiment (C1 vs T0 vs T1) ==", flush=True)

    # --- load checkpoints ------------------------------------------------
    groups = load_accepted_groups()
    qids = [g["group_id"] for g in groups]
    assert len(qids) == len(set(qids)), "duplicate group_id in accepted groups"
    empty_gold = [g for g in groups if not g["gold"]]
    empty_q = [g for g in groups if not g["question"]]
    print(f"[data] accepted groups: {len(groups)} (gold-empty: {len(empty_gold)}, "
          f"question-empty: {len(empty_q)})", flush=True)
    assert not empty_gold and not empty_q, "unexpected empty gold/question"

    # --- block map + article set A ---------------------------------------
    block_map, dup_blocks = build_block_map()
    gold_all = set().union(*(g["gold"] for g in groups))
    missing = sorted(gold_all - set(block_map))
    assert not missing, f"gold blocks missing from structures: {missing[:5]}"
    A = sorted({block_map[b][0] for b in gold_all})
    ineligible = [
        b for b in gold_all if not block_map[b][1].get("evidence_eligible", False)
    ]
    print(f"[data] block_id->article map: {len(block_map)} blocks, {dup_blocks} dup block_ids", flush=True)
    n_structures = len(list(CFG.structures_dir.glob("*.structure.json")))
    print(f"[data] article set A: {len(A)} articles, {len(gold_all)} unique gold blocks, "
          f"{len(ineligible)} gold blocks not evidence_eligible", flush=True)
    assert len(A) == n_structures, "A != all structures"

    # --- C1 corpus ---------------------------------------------------------
    c1_units, c1_skipped = build_c1_corpus(groups, block_map)
    print(f"[corpus] C1 units: {len(c1_units)} (skipped: {len(c1_skipped)})", flush=True)
    assert len(c1_units) == len(groups)

    # --- T0 / T1 corpora (repo builders; A == all 10 articles) -----------
    t0 = corpus_mod.build_t0_fixed_chunks(CFG.structures_dir, max_chars=T0_MAX_CHARS)
    t1 = corpus_mod.build_t1_parent_child(
        CFG.structures_dir, child_max_chars=T1_CHILD_MAX_CHARS,
        parent_max_chars=T1_PARENT_MAX_CHARS,
    )
    assert t0.audit()["status"] == "pass" and t1.audit()["status"] == "pass"
    print(f"[corpus] T0 units: {len(t0.units)}; T1 units: {len(t1.units)}", flush=True)

    # C1 unit->block coverage: derive directly from each unit's atom ids
    # (block_id = "blk-" + locator_id); no shell CorpusBuild needed.
    c1_coverage = {}
    for u in c1_units:
        c1_coverage[u.unit_id] = frozenset(f"blk-{a.locator_id}" for a in u.retrieval_evidence_atom_ids)
    t0_coverage = unit_block_coverage(t0)
    t1_coverage = unit_block_coverage(t1)
    print(
        f"[corpus] unit->block coverage built: C1={len(c1_coverage)} T0={len(t0_coverage)} "
        f"T1={len(t1_coverage)}", flush=True
    )

    # --- tokenizer (jieba, init once) --------------------------------------
    def tokenize(text):
        return [t for t in jieba.cut(text) if t.strip()]

    # warm-up so BM25 build timing is not polluted
    _ = tokenize("成都自贸试验区如何通过校企合作解决人才短缺问题")

    # --- dense backend ------------------------------------------------------
    embedding = LMStudioEmbedding()
    if embedding.dim != CFG.emb_dim:
        print(f"[lmstudio] WARNING: server dim={embedding.dim}, expected {CFG.emb_dim}", flush=True)

    # --- retrieve all questions x arms --------------------------------------
    questions = groups
    fused = {}  # (arm, qid) -> [(rank, unit)]
    arms = {"C1": c1_units, "T0": t0.units, "T1": t1.units}

    dense_mode = "full-corpus"
    for arm, units in arms.items():
        print(f"[retrieval] {arm}: building BM25 + dense index over {len(units)} units...", flush=True)
        ta = time.time()
        bm25 = BM25Retriever(units, tokenizer=tokenize)
        print(f"[retrieval] {arm}: BM25 ready in {time.time()-ta:.1f}s", flush=True)
        dense = InMemoryDenseRetriever(units, embedding=embedding)
        print(
            f"[retrieval] {arm}: dense ready in {time.time()-ta:.1f}s "
            f"(cum texts embedded: {embedding.texts_embedded})",
            flush=True,
        )
        for q in questions:
            qid = q["group_id"]
            bm25_hits = bm25.search(q["question"], limit=BM25_TOP_K)
            dense_hits = dense.search(q["question"], limit=DENSE_TOP_K)
            fused_hits = reciprocal_rank_fusion(
                (bm25_hits, dense_hits), limit=FUSION_TOP_K, rrf_k=RRF_K
            )
            unit_by_id = {u.unit_id: u for u in units}
            fused[(arm, qid)] = [(rank, unit_by_id[h.unit_id]) for rank, h in enumerate(fused_hits, start=1)]
    print(f"[retrieval] done. dense mode: {dense_mode}; total texts embedded: "
          f"{embedding.texts_embedded}; http requests: {embedding.http_requests}", flush=True)

    # --- metrics ------------------------------------------------------------
    rows = {}
    for arm in ("C1", "T0", "T1"):
        rows[arm] = evaluate_arm(
            arms[arm],
            {"C1": c1_coverage, "T0": t0_coverage, "T1": t1_coverage}[arm],
            questions,
            {qid: fused[(arm, qid)] for qid in qids},
            arm,
        )

    metrics = ["recall@5", "recall@10", "recall@20", "recall@50", "mrr", "ndcg@50"]
    pair_metrics = metrics + [EFF_METRIC]
    summary = {
        m: {arm: macro_mean(rows[arm], m) for arm in ("C1", "T0", "T1")}
        for m in pair_metrics
    }
    pairs = [("C1", "T0"), ("C1", "T1")]
    pairwise = {}
    eps = 1e-9
    for m in pair_metrics:
        pairwise[m] = {}
        for a, b in pairs:
            deltas = [rows[a][qid][m] - rows[b][qid][m] for qid in qids]
            wins = sum(1 for d in deltas if d > eps)
            ties = sum(1 for d in deltas if abs(d) <= eps)
            losses = sum(1 for d in deltas if d < -eps)
            pairwise[m][f"{a}-{b}"] = {
                "mean_delta": sum(deltas) / len(deltas),
                "wins": wins, "ties": ties, "losses": losses,
            }

    # corpus-level stats
    corpus_stats = {
        "A_articles": len(A),
        "gold_blocks": len(gold_all),
        "questions": len(questions),
        "units": {arm: len(arms[arm]) for arm in arms},
        "avg_unit_text_chars": {
            arm: round(sum(len(u.retrieval_text) for u in arms[arm]) / len(arms[arm]), 1)
            for arm in arms
        },
        "dense_mode": dense_mode,
        "embedding_texts_embedded": embedding.texts_embedded,
        "embedding_http_requests": embedding.http_requests,
        "token_approx_note": "len(text)/2 chars-per-token approximation for Chinese",
    }
    avg_tok_per_q = {
        arm: round(macro_mean(rows[arm], "avg_retrieved_text_tokens"), 1) for arm in arms
    }

    # --- judgment -----------------------------------------------------------
    # C1 leads if its mean beats both baselines on the majority of the 6
    # retrieval metrics (efficiency metric reported but not part of judgment).
    leads = sum(
        1 for m in metrics
        if pairwise[m]["C1-T0"]["mean_delta"] > 0 and pairwise[m]["C1-T1"]["mean_delta"] > 0
    )
    if leads >= 5:
        judgment = (
            "C1 占优（6 项检索指标中 %d 项同时胜过 T0 与 T1 的宏观均值，含符号统计）"
            % leads
        )
    elif leads >= 3:
        judgment = (
            "C1 部分占优（6 项检索指标中 %d 项同时胜过 T0 与 T1；其余项未占优）" % leads
        )
    elif leads >= 1:
        judgment = "C1 弱占优（仅 %d 项检索指标同时胜过 T0 与 T1，方向性证据不足）" % leads
    else:
        judgment = "C1 未占优（无检索指标同时胜过 T0 与 T1）"

    # --- output --------------------------------------------------------------
    payload = {
        "schema": "okfolio.micro-experiment.retrieval-quality.v1",
        "version": 2,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metrics": {"retrieval": metrics, "efficiency": [EFF_METRIC]},
        "metric_notes": {
            EFF_METRIC: (
                "recall@10 / (avg_retrieved_text_tokens / 1000); "
                "avg_retrieved_text_tokens = mean len(text)/2 Chinese-token "
                "approximation over the top-50 fused units per question"
            ),
            "token_approx": "len(text)/2 chars-per-token approximation for Chinese",
        },
        "config": {
            "bm25_top_k": BM25_TOP_K, "dense_top_k": DENSE_TOP_K,
            "fusion_top_k": FUSION_TOP_K, "rrf_k": RRF_K,
            "t0_max_chars": T0_MAX_CHARS, "t1_child_max_chars": T1_CHILD_MAX_CHARS,
            "t1_parent_max_chars": T1_PARENT_MAX_CHARS,
            "embedding_model": CFG.emb_model, "embedding_endpoint": CFG.emb_url,
            "rerank": "none (LM Studio has no rerank endpoint; verified 404)",
            "dense_mode": dense_mode,
        },
        "corpus_stats": corpus_stats,
        "per_question": {
            qid: {
                "question": g["question"],
                "gold_block_count": len(g["gold"]),
                "arms": {arm: rows[arm][qid] for arm in ("C1", "T0", "T1")},
            }
            for qid, g in zip(qids, questions)
        },
        "summary": summary,
        "avg_retrieved_text_tokens_per_question": avg_tok_per_q,
        "pairwise": pairwise,
        "judgment": judgment,
    }
    CFG.result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[output] wrote {CFG.result_path}", flush=True)

    # --- console tables ------------------------------------------------------
    print()
    print("### 宏观指标（macro mean over %d questions）" % len(questions))
    print("| metric | C1 | T0 | T1 |")
    print("|---|---|---|---|")
    for m in pair_metrics:
        print("| %s | %.4f | %.4f | %.4f |" % (m, summary[m]["C1"], summary[m]["T0"], summary[m]["T1"]))
    print()
    print("### 配对差值（macro mean delta, wins/ties/losses）")
    print("| metric | C1−T0 mean | C1−T0 w/t/l | C1−T1 mean | C1−T1 w/t/l |")
    print("|---|---|---|---|---|")
    for m in pair_metrics:
        p0 = pairwise[m]["C1-T0"]
        p1 = pairwise[m]["C1-T1"]
        print(
            "| %s | %+.4f | %d/%d/%d | %+.4f | %d/%d/%d |"
            % (m, p0["mean_delta"], p0["wins"], p0["ties"], p0["losses"],
               p1["mean_delta"], p1["wins"], p1["ties"], p1["losses"])
        )
    print()
    print("### 中间统计")
    print("- 文章集 A：%d 本（全部 10 本；gold block 全部落在其中，block_id 无重复）" % len(A))
    print("- 语料单位数：C1=%d, T0=%d, T1=%d" % (len(arms["C1"]), len(arms["T0"]), len(arms["T1"])))
    print("- 每问题平均检索文本 token 数（len/2 近似）：C1=%s, T0=%s, T1=%s" % (
        avg_tok_per_q["C1"], avg_tok_per_q["T0"], avg_tok_per_q["T1"]))
    print("- 总 embedding 调用：texts=%d, http_requests=%d（dense 模式：%s；无 rerank）" % (
        embedding.texts_embedded, embedding.http_requests, dense_mode))
    print()
    print("### 方向性判断")
    print(judgment)
    print()
    print(f"[done] total wall time: {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
