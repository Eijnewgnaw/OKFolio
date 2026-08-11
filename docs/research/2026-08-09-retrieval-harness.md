# Reproducible three-arm retrieval harness

## Implemented boundary

The experiment core uses a direct, framework-neutral adapter layer rather than
making Haystack a required dependency. This is deliberate:

- T0, T1, and C1 already emit audited `RetrievedUnit` objects with canonical
  evidence IDs. A framework must not re-chunk them or replace their metadata.
- `Retriever` returns only `(unit_id, score)`. The harness resolves all text and
  provenance from its immutable per-arm catalog and rejects unknown IDs.
- `Reranker` receives the same RRF candidate budget in every arm and returns one
  score per candidate.
- The final context is selected with the same tokenizer and joined-token budget
  for every arm. Parent contexts are deduplicated by `context_id`.
- Every run records the requested/actual stage budgets, ranked unit IDs,
  retrieval evidence, generation-context evidence, article IDs, and a frozen
  configuration fingerprint.

This makes Haystack an optional orchestration and evaluation shell, not the
source of truth. A later Haystack adapter should convert Haystack `Document`
results to stable `SearchHit` values and must preserve `unit_id` in metadata.
Haystack can then handle pipeline wiring and evaluators without changing the
core experiment contract. No Haystack package is needed by the unit tests.

## Retrieval chain

For each arm, an isolated index is built over `unit.retrieval_text`:

1. fixed `bm25_top_k` candidates;
2. fixed `dense_top_k` candidates using the BGE-M3 dense adapter;
3. reciprocal-rank fusion to `fusion_top_k` using rank, not incomparable raw
   BM25/vector scores;
4. a BGE reranker adapter scores the fused candidates and retains
   `rerank_top_k`;
5. generation contexts are greedily packed under one fixed token budget.

The dependency-free BM25 is intentionally tokenizer-injected. Production
Chinese experiments must record and inject a frozen tokenizer; the whitespace
tokenizer exists only for tests and small English probes. The BGE-M3 and BGE
reranker adapters wrap already loaded backends and never download models.

## Reproducibility contract

`RetrievalConfig` freezes:

- backend provider, model, revision/checksum, and serializable parameters;
- candidate limits for BM25, dense, RRF, and reranker;
- RRF constant and route weights;
- final joined-context token budget and separator.

Its SHA-256 fingerprint must be identical across T0, T1, and C1. Corpus-builder
parameters and gold-data version remain separate manifest fields because they
define inputs rather than retrieval behavior.

## Deliberate exclusions

- HyDE is not in this first frozen chain. It should be an explicitly named
  ablation because it introduces an LLM call and another cache/model contract.
- This module does not load BGE checkpoints or call LM Studio.
- This module does not claim a quality result; it supplies a deterministic path
  for the later real-model and gold-QA experiment.
