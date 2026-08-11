# Reproducible three-arm RAG experiment

This runner compares the same frozen questions under the same retrieval and
generation budgets:

- **T0** — block-preserving fixed-size chunks;
- **T1** — heading-aware Parent-Child chunks;
- **C1** — accepted AgentWiki Concepts with ConceptRef provenance.

The orchestration code is provider-neutral. It never loads a model or embeds an
endpoint. A project-specific adapter supplies indexing, sparse/dense search,
reranking, token counting, and generation.

## Safety and fairness gates

1. C1 is imported only when `manifest.json` has `status=complete`,
   `acceptance.json` has `status=pass`, every Concept is publishable, and all
   ConceptRefs resolve to eligible source blocks.
2. The runner freezes the complete config and hashes the gold file, all
   structure sidecars, and all C1 inputs in `experiment.lock.json`. A changed
   input requires a new output directory.
3. T0, T1, and C1 share one `RetrievalConfig` fingerprint, the original query,
   candidate limits, RRF constants, rerank limit, context-token budget, and
   answer prompt contract.
4. HyDE defaults to `off`. The only other accepted value is `ablation`; it is
   generated once per question and reused across all three arms. HyDE results
   must never be mixed into the primary comparison.
5. Natural-language answer judging is not hidden in the runner. The adapter
   must return both answer text and an `AnswerPrediction` aligned to stable gold
   fact IDs and evidence atoms. Use human review or a separately calibrated
   structured judge for formal claims.

## Configuration

Paths are resolved relative to the JSON config file. Provider credentials must
come from the adapter's environment; never put them in this file.

```json
{
  "experiment_id": "public-corpus-primary-v1",
  "structures_dir": "../data/structures",
  "c1_run_dir": "../runs/agentwiki-complete",
  "gold_path": "../evaluation/gold.test.jsonl",
  "corpus": {
    "t0_max_chars": 1200,
    "t1_child_max_chars": 600,
    "t1_parent_max_chars": 4800
  },
  "retrieval": {
    "bm25": {
      "provider": "adapter-owned",
      "model": "bm25-tokenizer-id",
      "revision": "frozen-revision"
    },
    "dense": {
      "provider": "adapter-owned",
      "model": "embedding-model-id",
      "revision": "frozen-revision"
    },
    "reranker": {
      "provider": "adapter-owned",
      "model": "reranker-model-id",
      "revision": "frozen-revision"
    },
    "bm25_top_k": 50,
    "dense_top_k": 50,
    "fusion_top_k": 50,
    "rerank_top_k": 20,
    "context_token_budget": 8192,
    "rrf_k": 60
  },
  "hyde_mode": "off",
  "bootstrap": {
    "samples": 10000,
    "confidence": 0.95,
    "seed": 0
  },
  "adapter": {
    "factory": "your_package.rag_adapter:create_backend",
    "options": {
      "profile": "frozen-primary"
    }
  }
}
```

The factory receives `(adapter_options, output_dir)` and returns an object with
these methods:

```python
count_tokens(text)
prepare_index(corpus, index_dir)
create_bm25(corpus, index_dir)
create_dense(corpus, index_dir)
create_reranker(corpus)
generate(gold=..., arm=..., request=...)
generate_hyde(query)  # called only in explicit ablation mode
```

Search methods return stable unit IDs, never replacement text. `generate`
returns `GeneratedAnswer`; its prediction cites canonical atoms such as
`article-id:p012:b007`.

The bundled local backend needs no `prediction_aligner_factory` to generate.
It requests the frozen `okfolio.rag-answer.v1` JSON contract, parses
answer/refusal/citations/atomic claim candidates, and deterministically maps
each cited context page to canonical evidence atoms. It never maps prose to
Gold fact IDs.

An explicit aligner remains optional through `AlignedGenerationPipeline` or
`prediction_aligner_factory`. It must be a human-reviewed importer or an
independently calibrated judge. Set `prediction_alignment_status` explicitly
to `human_reviewed` or `independent_judge` before treating required/forbidden
fact metrics as final. Without that step, the score trace is
`provisional_structured`: semantic fact metrics, answer accuracy, and Joint
Success are `null`, while deterministic retrieval, refusal, and citation
diagnostics remain available. The answer-generating model must not grade its
own answer for a claimed independent result.

## Dry-run readiness

This command loads no adapter, contacts no model, and writes no output. It
validates gold data, the C1 completion/acceptance gate, all provenance, and all
three corpus audits.

```bash
python scripts/run_rag_experiment.py \
  --config path/to/experiment.json \
  --output path/to/new-run \
  --stage readiness
```

The report must say `status: ready`, `writes_performed: false`, and show passing
audits for T0, T1, and C1.

## Run and resume

```bash
python scripts/run_rag_experiment.py --config path/to/experiment.json --output path/to/new-run --stage index
python scripts/run_rag_experiment.py --config path/to/experiment.json --output path/to/new-run --stage retrieve
python scripts/run_rag_experiment.py --config path/to/experiment.json --output path/to/new-run --stage generate
python scripts/run_rag_experiment.py --config path/to/experiment.json --output path/to/new-run --stage score
```

`--stage all` executes the same four stages in order. Every command is safe to
repeat with the unchanged lock:

- indexing checkpoints each arm independently;
- retrieval, generation, and score traces checkpoint each
  `(question_id, arm)` row;
- duplicate or malformed trace keys fail closed;
- completed rows are reused without another provider request.

## Artifacts

```text
run/
├── experiment.lock.json       # frozen config and input hashes
├── manifest.json              # stage status
├── corpora/
│   ├── T0.json
│   ├── T1.json
│   └── C1.json                # audited immutable corpus snapshots
├── indices/
│   ├── progress.json          # per-arm index checkpoint
│   ├── T0/
│   ├── T1/
│   └── C1/
├── traces/
│   ├── retrieval.jsonl        # ranks, budgets, atoms, selected context IDs
│   ├── generation.jsonl       # answer, aligned prediction, usage, timing
│   └── scores.jsonl           # deterministic per-question scores
└── summary.json               # arm metrics and paired bootstrap intervals
```

The summary reports retrieval recall, complete-set recall, MRR, nDCG, context
precision, refusal behavior, citation diagnostics, and—only after independent
semantic alignment—answer accuracy and provenance-gated Joint Success. It also
reports paired bootstrap intervals for T1−T0 and C1−T0; semantic intervals stay
`null` while alignment is provisional and are meaningful only on a frozen,
independently reviewed test set.
