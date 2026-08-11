# Ordered Work Plan

The work is deliberately split into evidence-preserving stages.

## 1. Remove the Coverage output bottleneck

- Keep one global Claim Contract for each Concept.
- Split only the draft-sentence audit into deterministic batches of at most 12
  consecutive sentences.
- Give every batch the same frozen Claim ledger and evidence provenance.
- Save each batch atomically so interruption resumes from the last completed
  batch.
- Merge batch results in code and run one final whole-Concept validation for
  omissions, contradictions, hard anchors, time qualifiers and scope.

This does not split a Concept into multiple Concepts and does not change the
332 global groups.

## 2. Verify the change before the full run

- Add unit tests for batch coverage, cross-batch claims and resume behavior.
- Run the complete test suite.
- Re-run the previously oversized Concept as a frozen single-group probe.
- Confirm that the source run and all previously accepted checkpoints remain
  unchanged.

## 3. Finish formal Concept compilation

- Start a new versioned formal run seeded from the current v3 checkpoints.
- Reuse the eight accepted groups.
- Process all 332 groups with checkpoints and no partial-publication mode.
- Re-open only deterministic `recompile_budget_exhausted` groups in subsequent
  repair runs; do not weaken factual or provenance gates.
- Finish only when all 445 Refs are represented and all 332 groups pass.

## 4. Materialize and audit C1

- Materialize `refs.json`, `concepts.json`, `acceptance.json`, Markdown Concepts
  and the formal manifest.
- Verify every Concept sentence and claim against Article, page and block
  provenance.
- Require 445/445 Ref coverage, 332/332 accepted Concepts, no review queue and
  no source-text anomaly in publishable output.

## 5. Build the formal local version

- Create a new immutable directory under the operator's home directory.
- Include the audited C1 assets, code, prompts, run manifest and integrity
  checksums.
- Exclude probes, failed histories, raw PDFs, model files, private endpoints and
  credentials.
- Build Bundle, Wiki and the V13-style Graph only after C1 passes.

## 6. Hold later experiments

BGE-M3, BGE Reranker, the T0/T1/C1 three-arm RAG comparison and the independent
LightRAG baseline remain paused until explicit instruction. Their code and
design notes may stay in the source snapshot, but they must not consume model
resources or alter the Concept run.
