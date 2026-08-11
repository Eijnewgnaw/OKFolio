# Public-10 Concept Compilation Snapshot

Snapshot date: 2026-08-10

This directory is a runnable source snapshot of the current OKFolio public
ten-book Concept experiment. It is not a completed C1 release and must not be
presented as one.

## Current state

| Stage | State | Count |
| --- | --- | ---: |
| MinerU parsing and structure normalization | complete | 10/10 books |
| ConceptRef discovery | complete | 445 accepted Refs |
| Global grouping | complete | 332 target Concept groups |
| Existing source-run drafts | partial | 146/332 groups |
| Former group-147 truncation probe | passed | 1/1 group |
| Formal Claim Review | partial | 8/332 groups accepted |
| Formal C1 materialization | not started | 0 |
| Bundle, Wiki and Graph | not started | 0 |
| Three-arm RAG and LightRAG evaluation | paused by instruction | 0 |

The number 147 belonged to the earlier draft-production pass. The formal Claim
Review is a separate pass over all 332 groups, so its counter starts again from
the beginning. No Ref discovery or global grouping has been lost.

## Current blocker

One large Concept produces a sentence-level Coverage response that reaches the
model output ceiling at both 8,192 and 16,384 tokens. The saved checkpoints are
valid; the next implementation change is deterministic sentence-batched
Coverage, not another increase of the output limit.

## Data boundary

The source tree contains no API key, SSH credential, model weights, raw PDF,
MinerU cache, private server endpoint, or copied MinIO URL. The git-ignored
master data tree `data/` (consolidated under the repo root 2026-08-11,
replacing the former `.local-runtime/` symlink mechanism) is not included in
the portable archive or checksum manifest.

Source Git commit at snapshot time:

```text
ccdd2e5a1343341d03feadaf2587c6d216e51c38
```

The snapshot also contains uncommitted, tested experiment work newer than that
commit. `MANIFEST.sha256` is the authoritative file-integrity list.
