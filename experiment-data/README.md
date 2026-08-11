# experiment-data

Experimental run data for the OKFolio public-ten-book Concept experiment,
checked in so the experiment can be restored from GitHub and continued on a
single RTX 3090 machine.

## Layout

| Path | Contents |
| --- | --- |
| `source-run/` | The source compilation run `public10-local-qwen36-semantic-v2-20260809` (groups.json, compile_progress.json, source_progress.json, candidates.json, ref_validation.json, agent_trace.json, events.log, manifest.json, refine-checkpoints/). |
| `structures/` | The 10 `*.structure.json` normalized-source files of the public corpus. |
| `runs/` | Formal claim-review runs v3–v6. Each contains checkpoints/, manifest.json, events.log, review_queue.json, reviewed_compile_progress.json, claim_review_progress.json, source_snapshot.json. v6 (`public10-claim-review-formal-qwen3p6-remote-v6-nothinking-20260810`) carries the full 332-group checkpoint set. |
| `runs/v7-launcher/` | Sanitized launcher for the v7 repair run (see its header). |
| `probes/` | Probe runs v13 (`...-v13-batched-...`) and v14 (`...-v14-fixed-...`). |
| `overrides/` | 77 manually repaired draft overrides (one JSON per proposition) used by `--draft-override-dir`. |
| `tools/` | `extract_defects.py` and `defects_worklist_20260811.md` (generated 2026-08-11). |

## Origin

Copied on 2026-08-11 from the workstation snapshot
`OKFolio-Concept-Compiler-Experiment-20260810/.local-runtime/` (symlinks
dereferenced, `cp -RL`). The `.local-runtime/` tree itself is git-ignored and
never enters the repository. Source-run and structures were copied from
`normalized-sources` and `agent-runs` on the original machine; the original
files are read-only assets and were not modified.

## Sanitization record

- The private MinIO asset-server host in `asset_uri` / image-reference fields
  was replaced repo-wide: `192.168.8.209:9000` → `minio.internal:9000`
  (2,738 replacements across 26 files in source-run/, structures/, runs/
  v3–v6 and probes/ v13). The `kmpro-wiki-assets` bucket name, dataset name
  and all asset object keys are unchanged, so references still resolve to the
  same objects when an asset server is re-exposed under `minio.internal`.
- `source_snapshot.json` sha256/size entries in all six runs (v3/v4/v5/v6 and
  probes v13/v14) were recomputed against the sanitized source files and
  verified (`_verify_snapshot` on restore passes).
- Some seed checkpoints embed a `checkpoint_sha256` captured at creation time
  against the pre-sanitization bytes; those embedded hashes no longer match
  the sanitized files. This is expected and does not affect resume: the run
  verifier validates `source_snapshot.json` (updated), not per-checkpoint
  hashes.
- Audit before commit: zero occurrences of API keys (`06225a0a`,
  `Authorization`, `Bearer`, real `OPENAI_API_KEY=` values), zero private
  network addresses, in every committed file.

## Restoring on a 3090 machine

1. Clone the repository and install the Python dependencies
   (`requirements.lock` / `requirements-rag.lock`).
2. Point `scripts/review_concept_claims.py` at this data:
   `--source-run experiment-data/source-run`,
   `--seed-run experiment-data/runs/<v6-run>` (for v7),
   `--draft-override-dir experiment-data/overrides`.
3. Export the API environment (`OPENAI_BASE_URL` = local vLLM endpoint,
   `OPENAI_MODEL`, `OPENAI_API_KEY`, `NO_PROXY`) — see
   `experiment-data/runs/v7-launcher/run_v7.sh` for the full v7 command.
4. The `minio.internal:9000` asset references are dead on a fresh machine
   unless an asset server is re-exposed; they are image references inside
   text content and do not affect compilation or claim review.
