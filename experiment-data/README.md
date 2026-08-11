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
never enters the repository; it symlinks to the master data directory
`~/kmpro-wiki-v15-data/` (moved there from the original Codex workspace on
2026-08-11). Source-run and structures were copied from `normalized-sources`
and `agent-runs` in that master directory; the original files are read-only
assets and were not modified.

## Sensitivity decision record

- The only private-infrastructure metadata in this tree is the MinIO asset
  endpoint `192.168.8.209:9000` (bucket `kmpro-wiki-assets`), which appears
  in `asset_uri` / image-reference fields. A full-tree scan found no other
  private addresses (no 10.x, no 172.16–31.x, no other 192.168.x) and no
  credential material of any kind.
- Decision (2026-08-11): the underlying corpus is publicly available PDFs and
  the structure/run files are MinerU parse output and review artifacts over
  that public content. The endpoint is internal infrastructure metadata, not
  classified content, so it is kept **byte-for-byte as produced** to guarantee
  that a 3090 restore reproduces the experiment exactly (snapshot hashes in
  each run's `source_snapshot.json` remain valid and `_verify_snapshot`
  passes without recomputation).
- Credential audit before commit: zero occurrences of API key material in
  every committed file. The v7 launcher reads credentials exclusively from
  environment variables.

## Restoring on a 3090 machine

1. Clone the repository and install the Python dependencies
   (`requirements.lock` / `requirements-rag.lock`).
2. Serve the model locally with vLLM and export the API environment
   (`OPENAI_BASE_URL` = the local vLLM endpoint, `OPENAI_MODEL`,
   `OPENAI_API_KEY`, `NO_PROXY`).
3. Continue or rerun review against this data:
   - Resume a frozen run: `python3 scripts/review_concept_claims.py
     --source-run experiment-data/source-run --output-dir
     experiment-data/runs/<frozen-run> --resume …`
   - Launch the v7 repair run: `experiment-data/runs/v7-launcher/run_v7.sh`
     (resolves paths relative to the repo root and inherits the API
     environment).
4. The `192.168.8.209:9000` asset references are image pointers inside text
   content; they are not needed for compilation or claim review and will not
   resolve from a new machine unless an asset server is re-exposed.
