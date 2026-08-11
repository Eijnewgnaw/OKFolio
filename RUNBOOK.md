# Local Runbook

## 1. Create an isolated environment

```bash
cd "$HOME/OKFolio-Concept-Compiler-Experiment-20260810"
python3 -m venv .snapshot-venv
source .snapshot-venv/bin/activate
python -m pip install -r requirements.lock
```

The existing workstation environment can also validate the snapshot without
installing another copy:

```bash
cd "$HOME/OKFolio-Concept-Compiler-Experiment-20260810"
PYTHONPATH=. python -m pytest -q
```

## 2. Configure an OpenAI-compatible model

Keep provider settings in the shell or an ignored `.env` file:

```bash
export OPENAI_BASE_URL='<openai-compatible-v1-base-url>'
export OPENAI_MODEL='<model-id>'
export OPENAI_API_KEY='<key-if-required>'
```

The current local LM Studio runtime does not require an API key. Do not put a
real key or a deployment endpoint into a tracked file.

## 3. Inspect the saved formal checkpoint

```bash
cd "$HOME/OKFolio-Concept-Compiler-Experiment-20260810"
jq '{status,summary,configuration}' \
  data/agent-runs/public10-claim-review-formal-qwen36-v3-20260810/manifest.json
```

The master experiment data lives in the git-ignored `data/` directory at the
repo root (moved there from the old `~/kmpro-wiki-v15-data` location on
2026-08-11; the `.local-runtime` symlink mechanism was removed). A committed
restore copy is kept under `experiment-data/`.

## 4. Review command shape

The formal review entry point is:

```bash
PYTHONPATH=. python scripts/review_concept_claims.py \
  --source-run data/agent-runs/public10-local-qwen36-semantic-v2-20260809 \
  --seed-run data/agent-runs/public10-claim-review-formal-qwen36-v3-20260810 \
  --output-dir data/agent-runs/<new-versioned-run-id> \
  --structures-dir data/normalized-sources \
  --api-base "$OPENAI_BASE_URL" \
  --model "$OPENAI_MODEL" \
  --timeout 600 \
  --max-tokens 8192 \
  --coverage-max-tokens 16384 \
  --max-recompile-attempts 2 \
  --response-format json_schema \
  --send-chat-template-kwargs \
  --contract-stage-thinking \
  --known-source-anomaly '见色号'
```

Do not start this full command until sentence-batched Coverage in
`NEXT_STEPS.md` has passed its frozen probe. The current code remains useful
for tests, inspection and single-group reproduction, but the known large group
will otherwise reach the output ceiling again.

## 5. Materialize C1 after 332/332 pass

```bash
PYTHONPATH=. python scripts/materialize_c1_run.py \
  --source-run data/agent-runs/public10-local-qwen36-semantic-v2-20260809 \
  --review-run data/agent-runs/<complete-formal-run-id> \
  --output runtime/releases/<formal-c1-version>
```

The materializer intentionally rejects partial runs, selected-group probes,
withheld groups, provenance mismatches and incomplete Ref coverage.
