#!/usr/bin/env bash
# v7 repair run launcher (WRITE-ONLY PREP — not executed).
#
# Semantics of v7 (vs the frozen v6 formal run):
#   - --seed-run v6: inherit every audited PASS checkpoint from v6 (accepted
#     groups are copied verbatim, seed_provenance recorded). v6's failed
#     groups are skipped as seeds and re-run under the REPAIR contract prompt;
#     v6's budget-exhausted human_review groups are re-opened as "running".
#   - --contract-prompt agent_claim_contract_repair.md + --recompile-prompt
#     agent_recompile_repair.md: the strengthened prompts change v7's
#     prompt_sha256 for contract/recompile vs v6's, so the seed-prompt
#     relaxation (contract_prompt_relaxed / recompile_prompt_relaxed) will be
#     recorded in v7's manifest instead of rejecting the seed.
#   - --draft-override-dir experiment-data/overrides: manually repaired drafts.
#     For a re-opened human_review seed the override replaces the inherited
#     draft and clears the inherited Coverage so the repaired draft is audited
#     from scratch; for a failed group the override is used when no checkpoint
#     draft exists. Overrides never touch the source run snapshot.
#   - --output-dir ...-v7-repair-<date>: a fresh run directory (v7 does not
#     resume v6; if v7 itself is interrupted, re-run with `--resume` appended
#     to this same command and the same --seed-run/--draft-override-dir).
#
# API environment: OPENAI_BASE_URL / OPENAI_MODEL / OPENAI_API_KEY / NO_PROXY
# are injected from the environment and NEVER committed. On the 3090 restore
# machine point OPENAI_BASE_URL at the local vLLM endpoint, e.g.
#   export OPENAI_BASE_URL="http://<vllm-host>:<port>/v1"
# (the original private gateway address must not be stored in this repo).
# Copy the values from the shell that launched v6, e.g. with:
#   env | grep -E '^(OPENAI_BASE_URL|OPENAI_MODEL|OPENAI_API_KEY|NO_PROXY)='
set -euo pipefail

# Repo root: this launcher lives at experiment-data/runs/v7-launcher/.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

EXPERIMENT_DATA="$ROOT/experiment-data"
SOURCE_RUN="$EXPERIMENT_DATA/source-run"
V6_RUN="public10-claim-review-formal-qwen3p6-remote-v6-nothinking-20260810"
V7_OUT="public10-claim-review-formal-qwen3p6-remote-v7-repair-$(date +%Y%m%d)"

# ---- guard: v7 should inherit a COMPLETED v6 ----
V6_MANIFEST="$EXPERIMENT_DATA/runs/$V6_RUN/manifest.json"
V6_STATUS="$(python3 -c "import json,sys; print(json.load(open('$V6_MANIFEST')).get('status',''))" 2>/dev/null || echo '')"
if [[ "$V6_STATUS" == "running" && "${FORCE:-}" != "1" ]]; then
    echo "error: v6 is still running (status=running). Launching v7 now would freeze" >&2
    echo "  v6's accepted set at its current position and miss later accepted groups." >&2
    echo "  Wait for v6 to complete, or set FORCE=1 to proceed anyway." >&2
    exit 2
fi

# ---- environment (fill in from v6's launch environment) ----
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export OPENAI_MODEL="${OPENAI_MODEL:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export NO_PROXY="${NO_PROXY:-}"
if [[ -z "${OPENAI_BASE_URL}" || -z "${OPENAI_API_KEY}" ]]; then
    echo "error: OPENAI_BASE_URL and OPENAI_API_KEY must be set" >&2
    echo "  (copy from v6's launch env; see header comment)" >&2
    exit 2
fi

# ---- v7 run (same parameters as v6, with repair overrides) ----
PYTHONPATH="$ROOT" nohup python3 "$ROOT/scripts/review_concept_claims.py" \
    --source-run "$SOURCE_RUN" \
    --seed-run "$EXPERIMENT_DATA/runs/$V6_RUN" \
    --output-dir "$EXPERIMENT_DATA/runs/$V7_OUT" \
    --contract-prompt "$ROOT/prompts/agent_claim_contract_repair.md" \
    --coverage-prompt "$ROOT/prompts/agent_claim_coverage.md" \
    --compile-prompt "$ROOT/prompts/compile.md" \
    --recompile-prompt "$ROOT/prompts/agent_recompile_repair.md" \
    --draft-override-dir "$EXPERIMENT_DATA/overrides" \
    --api-base "$OPENAI_BASE_URL" \
    --model "$OPENAI_MODEL" \
    --api-key-env OPENAI_API_KEY \
    --timeout 600 \
    --max-tokens 8192 \
    --coverage-max-tokens 16384 \
    --max-recompile-attempts 2 \
    --coverage-batch-size 12 \
    --response-format json_schema \
    --send-chat-template-kwargs \
    --known-source-anomaly '见色号' \
    > /tmp/v7-run.log 2>&1 &
echo $! > /tmp/v7-run.pid
echo "v7 launched pid=$(cat /tmp/v7-run.pid) log=/tmp/v7-run.log"
