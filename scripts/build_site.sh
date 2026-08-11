#!/usr/bin/env bash
set -euo pipefail

python3 /app/scripts/generate_graph.py
python3 -m okfolio.agentwiki.site \
  --data-dir "${DATA_DIR:-/app/runtime/data}" \
  --config-file /app/mkdocs.yml
