#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
DATA_DIR="${DATA_DIR:-${APP_DIR}/runtime/data}"
INBOX_DIR="${DATA_DIR}/inbox"
SOURCES_DIR="${DATA_DIR}/sources"

mkdir -p "${INBOX_DIR}" "${SOURCES_DIR}" "${SOURCES_DIR}/images"

shopt -s nullglob
for source in "${INBOX_DIR}"/*.md; do
  destination="${SOURCES_DIR}/$(basename "${source}")"
  if [[ -f "${destination}" ]] && cmp -s "${source}" "${destination}"; then
    continue
  fi
  temporary="${destination}.tmp"
  cp "${source}" "${temporary}"
  mv "${temporary}" "${destination}"
  echo "Source updated: $(basename "${source}")"
done

for source in "${INBOX_DIR}"/*.pdf; do
  echo "MinerU deferred: $(basename "${source}")"
done

compile_output_file="$(mktemp)"
cleanup() {
  rm -f "${compile_output_file}"
}
trap cleanup EXIT

set +e
python3 "${APP_DIR}/scripts/compile_and_enrich.py" | tee "${compile_output_file}"
compile_status=${PIPESTATUS[0]}
set -e
if [[ ${compile_status} -ne 0 ]]; then
  exit "${compile_status}"
fi

if grep -Eq 'compiled=0([[:space:]]|$)' "${compile_output_file}"; then
  echo "No wiki changes; index and site build skipped"
  exit 0
fi

python3 "${APP_DIR}/scripts/rebuild_index.py"
bash "${APP_DIR}/scripts/build_site.sh"
