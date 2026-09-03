#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_HOST="${1:?usage: run_experiment_gpu1.sh MANIFEST SOURCE_ROOT_CONTAINER OUTPUT_ROOT_CONTAINER}"
SOURCE_ROOT="${2:?source root required}"
OUTPUT_ROOT="${3:?output root required}"
MANIFEST_NAME="$(basename "$MANIFEST_HOST")"
RUN="$ROOT_DIR/run.sh"
STARTED="$(date +%s)"

if [[ ! -f "$MANIFEST_HOST" ]]; then
  echo "Manifest not found: $MANIFEST_HOST" >&2
  exit 2
fi

stop_qwen() {
  "$RUN" stop-qwen >/dev/null 2>&1 || true
}
trap stop_qwen EXIT

batch() {
  "$RUN" batch --stage "$1" --manifest "/manifests/$MANIFEST_NAME" \
    --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" --fail-fast
}

start_qwen_gpu1() {
  "$RUN" start-qwen
  "$RUN" wait-qwen
}

batch prepare
start_qwen_gpu1
batch entities
stop_qwen
batch sam
start_qwen_gpu1
batch graph
stop_qwen

while IFS= read -r -u 3 relative_path; do
  "$RUN" da3 --scene "$OUTPUT_ROOT/$relative_path"
done 3< <(python3 - "$MANIFEST_HOST" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for item in value.get("episodes", value.get("scenes", [])):
    print(item["relative_path"])
PY
)

batch trajectory
batch render
batch validate
"$RUN" review-index --manifest "/manifests/$MANIFEST_NAME" --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT"
echo "Experiment completed in $(($(date +%s) - STARTED)) seconds: $OUTPUT_ROOT"
