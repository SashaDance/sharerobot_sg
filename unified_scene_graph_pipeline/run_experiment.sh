#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_HOST="${1:?usage: run_experiment.sh MANIFEST SOURCE_ROOT_CONTAINER OUTPUT_ROOT_CONTAINER}"
SOURCE_ROOT="${2:?source root required}"
OUTPUT_ROOT="${3:?output root required}"
MANIFEST_NAME="$(basename "$MANIFEST_HOST")"
RUN="$ROOT_DIR/run.sh"

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

batch prepare
"$RUN" start-qwen
if ! "$RUN" wait-qwen; then
  "$RUN" start-qwen-tp2
  "$RUN" wait-qwen
fi
batch entities
"$RUN" stop-qwen
batch sam
"$RUN" start-qwen
if ! "$RUN" wait-qwen; then
  "$RUN" start-qwen-tp2
  "$RUN" wait-qwen
fi
batch graph
"$RUN" stop-qwen

while IFS= read -r relative_path; do
  "$RUN" da3 --scene "$OUTPUT_ROOT/$relative_path"
done < <(python3 - "$MANIFEST_HOST" <<'PY'
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
echo "Experiment completed: $OUTPUT_ROOT"
