#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_HOST="${1:?usage: run_experiment_dual_gpu.sh MANIFEST SOURCE_ROOT_CONTAINER OUTPUT_ROOT_CONTAINER}"
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

started="$(date +%s)"
batch prepare
"$RUN" start-qwen-gpu0
"$RUN" wait-qwen

# Entity concept extraction runs on GPU 0 without frame-level boxes. SAM3 runs
# on GPU 1 and calls the still-resident Qwen server only for mask-track pruning.
batch entities
batch sam
batch graph

# Qwen remains resident on GPU 0 while DA3 uses GPU 1, avoiding a second model
# load. It is stopped after the GPU stages by the EXIT trap.
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
"$RUN" review-index --manifest "/manifests/$MANIFEST_NAME" \
  --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT"
elapsed="$(( $(date +%s) - started ))"
echo "Experiment completed in ${elapsed} seconds: $OUTPUT_ROOT"
