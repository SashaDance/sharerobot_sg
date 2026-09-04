#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_HOST="${1:?usage: run_graph_verifier_variant.sh MANIFEST SOURCE_ROOT OUTPUT_ROOT CONFIG_CONTAINER}"
SOURCE_ROOT="${2:?source root required}"
OUTPUT_ROOT="${3:?output root required}"
CONFIG_CONTAINER="${4:?container config path required}"
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
    --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" \
    --config "$CONFIG_CONTAINER" --fail-fast
}

started="$(date +%s)"
batch prepare
"$RUN" start-qwen
"$RUN" wait-qwen
batch entities
"$RUN" stop-qwen
batch sam
"$RUN" start-qwen
"$RUN" wait-qwen
batch graph
"$RUN" stop-qwen
batch render
"$RUN" batch --stage validate --manifest "/manifests/$MANIFEST_NAME" \
  --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT" \
  --config "$CONFIG_CONTAINER" --without-da3 --fail-fast
"$RUN" review-index --manifest "/manifests/$MANIFEST_NAME" \
  --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT"
elapsed="$(( $(date +%s) - started ))"
echo "Verifier experiment completed in ${elapsed} seconds: $OUTPUT_ROOT"
