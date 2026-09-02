#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_HOST="${1:-$ROOT_DIR/manifests/pilot_11.json}"
SOURCE_ROOT="${2:-/datasets/sharerobot_planning_selected}"
OUTPUT_ROOT="${3:?usage: run_robot_tracking_experiment.sh [MANIFEST] [SOURCE_ROOT_CONTAINER] OUTPUT_ROOT_CONTAINER}"
MANIFEST_NAME="$(basename "$MANIFEST_HOST")"
RUN="$ROOT_DIR/run.sh"

if [[ ! -f "$MANIFEST_HOST" ]]; then
  echo "Manifest not found: $MANIFEST_HOST" >&2
  exit 2
fi

"$RUN" stop-qwen >/dev/null 2>&1 || true
"$RUN" robot-track-sam3 --manifest "/manifests/$MANIFEST_NAME" \
  --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT"
"$RUN" robot-track-robotseg --manifest "/manifests/$MANIFEST_NAME" \
  --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT"
"$RUN" robot-track-render --manifest "/manifests/$MANIFEST_NAME" \
  --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT"

echo "Robot tracking comparison completed: $OUTPUT_ROOT"
