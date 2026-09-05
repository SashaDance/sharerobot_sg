#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_A="${1:?usage: run_full_resumable_gpu1.sh MANIFEST_A MANIFEST_B SOURCE_ROOT_CONTAINER OUTPUT_ROOT_CONTAINER}"
MANIFEST_B="${2:?second manifest required}"
SOURCE_ROOT="${3:?source root required}"
OUTPUT_ROOT="${4:?output root required}"
RUNTIME_ENV="${UNIFIED_SGG_ENV:-/datasets/unified_scene_graph_pipeline.runtime.env}"
RUN="$ROOT_DIR/run.sh"
MAX_BATCH_PASSES="${MAX_BATCH_PASSES:-5}"
MAX_DA3_PASSES="${MAX_DA3_PASSES:-3}"
STARTED="$(date +%s)"

for manifest in "$MANIFEST_A" "$MANIFEST_B"; do
  if [[ ! -f "$manifest" ]]; then
    echo "Manifest not found: $manifest" >&2
    exit 2
  fi
done
if [[ "$OUTPUT_ROOT" != /runs/* ]]; then
  echo "OUTPUT_ROOT must be below /runs so resumable reports can be inspected on the host" >&2
  exit 2
fi
if [[ ! -f "$RUNTIME_ENV" ]]; then
  echo "Runtime env not found: $RUNTIME_ENV" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$RUNTIME_ENV"
set +a
HOST_OUTPUT_ROOT="${RUN_ROOT:?RUN_ROOT is required}/${OUTPUT_ROOT#/runs/}"
MANIFESTS=("$MANIFEST_A" "$MANIFEST_B")
SHARDS=("shard_gpu2" "shard_gpu3")

stop_qwen() {
  "$RUN" stop-qwen >/dev/null 2>&1 || true
}
trap stop_qwen EXIT

container_manifest() {
  printf '/manifests/%s' "$(basename "$1")"
}

batch_retry() {
  local stage="$1" manifest="$2" shard="$3"
  local output="$OUTPUT_ROOT/$shard"
  local host_output="$HOST_OUTPUT_ROOT/$shard"
  local pass failures
  for ((pass = 1; pass <= MAX_BATCH_PASSES; pass++)); do
    if [[ "$stage" == "entities" || "$stage" == "graph" ]]; then
      if ! "$RUN" wait-qwen; then
        "$RUN" start-qwen
        "$RUN" wait-qwen
      fi
    fi
    echo "BATCH_BEGIN stage=$stage shard=$shard pass=$pass"
    "$RUN" batch --stage "$stage" --manifest "$(container_manifest "$manifest")" \
      --source-root "$SOURCE_ROOT" --output-root "$output"
    failures="$(python3 - "$host_output/batch_${stage}_report.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["failure_count"])
PY
)"
    echo "BATCH_END stage=$stage shard=$shard pass=$pass failures=$failures"
    if [[ "$failures" == "0" ]]; then
      return 0
    fi
  done
  echo "Stage $stage still has $failures failed scenes in $shard after $MAX_BATCH_PASSES passes" >&2
  return 1
}

start_qwen_gpu1() {
  "$RUN" start-qwen
  "$RUN" wait-qwen
}

run_da3_shard() {
  local manifest="$1" shard="$2"
  local output="$OUTPUT_ROOT/$shard"
  local retry_dir="$HOST_OUTPUT_ROOT/$shard/.da3_retries"
  local current="$retry_dir/current.txt" next="$retry_dir/next.txt"
  local pass relative failures
  mkdir -p "$retry_dir"
  python3 - "$manifest" > "$current" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
for item in value.get("episodes", value.get("scenes", [])):
    print(item["relative_path"])
PY
  for ((pass = 1; pass <= MAX_DA3_PASSES; pass++)); do
    : > "$next"
    failures=0
    echo "DA3_PASS_BEGIN shard=$shard pass=$pass scenes=$(wc -l < "$current")"
    while IFS= read -r relative; do
      [[ -n "$relative" ]] || continue
      if ! "$RUN" da3 --scene "$output/$relative"; then
        printf '%s\n' "$relative" >> "$next"
        failures=$((failures + 1))
      fi
    done < "$current"
    echo "DA3_PASS_END shard=$shard pass=$pass failures=$failures"
    if ((failures == 0)); then
      rm -f "$current" "$next"
      return 0
    fi
    mv "$next" "$current"
  done
  echo "DA3 still has $failures failed scenes in $shard after $MAX_DA3_PASSES passes" >&2
  return 1
}

for index in 0 1; do
  batch_retry prepare "${MANIFESTS[$index]}" "${SHARDS[$index]}"
done

start_qwen_gpu1
for index in 0 1; do
  batch_retry entities "${MANIFESTS[$index]}" "${SHARDS[$index]}"
done
stop_qwen

for index in 0 1; do
  batch_retry sam "${MANIFESTS[$index]}" "${SHARDS[$index]}"
done

start_qwen_gpu1
for index in 0 1; do
  batch_retry graph "${MANIFESTS[$index]}" "${SHARDS[$index]}"
done
stop_qwen

for index in 0 1; do
  run_da3_shard "${MANIFESTS[$index]}" "${SHARDS[$index]}"
done

for stage in trajectory render validate; do
  for index in 0 1; do
    batch_retry "$stage" "${MANIFESTS[$index]}" "${SHARDS[$index]}"
  done
done

for index in 0 1; do
  "$RUN" review-index --manifest "$(container_manifest "${MANIFESTS[$index]}")" \
    --source-root "$SOURCE_ROOT" --output-root "$OUTPUT_ROOT/${SHARDS[$index]}"
done

echo "Full resumable experiment completed in $(( $(date +%s) - STARTED )) seconds: $OUTPUT_ROOT"
