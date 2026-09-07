#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ENV="${SEGMENTATION_ENV:-/datasets/goal_guided_segmentation.runtime.env}"
COMPOSE=(docker compose --project-directory "$ROOT_DIR" --env-file "$RUNTIME_ENV" -f "$ROOT_DIR/compose.yaml")

require_env() {
  if [[ ! -f "$RUNTIME_ENV" ]]; then
    echo "Missing runtime env: $RUNTIME_ENV" >&2
    exit 2
  fi
  local mode
  mode="$(stat -c '%a' "$RUNTIME_ENV")"
  if [[ "$mode" != "600" ]]; then
    echo "Runtime env must have mode 600, got $mode" >&2
    exit 2
  fi
}

core() {
  "${COMPOSE[@]}" run --rm --no-deps core "$@"
}

case "${1:-}" in
  build)
    require_env
    "${COMPOSE[@]}" --profile tools build core
    ;;
  start-qwen)
    require_env
    "${COMPOSE[@]}" up -d qwen qwen-proxy
    ;;
  wait-qwen)
    require_env
    set -a
    # shellcheck disable=SC1090
    source "$RUNTIME_ENV"
    set +a
    for attempt in $(seq 1 120); do
      if curl --silent --fail --max-time 3 \
        -H "Authorization: Bearer ${INFERENCE_API_KEY}" \
        "http://127.0.0.1:${QWEN_HOST_PORT:-8008}/v1/models" >/dev/null; then
        echo "Qwen is ready"
        exit 0
      fi
      if (( attempt > 2 )) && [[ "$("${COMPOSE[@]}" ps --status running -q qwen | wc -l)" == "0" ]]; then
        echo "Qwen container exited before becoming ready" >&2
        exit 1
      fi
      sleep 5
    done
    echo "Qwen did not become ready in 10 minutes" >&2
    exit 1
    ;;
  stop-qwen)
    require_env
    "${COMPOSE[@]}" stop qwen qwen-proxy
    ;;
  download-qwen)
    require_env
    set -a
    # shellcheck disable=SC1090
    source "$RUNTIME_ENV"
    set +a
    mkdir -p "${MODEL_ROOT}/qwen3.8-27b"
    docker run --rm --entrypoint python3 --env-file "$RUNTIME_ENV" \
      -v "${MODEL_ROOT}:/models" -v "$ROOT_DIR/model_download.py:/download.py:ro" \
      vllm/vllm-openai:v0.24.0 /download.py
    ;;
  download-sam2)
    require_env
    set -a
    # shellcheck disable=SC1090
    source "$RUNTIME_ENV"
    set +a
    mkdir -p "${MODEL_ROOT}"
    target="${MODEL_ROOT}/sam2.1_hiera_large.pt"
    expected="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
    if [[ -f "$target" ]] && [[ "$(sha256sum "$target" | awk '{print $1}')" == "$expected" ]]; then
      echo "SAM2.1 checkpoint already present and verified"
      exit 0
    fi
    docker run --rm --user 0:0 -v "${MODEL_ROOT}:/models" curlimages/curl:8.12.1 \
      -fL --retry 3 --output /models/.sam2.1_hiera_large.pt.tmp \
      https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt
    actual="$(sha256sum "${MODEL_ROOT}/.sam2.1_hiera_large.pt.tmp" | awk '{print $1}')"
    if [[ "$actual" != "$expected" ]]; then
      echo "SAM2.1 checkpoint hash mismatch: $actual" >&2
      exit 1
    fi
    mv "${MODEL_ROOT}/.sam2.1_hiera_large.pt.tmp" "$target"
    ;;
  prepare|entities|sam|batch)
    require_env
    command="$1"
    shift
    core "$command" "$@"
    ;;
  *)
    echo "Usage: $0 {build|download-qwen|download-sam2|start-qwen|wait-qwen|stop-qwen|prepare|entities|sam|batch}" >&2
    exit 2
    ;;
esac
