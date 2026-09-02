#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ENV="${UNIFIED_SGG_ENV:-/datasets/unified_scene_graph_pipeline.runtime.env}"
COMPOSE=(docker compose --project-directory "$ROOT_DIR" --env-file "$RUNTIME_ENV" -f "$ROOT_DIR/compose.yaml")
COMPOSE_TP2=(docker compose --project-directory "$ROOT_DIR" --env-file "$RUNTIME_ENV" -f "$ROOT_DIR/compose.yaml" -f "$ROOT_DIR/compose.tp2.yaml")

require_env() {
  if [[ ! -f "$RUNTIME_ENV" ]]; then
    echo "Missing runtime env: $RUNTIME_ENV" >&2
    exit 2
  fi
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
    "${COMPOSE[@]}" --profile tools build da3
    "${COMPOSE[@]}" --profile tools build robotseg
    ;;
  build-core)
    require_env
    "${COMPOSE[@]}" --profile tools build core
    ;;
  build-da3)
    require_env
    "${COMPOSE[@]}" --profile tools build da3
    ;;
  build-robotseg)
    require_env
    "${COMPOSE[@]}" --profile tools build robotseg
    ;;
  start-qwen)
    require_env
    "${COMPOSE[@]}" up -d qwen qwen-proxy
    ;;
  start-qwen-tp2)
    require_env
    gpu0_free="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')"
    if (( gpu0_free < 40000 )); then
      echo "Refusing TP=2 fallback: GPU 0 has only ${gpu0_free} MiB free (40000 required)" >&2
      exit 1
    fi
    "${COMPOSE[@]}" stop qwen qwen-proxy >/dev/null 2>&1 || true
    "${COMPOSE_TP2[@]}" up -d qwen qwen-proxy
    ;;
  wait-qwen)
    require_env
    set -a
    # shellcheck disable=SC1090
    source "$RUNTIME_ENV"
    set +a
    for attempt in $(seq 1 120); do
      if curl --silent --fail --max-time 3 -H "Authorization: Bearer ${INFERENCE_API_KEY}" http://127.0.0.1:8008/v1/models >/dev/null; then
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
    echo "SAM2.1 checkpoint downloaded and verified"
    ;;
  stop-qwen)
    require_env
    "${COMPOSE[@]}" stop qwen qwen-proxy
    ;;
  prepare|entities|sam|graph|trajectory|render|validate|batch|review-index)
    require_env
    command="$1"; shift
    core "$command" "$@"
    ;;
  da3)
    require_env
    shift
    "${COMPOSE[@]}" run --rm --no-deps -T da3 "$@"
    ;;
  robot-track-sam3)
    require_env
    shift
    "${COMPOSE[@]}" run --rm --no-deps -T --entrypoint python core \
      /pipeline/robot_tracking_compare.py segment --backend sam3 "$@"
    ;;
  robot-track-robotseg)
    require_env
    shift
    "${COMPOSE[@]}" run --rm --no-deps -T robotseg segment --backend robotseg "$@"
    ;;
  robot-track-render)
    require_env
    shift
    "${COMPOSE[@]}" run --rm --no-deps -T --entrypoint python core \
      /pipeline/robot_tracking_compare.py render "$@"
    ;;
  *)
    echo "Usage: $0 {build|build-core|build-da3|build-robotseg|download-qwen|download-sam2|start-qwen|start-qwen-tp2|wait-qwen|stop-qwen|prepare|entities|sam|graph|da3|trajectory|render|validate|batch|review-index|robot-track-sam3|robot-track-robotseg|robot-track-render} ..." >&2
    exit 2
    ;;
esac
