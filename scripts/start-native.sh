#!/usr/bin/env bash
# NVIDIA GLM FP4: native CUDA ranks rendezvous through NCCL_COMM_ID, not --tp.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source scripts/config.sh
source scripts/native-config.sh
source scripts/nodes.sh
native_check
mode=start
[[ "${1:-}" != restart ]] || { mode=restart; shift; }
if [[ "${1:-}" == help || "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: ./start.sh [restart] [native tensorfold serve flags]'
  echo 'DRY_RUN=1 prints rank commands; ./stop.sh stops both ranks.'
  exit 0
fi
args=(serve "$NATIVE_MODEL_ARG" --context "$CONTEXT" --parallel "$PARALLEL"
      --max-tokens "$MAX_TOKENS" --name "$SERVED_NAME" --host "$HOST" --port "$PORT"
      --reasoning-effort "${NATIVE_REASONING_EFFORT:-high}")
if [[ "$THINKING" == 1 ]]; then args+=(--thinking); else args+=(--no-thinking); fi
[[ "$NATIVE_VISION" != 1 ]] || args+=(--vision)
args+=("$@")
need_workers
detect_links
cache_worker=$(worker_hf_cache 1)
run=(docker run -d --name "$CONTAINER_NAME" --gpus all --network host --ipc host
     --ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband
     --security-opt seccomp=unconfined --cap-add IPC_LOCK
     -e TF_GLM_WORLD=2 -e "NCCL_COMM_ID=$HEAD_ADDR:$MASTER_PORT"
     -e "TENSORFOLD_MEMORY_RESERVE_GIB=$MEMORY_RESERVE_GIB"
     -e "TF_ROCE=${NATIVE_ROCE:-1}" -e NCCL_CUMEM_HOST_ENABLE=0 -e NCCL_CUMEM_ENABLE=0
     -e NCCL_RUNTIME_CONNECT=0 -e "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
     --log-opt max-size=50m --log-opt max-file=3)
declare -a net0 net1
read -r -a net0 <<< "$(nccl_env "$HEAD_DEV" "$HEAD_HCAS" "$HEAD_GID")"
read -r -a net1 <<< "$(nccl_env "$WORKER_DEV" "$WORKER_HCAS" "$WORKER_GID")"
rank0=("${run[@]}" "${net0[@]}" -e TF_GLM_RANK=0 -v "$HF_CACHE:/root/.cache/huggingface:ro" "$NATIVE_IMAGE" "${args[@]}")
rank1=("${run[@]}" "${net1[@]}" -e TF_GLM_RANK=1 -v "$cache_worker:/root/.cache/huggingface:ro" "$NATIVE_IMAGE" "${args[@]}")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'rank 0: '; printf '%q ' "${rank0[@]}"; printf '\nrank 1 on %s: ' "$WORKER"
  printf '%q ' "${rank1[@]}"; printf '\n'; exit 0
fi
mkdir -p "$NATIVE_STATE_DIR"
exec 8>"$NATIVE_STATE_DIR/start.lock"
flock -n 8 || die "native start is already running"
running() { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }
if [[ "$mode" == start ]] && running; then
  die "$CONTAINER_NAME already runs here; use ./start.sh restart to replace it"
fi
bash scripts/prepare-native.sh
if [[ "$mode" == restart ]]; then bash stop.sh; fi
for rank in 0 1; do
  if (( rank == 0 )); then
    if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
      running && die "rank 0 is already running"
      save_log "$LOG_DIR" 0 "$CONTAINER_NAME" "$LOG_KEEP"
      docker rm "$CONTAINER_NAME" >/dev/null
    fi
  elif worker 1 docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    [[ "$(worker 1 docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != true ]] || die "rank 1 is already running; use restart"
    worker_save_log 1
    worker 1 docker rm "$CONTAINER_NAME" >/dev/null
  fi
done
api_host="$HOST"; [[ "$HOST" != 0.0.0.0 && "$HOST" != :: ]] || api_host=127.0.0.1
url="http://$api_host:$PORT"
if curl -s --max-time 2 "$url/health" >/dev/null; then die "port $PORT is already in use"; fi
printf -v remote '%q ' "${rank1[@]}"
log "Starting rank 1 on $WORKER, then rank 0; native weights may take several minutes to load"
worker 1 "$remote"
"${rank0[@]}"
deadline=$((SECONDS + ${WAIT_TIMEOUT:-1800}))
while (( SECONDS < deadline )); do
  if curl -fsS --max-time 3 "$url/v1/models" >/dev/null 2>&1; then
    curl -fsS --max-time 180 "$url/v1/chat/completions" -H 'Content-Type: application/json' \
      --data "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: TensorFold TP2 ready\"}],\"max_tokens\":128,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
    printf '\n'; log "Ready: $url/v1 ($SERVED_NAME, NVIDIA FP4, TP2)"; exit 0
  fi
  if ! running || [[ "$(worker 1 docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" != true ]]; then
    docker logs --tail 60 "$CONTAINER_NAME" >&2 || true
    worker 1 docker logs --tail 60 "$CONTAINER_NAME" >&2 || true
    die "a native rank exited; inspect both container logs"
  fi
  sleep 10
done
die "startup timed out; inspect docker logs $CONTAINER_NAME on both Sparks"
