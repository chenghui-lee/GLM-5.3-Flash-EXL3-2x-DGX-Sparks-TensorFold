#!/usr/bin/env bash
# Prepare the official native TensorFold release and NVIDIA GLM FP4 on both Sparks.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source scripts/config.sh
source scripts/native-config.sh
source scripts/nodes.sh
native_check
rebuild=0
for arg in "$@"; do
  case "$arg" in --rebuild) rebuild=1 ;; *) die "unknown prepare argument: $arg" ;; esac
done
mkdir -p "$NATIVE_STATE_DIR" "$HF_CACHE/hub"
exec 9>"$NATIVE_STATE_DIR/prepare.lock"
flock -n 9 || die "native preparation is already running"
need_workers
detect_links
cache_worker=$(worker_hf_cache 1)
worker 1 "mkdir -p $(printf %q "$cache_worker/hub")"
log "Preparing TensorFold $NATIVE_VERSION for $MODEL_ID at TP2"
have=$(docker image inspect -f '{{index .Config.Labels "tf.native.sha256"}}' "$NATIVE_IMAGE" 2>/dev/null || true)
if [[ "$have" != "$NATIVE_SHA256" || "$rebuild" == 1 ]]; then
  ctx=$(mktemp -d "$NATIVE_STATE_DIR/build.XXXXXX")
  archive="$NATIVE_STATE_DIR/$NATIVE_ARCHIVE"
  if ! printf '%s  %s\n' "$NATIVE_SHA256" "$archive" | sha256sum -c - >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$archive" "https://github.com/ashhart/TensorFold/releases/download/v$NATIVE_VERSION/$NATIVE_ARCHIVE"
  fi
  printf '%s  %s\n' "$NATIVE_SHA256" "$archive" | sha256sum -c -
  cp "$archive" "$ctx/release.tar.gz"
  docker build -f scripts/Dockerfile.native -t "$NATIVE_IMAGE" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg NATIVE_VERSION="$NATIVE_VERSION" \
    --build-arg NATIVE_SHA256="$NATIVE_SHA256" "$ctx"
fi
local_id=$(image_ident "$NATIVE_IMAGE")
if [[ "$(worker_image_ident 1 "$NATIVE_IMAGE")" != "$local_id" ]]; then
  log "Copying native image to $(worker_link_host 1)"
  docker save "$NATIVE_IMAGE" | worker_link 1 docker load
fi
[[ "$(worker_image_ident 1 "$NATIVE_IMAGE")" == "$local_id" ]] || die "native image differs between ranks"
snap="$HF_CACHE/$NATIVE_MODEL_SUB"
if ! python3 tools/verify_native_checkpoint.py "$snap" >/dev/null 2>&1; then
  log "Downloading pinned NVIDIA checkpoint (~177 GB, resumable)"
  HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}" \
    hf download "$MODEL_ID" --revision "$MODEL_REVISION" --cache-dir "$HF_CACHE/hub" --max-workers "${HF_DOWNLOAD_WORKERS:-8}"
fi
python3 tools/verify_native_checkpoint.py "$snap"
repo="models--${MODEL_ID//\//--}"
log "Synchronizing checkpoint to worker over $WORKER_ADDR"
worker 1 "mkdir -p $(printf %q "$cache_worker/hub/$repo")"
# hf >=1.31 may link repo blobs into the cache-wide blob store. Materialize
# those external links while preserving snapshot links inside this repo.
rsync -a --copy-unsafe-links --info=progress2 -e 'ssh -o BatchMode=yes -o ConnectTimeout=10' \
  "$HF_CACHE/hub/$repo/" "$(worker_link_host 1):$cache_worker/hub/$repo/"
worker 1 "python3 - $(printf %q "$cache_worker/$NATIVE_MODEL_SUB")" < tools/verify_native_checkpoint.py
log "Native TensorFold image and pinned NVIDIA checkpoint ready on both Sparks"
