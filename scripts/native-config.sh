# Sourced after config.sh. Native CUDA GLM uses environment ranks and embedded
# kernels; the Python engine's EXL3, dense, DFlash2 and KV controls do not apply.
NATIVE_VERSION="${NATIVE_VERSION:-1.1.1}"
NATIVE_SHA256="${NATIVE_SHA256:-2e0103a544e64354bfec65bd8a052ede17aa9be1f6a56d8e37c868af9139cb39}"
NATIVE_IMAGE="${NATIVE_IMAGE:-tensorfold-glm53:native-${NATIVE_VERSION}}"
NATIVE_VISION="${NATIVE_VISION:-0}"
NATIVE_STATE_DIR="${NATIVE_STATE_DIR:-$STATE_DIR/native}"
NATIVE_ARCHIVE="tensorfold-${NATIVE_VERSION}-linux-aarch64.tar.gz"
NATIVE_MODEL_SUB="hub/models--${MODEL_ID//\//--}/snapshots/$MODEL_REVISION"
NATIVE_MODEL_ARG="/root/.cache/huggingface/$NATIVE_MODEL_SUB"
native_check() {
  [[ "$TP" == 2 ]] || die "NVIDIA GLM FP4 deployment requires TP=2"
  [[ "$MODEL_REVISION" =~ ^[0-9a-f]{40}$ ]] || die "MODEL_REVISION must be a pinned Hugging Face commit"
  [[ "$NATIVE_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "invalid NATIVE_VERSION"
  [[ "$NATIVE_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "invalid NATIVE_SHA256"
  [[ "$NATIVE_VISION" =~ ^[01]$ ]] || die "NATIVE_VISION must be 0 or 1"
  [[ "$WORKER_WEIGHTS" == copy ]] || die "native deployment currently requires WORKER_WEIGHTS=copy"
}
