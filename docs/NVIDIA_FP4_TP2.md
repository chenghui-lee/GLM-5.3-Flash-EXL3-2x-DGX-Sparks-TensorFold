# NVIDIA GLM FP4 on two DGX Sparks

The local `.env` selects `nvidia/GLM-5.3-Flash-FP4-Local` at revision
`76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`. `./start.sh` and
`scripts/prepare.sh` dispatch to the native deployment for this model. Other model
IDs continue using the existing Python/EXL3 recipe.

The native path packages the official TensorFold **1.1.1 ARM64 release**, checks
its pinned SHA256, and reuses the NVIDIA PyTorch base image for NCCL and RDMA
libraries. Both nodes get the identical image and pinned checkpoint. Preparation
verifies every indexed tensor and shard extent before serving.

```sh
./start.sh                   # prepare, start worker/head, wait and smoke test
./start.sh restart           # restart both ranks
DRY_RUN=1 ./start.sh         # print commands without preparing or starting
./stop.sh                   # stop both ranks, preserving their logs
docker logs -f glm53-flash-tf
```

To resume an interrupted setup, run `./start.sh` again from this repository.
It reuses the native images and cached model files, downloads the remaining
weights, verifies and synchronizes the checkpoint, then starts and smoke-tests
both ranks. To prepare the files without starting inference, run
`./scripts/prepare.sh` instead, followed by `./start.sh` when ready.

The endpoint remains `http://localhost:18300/v1`, model `GLM-5.3-Flash`.
Defaults: TP2, context 245,760, one concurrent request, built-in MTP drafts,
12.2 GiB memory reserve per node, four NCCL channels and the detected two-rail
RoCE link. Native GLM uses `TF_GLM_WORLD=2`, `TF_GLM_RANK=0/1`, and
`NCCL_COMM_ID=<head fabric IP>:29551`; this release does not expose `--tp`.

The Python recipe's `KV`, `KV_POOL_GIB`, `DENSE`, `DRAFTER`, and `TF_GLM_*`
tuning are not forwarded. The native engine controls its own weight math and
cache representation. Vision starts disabled (`NATIVE_VISION=0`); enable it
only after checking the remaining memory budget and codec dependencies.
`NATIVE_ROCE=0` uses NCCL for all communication.

The checkpoint download is approximately 177 GB, with a complete copy on each
node. Downloads and synchronization are resumable. Existing EXL3 weights and
images are retained. Override `NATIVE_IMAGE`, `NATIVE_REASONING_EFFORT`, or
`NATIVE_STATE_DIR` as needed. Updating `NATIVE_VERSION` also requires the
matching `NATIVE_SHA256`.

References: [NVIDIA model card](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local),
[TensorFold 1.1.1](https://github.com/ashhart/TensorFold/releases/tag/v1.1.1),
[upstream two-GB10 full-checkpoint run](https://github.com/ashhart/TensorFold/issues/1040).
