"""Verify GLM ModelOpt metadata and every indexed tensor's on-disk extent."""
import json
import struct
import sys
from pathlib import Path


def verify(root):
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "glm5_next":
        raise ValueError("expected glm5_next model_type")
    quant = config.get("quantization_config", {})
    if quant.get("quant_method") != "modelopt":
        raise ValueError("expected ModelOpt quantization")
    if not any(v.get("quant_algo") == "NVFP4" for v in quant.get("quantized_layers", {}).values()):
        raise ValueError("no NVFP4 quantized layers")
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not (root / name).is_file():
            raise ValueError(f"missing {name}")
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}
    for shard in sorted(set(weight_map.values())):
        path = root / shard
        size = path.stat().st_size
        with path.open("rb") as stream:
            header_len = struct.unpack("<Q", stream.read(8))[0]
            if header_len > min(size - 8, 64 * 1024 * 1024):
                raise ValueError(f"invalid safetensors header: {shard}")
            header = json.loads(stream.read(header_len))
        end = max(t["data_offsets"][1] for k, t in header.items() if k != "__metadata__")
        if size != 8 + header_len + end:
            raise ValueError(f"truncated or invalid shard: {shard}")
        headers[shard] = header
    for name, shard in weight_map.items():
        if name not in headers[shard]:
            raise ValueError(f"missing tensor {name} in {shard}")
    print(f"GLM ModelOpt NVFP4: {len(weight_map)} tensors, {len(headers)} complete shards")


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
