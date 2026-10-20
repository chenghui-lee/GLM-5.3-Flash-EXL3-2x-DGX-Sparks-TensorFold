"""Incomplete downloads must never be accepted as a deployable checkpoint."""
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "verify_native_checkpoint", Path(__file__).parents[1] / "tools/verify_native_checkpoint.py"
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class NativeCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = {"model_type": "glm5_next", "quantization_config": {
            "quant_method": "modelopt", "quantized_layers": {"expert": {"quant_algo": "NVFP4"}}
        }}
        (self.root / "config.json").write_text(json.dumps(config))
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            (self.root / name).write_text("{}")
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"expert.weight": "model.safetensors"}
        }))
        header = json.dumps({"expert.weight": {
            "dtype": "U8", "shape": [4], "data_offsets": [0, 4]
        }}).encode()
        self.shard = self.root / "model.safetensors"
        self.shard.write_bytes(struct.pack("<Q", len(header)) + header + b"1234")

    def test_complete_checkpoint(self):
        checker.verify(self.root)

    def test_truncated_shard(self):
        self.shard.write_bytes(self.shard.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, "truncated"):
            checker.verify(self.root)

    def test_missing_shard(self):
        self.shard.unlink()
        with self.assertRaises(FileNotFoundError):
            checker.verify(self.root)

    def test_missing_indexed_tensor(self):
        (self.root / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"missing.weight": "model.safetensors"}
        }))
        with self.assertRaisesRegex(ValueError, "missing tensor"):
            checker.verify(self.root)

    def test_wrong_quantization(self):
        config = json.loads((self.root / "config.json").read_text())
        config["quantization_config"]["quant_method"] = "exl3"
        (self.root / "config.json").write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "ModelOpt"):
            checker.verify(self.root)


if __name__ == "__main__":
    unittest.main()
