"""Interoperability tests for the compiled C++ loader (run via test-tensor-loader.cmd)."""
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "_build/loader-test/load_tensor.exe"


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tora-weights-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.file = self.root / "weights.safetensors"

    def run_loader(self, name, file=None, base=None):
        command = [str(EXE), str(file or self.file), name, str(self.root / "result.bin")]
        if base is not None:
            command.append(str(base))
        return subprocess.run(command, capture_output=True, text=True, encoding="utf8")

    def write_raw(self, header, data):
        text = json.dumps(header).encode() if not isinstance(header, bytes) else header
        self.file.write_bytes(struct.pack('<Q', len(text)) + text + data)

    def test_official_writer_shapes_dtypes_and_names(self):
        tensors = {
            "linear.weight": torch.tensor([[1., 2.], [-3., 4.]]),
            "linear.bias": torch.tensor([0.5, -0.5], dtype=torch.float64),
            "half": torch.tensor([1.5, -2.], dtype=torch.float16),
            "bfloat": torch.tensor([1.5, -2.], dtype=torch.bfloat16),
            "integer": torch.tensor([2**60, -2**60], dtype=torch.int64),
            "boolean": torch.tensor([True, False]),
            "scalar": torch.tensor(3.25),
            "empty": torch.empty(0, 2),
        }
        save_file(tensors, self.file)
        for name, expected in tensors.items():
            with self.subTest(name=name):
                result = self.run_loader(name)
                self.assertEqual(result.returncode, 0, result.stderr)
                lines = result.stdout.splitlines()
                self.assertEqual([int(x) for x in lines[1].split()], list(expected.shape))
                # A raw byte comparison catches dtype changes and precision loss.
                expected_bytes = bytes(expected.reshape(-1).view(torch.uint8).tolist())
                self.assertEqual((self.root / "result.bin").read_bytes(), expected_bytes)
        self.assertEqual(self.run_loader('linear.weight', 'weights.safetensors', self.root).returncode, 0)
        self.assertNotEqual(self.run_loader('linear.weight', 'weights.safetensors').returncode, 0)
        unicode_file = self.root / 'uteži-模型.safetensors'
        save_file(tensors, unicode_file)
        self.assertEqual(self.run_loader('linear.bias', unicode_file).returncode, 0)

    def test_missing_tensor_file_and_unsupported_dtype(self):
        save_file({"x": torch.ones(1)}, self.file)
        self.assertIn('not found', self.run_loader('missing').stderr)
        self.assertIn('Cannot open', self.run_loader('x', self.root / 'absent.safetensors').stderr)
        self.write_raw({"x": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]}}, b'\0')
        self.assertIn('Unsupported', self.run_loader('x').stderr)

    def test_malformed_archives_are_rejected(self):
        valid = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}
        cases = [
            ({"x": {**valid, "shape": [-1]}}, b'\0' * 4),
            ({"x": {**valid, "shape": [2**62, 8]}}, b'\0' * 4),
            ({"x": valid}, b'\0' * 3),
            ({"x": valid}, b'\0' * 5),
            ({"x": valid, "y": valid}, b'\0' * 4),
            ({"x": {**valid, "data_offsets": [1, 5]}}, b'\0' * 5),
            (b'{"x":{},"x":{}}', b''),
            (b'{"x":{"dtype":"F32","dtype":"F32","shape":[],"data_offsets":[0,4]}}', b'\0' * 4),
            (b'{}\0invalid', b''),
            ({"__metadata__": {"x": 42}, "x": valid}, b'\0' * 4),
            ({"x": {"dtype": "BOOL", "shape": [1], "data_offsets": [0, 1]}}, b'\x02'),
        ]
        for header, data in cases:
            with self.subTest(header=header):
                self.write_raw(header, data)
                self.assertNotEqual(self.run_loader('x').returncode, 0)
        self.file.write_bytes(struct.pack('<Q', 2**64 - 1))
        self.assertIn('header length', self.run_loader('x').stderr)


if __name__ == '__main__':
    unittest.main()
