import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "examples"))
from capture_program import CaptureError, capture_program
from conv2d_module import example


def replay(program, args, directory):
    values = dict(zip((item["id"] for item in program["inputs"]), args))
    archives = {}
    for resource in program.get("resources", []):
        file = resource["file"]
        if file not in archives:
            archives[file] = load_file(Path(directory) / file)
        values[resource["id"]] = archives[file][resource["key"]]

    def resolve(argument):
        if "items" in argument:
            return [resolve(item) for item in argument["items"]]
        return values[argument["value"]] if "value" in argument else argument["literal"]

    for operation in program["operations"]:
        _, name, overload = operation["operator"].split(".")
        result = getattr(getattr(torch.ops.aten, name), overload)(
            *(resolve(arg) for arg in operation["args"]),
            **{key: resolve(arg) for key, arg in operation["kwargs"].items()})
        values.update(zip(operation["outputs"], result if isinstance(result, tuple) else (result,)))
    return tuple(values[output["value"]] for output in program["outputs"])


class ResourceCaptureTests(unittest.TestCase):
    def test_conv_parameters_and_buffer_replay_from_one_archive(self):
        model, args = example()
        with tempfile.TemporaryDirectory(prefix="tora-state-") as directory:
            file = Path(directory) / "model.safetensors"
            program = capture_program(model, args, resource_file=file)
            self.assertEqual(program["version"], 2)
            self.assertEqual(len(program["inputs"]), 1)
            self.assertEqual(len(program["capture"]["inputs"]), 1)
            self.assertEqual({r["file"] for r in program["resources"]}, {file.name})
            self.assertEqual({r["key"] for r in program["resources"]}, {"conv.weight", "conv.bias", "offset"})
            for key, tensor in load_file(file).items():
                torch.testing.assert_close(tensor, model.state_dict()[key])
            torch.testing.assert_close(replay(program, args, directory)[0], model(*args))

    def test_constants_nonpersistent_buffers_views_and_tied_parameters(self):
        class State(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.arange(6.).reshape(2, 3).t())
                self.other = self.weight
                self.register_buffer("offset", torch.tensor(0.25), persistent=False)
                self.register_buffer("empty", torch.empty(0, 2))
                self.constant = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])

            def forward(self, x):
                return x * self.weight + self.other + self.offset + self.constant, self.empty

        model = State().eval()
        args = (torch.ones(3, 2),)
        with tempfile.TemporaryDirectory(prefix="tora-state-") as directory:
            file = Path(directory) / "state.safetensors"
            program = capture_program(model, args, resource_file=file)
            tensors = load_file(file)
            self.assertTrue(any(tensor.numel() == 0 for tensor in tensors.values()))
            self.assertTrue(any(tensor.ndim == 0 for tensor in tensors.values()))
            self.assertTrue(all(tensor.is_contiguous() for tensor in tensors.values()))
            for actual, expected in zip(replay(program, args, directory), model(*args)):
                torch.testing.assert_close(actual, expected)
            # Captured resources are snapshots, independent of subsequent updates.
            before = file.read_bytes()
            with torch.no_grad():
                model.weight.fill_(99)
            self.assertEqual(file.read_bytes(), before)

    def test_inline_tensor_constant_is_saved_and_replayed(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                return x + torch.tensor([1., 2.])
        with tempfile.TemporaryDirectory(prefix="tora-constant-") as directory:
            model, args = Constant(), (torch.ones(2),)
            program = capture_program(model, args, resource_file=Path(directory) / "constant.safetensors")
            self.assertEqual(len(program["resources"]), 1)
            torch.testing.assert_close(replay(program, args, directory)[0], model(*args))

    def test_invalid_resources_and_mutation_preserve_existing_archive(self):
        class Complex(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.ones(2, dtype=torch.complex64))

            def forward(self, x):
                return x + self.weight

        class Mutate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.ones(2))

            def forward(self, x):
                self.state.add_(x)
                return self.state

        with tempfile.TemporaryDirectory(prefix="tora-invalid-") as directory:
            file = Path(directory) / "preserved.safetensors"
            file.write_bytes(b"preserved")
            for model, message in [(Complex(), "Unsupported tensor resource"), (Mutate(), "mutat|mutable")]:
                with self.subTest(model=type(model).__name__), self.assertRaisesRegex(CaptureError, message):
                    capture_program(model, (torch.ones(2),), resource_file=file)
                self.assertEqual(file.read_bytes(), b"preserved")

    def test_cli_creates_archive_and_graph_without_manual_nodes(self):
        with tempfile.TemporaryDirectory(prefix="tora-state-cli-") as directory:
            work = Path(directory)
            program_path = work / "model.program.json"
            registry_path = work / "registry.json"
            graph_path = work / "elsewhere/model.nodegraph.json"
            subprocess.run([sys.executable, "scripts/capture_program.py", "--factory", "examples/conv2d_module.py:example",
                            "--out", str(program_path)], cwd=ROOT, check=True, capture_output=True)
            script = """
import fs from 'node:fs';
import {buildNodeRegistry} from './dist/registry.js';
import {parseFuncSchema} from './dist/schemaParser.js';
import {filterSubset} from './dist/subset.js';
import {defaultConfig} from './dist/config.js';
const schemas = ['conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=1, SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1) -> Tensor',
                 'add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor', 'relu(Tensor self) -> Tensor'];
const selected = await filterSubset(schemas.map(s=>parseFuncSchema(s,[],[])), defaultConfig.subset, undefined, 'config/operations.yaml');
fs.writeFileSync(process.argv[1], JSON.stringify(buildNodeRegistry(selected)));
"""
            subprocess.run(["node", "--input-type=module", "-e", script, str(registry_path)], cwd=ROOT, check=True, capture_output=True)
            subprocess.run(["node", "dist/graphCli.js", "--program", str(program_path), "--registry", str(registry_path),
                            "--out", str(graph_path)], cwd=ROOT, check=True, capture_output=True)
            graph = json.loads(graph_path.read_text())
            program = json.loads(program_path.read_text())
            self.assertEqual(len(graph["Nodes"]), 8)
            loaders = [node for node in graph["Nodes"].values() if graph["NodeBases"][node["NodeBase"]]["Name"] == "Load Tensor"]
            self.assertEqual(len(loaders), 3)
            self.assertEqual({n["Properties"]["File"] for n in loaders}, {"model.program.safetensors"})
            self.assertEqual({Path(n["Properties"]["Base Directory"]) for n in loaders}, {work})
            model, args = example()
            torch.testing.assert_close(replay(program, args, work)[0], model(*args))


if __name__ == "__main__":
    unittest.main()
