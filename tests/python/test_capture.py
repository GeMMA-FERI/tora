import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from capture_program import CaptureError, capture_program


class AddReLU(torch.nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y)


class CaptureTests(unittest.TestCase):
    def test_capture_matches_eager_computation(self):
        model = AddReLU().eval()
        for args in [(torch.tensor([-2., 3.]), torch.tensor([1., -1.])),
                     (torch.randn(2, 3), torch.randn(2, 3))]:
            program = capture_program(model, args)
            self.assertEqual([op["operator"] for op in program["operations"]],
                             ["aten.add.Tensor", "aten.relu.default"])
            self.assertEqual(program["capture"]["inputs"][0]["shape"], list(args[0].shape))
            self.assertEqual(program["capture"]["inputs"][0]["stride"], list(args[0].stride()))
            self.assertEqual(program["capture"]["version"], torch.__version__)
            values = dict(zip((i["id"] for i in program["inputs"]), args))
            def resolve(arg):
                return values[arg["value"]] if "value" in arg else arg["literal"]
            for op in program["operations"]:
                _, name, overload = op["operator"].split(".")
                result = getattr(getattr(torch.ops.aten, name), overload)(
                    *(resolve(arg) for arg in op["args"]),
                    **{key: resolve(arg) for key, arg in op["kwargs"].items()})
                results = result if isinstance(result, tuple) else (result,)
                values.update(zip(op["outputs"], results))
            torch.testing.assert_close(values[program["outputs"][0]["value"]], model(*args))

    def test_tuple_selection_maps_to_return_positions(self):
        class Max(torch.nn.Module):
            def forward(self, x):
                values, indices = torch.max(x, dim=0)
                return indices, values
        program = capture_program(Max(), (torch.tensor([2., 5.]),))
        op = program["operations"][0]
        self.assertEqual(op["operator"], "aten.max.dim")
        self.assertEqual([o["value"] for o in program["outputs"]], list(reversed(op["outputs"])))

    def test_rejects_parameters_mutation_and_primitive_inputs(self):
        with self.assertRaisesRegex(CaptureError, "PARAMETER"):
            capture_program(torch.nn.Linear(2, 2), (torch.ones(1, 2),))
        class Mutate(torch.nn.Module):
            def forward(self, x):
                return x.add_(1)
        with self.assertRaisesRegex(CaptureError, "mutat|mutable"):
            capture_program(Mutate(), (torch.ones(2),))
        with self.assertRaisesRegex(CaptureError, "flat tensor inputs"):
            capture_program(AddReLU(), (torch.ones(2), 1))

    def test_rejects_tensor_constants_and_captures_integer_sequences(self):
        class Constant(torch.nn.Module):
            def forward(self, x):
                return x + torch.tensor([1., 2.])
        with self.assertRaisesRegex(CaptureError, "CONSTANT_TENSOR"):
            capture_program(Constant(), (torch.ones(2),))
        class Reshape(torch.nn.Module):
            def forward(self, x):
                return x.reshape([1, 2])
        program = capture_program(Reshape(), (torch.ones(2),))
        self.assertEqual(program["operations"][0]["args"][1], {"items": [{"literal": 1}, {"literal": 2}]})

    def test_python_to_nodegraph_cli(self):
        with tempfile.TemporaryDirectory(prefix="tora-capture-") as directory:
            work = Path(directory)
            program = work / "program.json"
            registry = work / "registry.json"
            graph = work / "graph.json"
            subprocess.run([sys.executable, "scripts/capture_program.py", "--factory",
                            "examples/add_relu.py:example", "--out", str(program)], cwd=ROOT, check=True, capture_output=True)
            script = """
import fs from 'node:fs';
import {buildNodeRegistry} from './dist/registry.js';
import {parseFuncSchema} from './dist/schemaParser.js';
const schemas = ['add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor', 'relu(Tensor self) -> Tensor'];
fs.writeFileSync(process.argv[1], JSON.stringify(buildNodeRegistry(schemas.map(s=>parseFuncSchema(s,[],[])))));
"""
            subprocess.run(["node", "--input-type=module", "-e", script, str(registry)], cwd=ROOT, check=True, capture_output=True)
            subprocess.run(["node", "dist/graphCli.js", "--program", str(program),
                            "--registry", str(registry), "--out", str(graph)], cwd=ROOT, check=True, capture_output=True)
            document = json.loads(graph.read_text())
            self.assertEqual(len(document["Nodes"]), 4)
            self.assertEqual(len(document["Edges"]), 4)
            self.assertEqual(document["Nodes"]["1"]["Properties"], {"Alpha (kw_only)": 1})
            self.assertEqual(document["Tora"]["capture"]["version"], torch.__version__)
            self.assertEqual(document["Edges"]["3"]["VT"], "output_0")
            # A failed factory must not replace a previously successful artifact.
            before = program.read_bytes()
            failure = subprocess.run([sys.executable, "scripts/capture_program.py", "--factory",
                                      "examples/add_relu.py:missing", "--out", str(program)], cwd=ROOT, capture_output=True)
            self.assertNotEqual(failure.returncode, 0)
            self.assertEqual(program.read_bytes(), before)

    def test_convolutions_capture_and_export_scalar_ports(self):
        class Conv1d(torch.nn.Module):
            def forward(self, x, w):
                return torch.nn.functional.conv1d(x, w, stride=2, padding=1)
        class Conv2d(torch.nn.Module):
            def forward(self, x, w):
                return torch.nn.functional.conv2d(x, w, stride=(2, 1), padding=(1, 0), dilation=(1, 2))
        examples = [(Conv1d(), (torch.randn(1, 1, 8), torch.randn(1, 1, 3))),
                    (Conv2d(), (torch.randn(1, 1, 8, 9), torch.randn(1, 1, 3, 2)))]
        for n, (model, args) in enumerate(examples, 1):
            with self.subTest(dimensions=n), tempfile.TemporaryDirectory(prefix="tora-conv-") as directory:
                program = capture_program(model, args)
                op = program["operations"][0]
                self.assertEqual(op["operator"], f"aten.conv{n}d.default")
                values = dict(zip((p['id'] for p in program['inputs']), args))
                def resolve(argument):
                    if 'items' in argument:
                        return [resolve(item) for item in argument['items']]
                    return values[argument['value']] if 'value' in argument else argument['literal']
                result = getattr(torch.ops.aten, f'conv{n}d').default(
                    *(resolve(arg) for arg in op['args']), **{key: resolve(arg) for key, arg in op['kwargs'].items()})
                torch.testing.assert_close(result, model(*args))
                work = Path(directory)
                program_path, registry_path, graph_path = [work / name for name in ('program.json', 'registry.json', 'graph.json')]
                program_path.write_text(json.dumps(program))
                script = """
import fs from 'node:fs';
import {buildNodeRegistry} from './dist/registry.js';
import {parseFuncSchema} from './dist/schemaParser.js';
import {filterSubset} from './dist/subset.js';
import {defaultConfig} from './dist/config.js';
const schemas = [1,2].map(n=>`conv${n}d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[${n}] stride=1, SymInt[${n}] padding=0, SymInt[${n}] dilation=1, SymInt groups=1) -> Tensor`);
const selected = await filterSubset(schemas.map(s=>parseFuncSchema(s,[],[])),defaultConfig.subset,undefined,'config/operations.yaml');
fs.writeFileSync(process.argv[1],JSON.stringify(buildNodeRegistry(selected)));
"""
                subprocess.run(['node', '--input-type=module', '-e', script, str(registry_path)], cwd=ROOT, check=True, capture_output=True)
                subprocess.run(['node', 'dist/graphCli.js', '--program', str(program_path), '--registry', str(registry_path), '--out', str(graph_path)], cwd=ROOT, check=True, capture_output=True)
                graph = json.loads(graph_path.read_text())
                properties = graph['Nodes']['1']['Properties']
                expected = {'Stride': 2, 'Padding': 1, 'Dilation': 1, 'Groups': 1} if n == 1 else {
                    'Stride Height': 2, 'Stride Width': 1, 'Padding Height': 1, 'Padding Width': 0,
                    'Dilation Height': 1, 'Dilation Width': 2, 'Groups': 1}
                self.assertEqual(properties, expected)


if __name__ == "__main__":
    unittest.main()
