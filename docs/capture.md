[← Tora](../README.md)

# Capturing PyTorch modules

Run commands from the repository root. Generate and install the matching
[extension and registry](generation.md) before executing exported graphs.

## Capturing a Python module

`scripts/capture_program.py` uses `torch.export.export(..., strict=True)` to
capture a module with example inputs, then translates its ATen graph to the
program format. It follows PyTorch's
[export graph representation](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html).
This is a capture frontend for a supported subset, not a general Python parser.

The adapter is tested with Python 3.14 and PyTorch 2.12.1+cpu, matching the release
of the local libtorch 2.12.1+cu126 installation. For a project-local CPU environment:

```powershell
python -m venv _build/capture-venv
_build/capture-venv/Scripts/python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
_build/capture-venv/Scripts/python.exe -m pip install -r scripts/requirements-capture.txt
```

A factory supplies the module and its example tensors. It returns `(module,
args)` or `(module, args, kwargs)`, where `args` is a tuple. For example,
`examples/add_relu.py:example` returns an evaluation-mode module and two tensors.
The adapter uses the mode supplied by the factory; it does not switch modes.

```powershell
_build/capture-venv/Scripts/python.exe scripts/capture_program.py --factory examples/add_relu.py:example --out _build/add-relu.captured.program.json
npm run graph -- --program _build/add-relu.captured.program.json --registry _build/graph-example-extension/registry.json --out _build/add-relu.from-python.nodegraph.json
```

Generate the registry as described in [graph export](program-format.md#exporting-a-nodegraph) first. `--factory` executes the selected
Python factory and captures the returned module. Input IDs come from graph
placeholders; outputs are flattened in export order and named `output_0`,
`output_1`, etc. Original nested Python return structures are not reconstructed.
Tuple selection maps to the producer's return position rather than an editor
selection node. The graph compiler still decides whether each exact ATen overload
has an available extension binding.

The adapter accepts flat, static, strided tensor inputs and ATen operations
producing tensors or tuples of tensors. Parameters, buffers (including
non-persistent buffers), and lifted tensor constants are captured as external
resources. It rejects input/buffer mutation, symbolic dimensions, non-integer
sequences, and control-flow operators. Primitive user inputs are rejected because export
can specialize them to the example values. Literal operator arguments remain
supported by the program format. Static Python branches may be specialized and
loops may be unrolled during capture; their original structure is not retained.

Capture records PyTorch version and input shapes, strides, dtypes, devices, and
gradient flags in `program.capture`. The exporter copies this to an advisory
`Tora.capture` field. Supply inputs consistent with the capture assumptions:
the editor does not enforce this metadata and may drop it when re-exporting.
This metadata is not a complete serialization of PyTorch export guards.
Tensor contents and model weights are not embedded in the generated graph.

### Capturing module weights and buffers

Capture a module with its own Conv2d weights, bias, and an offset buffer:

```powershell
npm run gen -- --source _build/native_functions.yaml
_build/capture-venv/Scripts/python.exe scripts/capture_program.py --factory examples/conv2d_module.py:example --out _build/conv2d-module.program.json
npm run graph -- --program _build/conv2d-module.program.json --registry extensions/GF.Extensions.PyTorch/generated/registry.json --out _build/conv2d-module.nodegraph.json
```

Capture writes `_build/conv2d-module.program.safetensors` beside the program.
One archive stores all captured model tensors, using their exported target names
(for example `conv.weight`, `conv.bias`, and `offset`). Runtime inputs remain
Generic Inputs. The graph exporter creates Load Tensor nodes automatically;
there is no need to insert loaders or rewrite edges in example scripts.

Programs with resources use version 2. Version 1 programs remain supported, and
stateless captures still emit version 1. Resource IDs share the value namespace
with inputs and operation results, so collisions are rejected. For example:

```json
{
  "id": "p_conv_weight",
  "type": "Tensor",
  "file": "conv2d-module.program.safetensors",
  "key": "conv.weight"
}
```

These entries appear in a top-level `resources` array. Operations reference them
with the existing `{ "value": "p_conv_weight" }` syntax. The graph command resolves
relative resource files against the **program's directory**, then stores that
absolute directory in each loader's Base Directory property. Moving the program
and archive together and regenerating the graph updates those paths. Moving only
the graph does not relocate its resources. The graph command does not read or
validate tensor payloads; the native loader validates them at execution time.

The library API requires an explicit absolute `resourceDirectory` when resources
use relative filenames:

```js
const graph = exportNodeGraph(program, registry, { resourceDirectory: absoluteModelDirectory });
```

Python callers use `capture_program(module, args, resource_file=Path(...))` and
save the returned program beside that archive. Resources are independent,
contiguous CPU snapshots with their original shape and dtype. Tied parameters
and views are saved as independent values; storage aliasing and gradient state
are not preserved. Resource-bearing modules currently require CPU state and CPU
inputs; device placement needs a separate lowering step. Sparse, quantized,
complex, and other loader-unsupported resource dtypes are rejected.

Inline tensor construction is functionalized by PyTorch without requesting
operator decompositions. It can introduce operations such as `aten.clone.default`;
capture support does not imply that all resulting operators have an enabled
extension binding. Binding reports unsupported operators rather than changing
their behavior. Attribute tensor constants can be referenced directly.

Run `scripts\test-captured-module.cmd` after generating the extension to capture
this example, export its graph, and execute its loader and generated Conv2d/Add/ReLU
processors through the SDK using the graph's properties and edge keys. It compares
the result against an independent eager PyTorch reference. This is a native test
harness, not execution inside the Node Editor. The graph with file-backed inputs below was verified in the Node Editor by
Mitko on 2026-11-02. The generic-input variant still requires an input tensor
with shape `[1, 1, 8, 9]`.

For a ready-to-run graph with that same input connected to Conv2d:

```powershell
_build/capture-venv/Scripts/python.exe examples/capture_conv2d_module_input.py
npm run graph -- --program _build/conv2d-module-with-input.program.json --registry extensions/GF.Extensions.PyTorch/generated/registry.json --out _build/conv2d-module-with-input.nodegraph.json
```

Open `_build/conv2d-module-with-input.nodegraph.json` and execute it. Four Load
Tensor nodes supply the original Float32 input `[1, 1, 8, 9]`, weights, bias, and
offset buffer. The output has shape `[1, 2, 4, 8]`. Model state is stored in
`conv2d-module-with-input.safetensors`; the input and an eager `expected` reference
are stored in `conv2d-module-input.safetensors`. Inspect `output_0` in TensorEditor;
slice indices `0, 0` and `0, 1` select its two channels.

The loader currently opens and validates the archive on each load and reads only
the selected tensor's payload into owned CPU storage. Shared header/tensor caching
across nodes and executions is not implemented yet.

See [NodeEngine CLI execution](nodeengine.md) for a repeatable command-line check.
