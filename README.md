# Tora

Tora turns supported PyTorch models into graphs you can run in Gemma Fusion.
It also generates the C++/libtorch processors that those graphs use.

The workflow has two parts:

- **Build the extension:** select PyTorch operations and generate their C++ nodes
  and a registry of bindings.
- **Capture a model:** export a PyTorch module, save its weights in safetensors,
  and turn the captured operations into a NodeGraph using that registry.

Weights stay in external files. Graphs contain references to those files and can
run in the Node Editor or through the NodeEngine CLI.

## Getting started

The current native workflow targets Windows/MSVC and requires Node.js/npm,
Python, libtorch, and the Gemma Fusion SDK. Python capture is tested with
Python 3.14 and PyTorch 2.12.1; use the matching libtorch release.

Run these commands from the repository root:

```powershell
npm install
python -m venv _build/capture-venv
_build/capture-venv/Scripts/python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
_build/capture-venv/Scripts/python.exe -m pip install -r scripts/requirements-capture.txt
```

### 1. Generate and build the extension

Use a local `native_functions.yaml` matching your libtorch release:

```powershell
npm run gen -- --source _build/native_functions.yaml
```

This writes C++ headers and `registry.json` under
`extensions/GF.Extensions.PyTorch/generated/`. Build and install the extension
using the [extension setup](extensions/README.md#getting-started), then load it in
Gemma Fusion. The extension and registry must come from the same generation.
See [generation and configuration](docs/generation.md) for schema selection,
operation filters, and native setup details.

### 2. Capture the example model

The supplied example is a Conv2d module followed by an offset and ReLU. This
command captures its weights and buffers and saves a sample input alongside an
eager PyTorch reference:

```powershell
_build/capture-venv/Scripts/python.exe examples/capture_conv2d_module_input.py
npm run graph -- --program _build/conv2d-module-with-input.program.json --registry extensions/GF.Extensions.PyTorch/generated/registry.json --out _build/conv2d-module-with-input.nodegraph.json
```

### 3. Run it

Open `_build/conv2d-module-with-input.nodegraph.json` in the Node Editor and
execute it. Its four Load Tensor nodes supply the input, weights, bias, and
offset. The output is a Float32 tensor with shape `[1, 2, 4, 8]`.
This example has been verified in the Node Editor.

To run the same model from a terminal, follow the
[NodeEngine CLI guide](docs/nodeengine.md). It adds text outputs so the result
can be inspected and compared with the PyTorch reference.

## Capture your own model

Provide a Python factory returning `(module, args)` or `(module, args, kwargs)`.
Set the module's training/evaluation mode in the factory and use a tuple for
`args`.

```powershell
_build/capture-venv/Scripts/python.exe scripts/capture_program.py --factory examples/conv2d_module.py:example --out _build/my-model.program.json
npm run graph -- --program _build/my-model.program.json --registry extensions/GF.Extensions.PyTorch/generated/registry.json --out _build/my-model.nodegraph.json
```

The capture command saves weights to `_build/my-model.program.safetensors`.
Runtime inputs remain graph inputs; supply tensors matching the capture's
shapes, dtypes and devices. See [capture and resource handling](docs/capture.md)
for the full contract.

## What is supported?

Tora currently handles a static, straight-line subset of PyTorch with flat tensor
inputs, tensor outputs, supported literals, and configured fixed-size integer
arguments. Module parameters, buffers, and tensor constants are saved as CPU
snapshots. Conv1d and Conv2d expose stride, padding, and dilation as scalar ports.

The catalog has 651 enabled operations with reviewed names and descriptions.
That is a set of candidates, not a guarantee that every operation compiles or
runs with your installed libtorch. Unsupported bindings fail with diagnostics.

Dynamic shapes, mutation, control flow, general list ports, automatic conversions,
and device-placement lowering remain unsupported. Moving only a graph does not
move its weights: keep program and tensor files together and regenerate the graph
to update its absolute resource directory.

## Documentation

- [Capture models and external tensors](docs/capture.md)
- [Generate and configure the extension](docs/generation.md)
- [Run graphs with the NodeEngine CLI](docs/nodeengine.md)
- [Inspect tensors and explore convolution examples](docs/tensors.md)
- [Program format, binding, and graph export](docs/program-format.md)
- [Review operation categories and display text](docs/operation-review.md)
- [Develop and validate Tora](docs/development.md)
