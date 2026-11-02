[← Tora](../README.md)

# Running a captured model with NodeEngine

NodeEngine can execute the captured Conv2d model from the command line. No
additional model lowering is needed for this example. The CLI must discover the
matching PyTorch extension and its runtime DLLs.

Run commands from the Tora repository root. Generate and build the
[extension](generation.md) first, and create the [capture environment](capture.md).

## Configure the CLI

Use the Gemma Fusion extension host (`gfexthost.exe`), or a working `gf.exe`
installation exposing `node execute`. Check its options:

```powershell
$cli = "C:/path/to/gfexthost.exe"
& $cli node execute --help
& $cli configure show
```

The extension host's default search root is `Extensions/` beside its executable.
Place the built extension in
`Extensions/GF.Extensions.PyTorch/GF.Extensions.PyTorch.dll`, with libtorch and
other required runtime DLLs available beside the extension or on `PATH`.

Alternatively, configure an extension root containing the `GF.Extensions.*`
folders:

```powershell
& $cli configure set --path "C:/path/to/extension-root" --verbose --log-level 0
```

`configure set` replaces this host's saved configuration, so inspect it first.
For an isolated test, use a separate copy of the host/runtime with a distinct
executable name and only the PyTorch extension under its `Extensions/` directory.
Do not mistake imported ghost-node snapshots for loaded processors: the
execution log must show the real loader and convolution stages running.

## Run the comparison

```powershell
scripts/test-nodeengine.ps1 -Cli $cli
```

The script builds TypeScript, captures the example with file-backed inputs,
generates a graph, runs NodeEngine with JIT disabled, and checks its output.
Use `-Jit 1` to check the JIT-enabled path as well.

Artifacts are written under `_build/`:

- `conv2d-module-cli.nodegraph.json`: captured model plus output-formatting nodes.
- `conv2d-module-cli-output.json`: `Text` and `Expected` strings.
- `conv2d-module-cli-report.json`: execution performance report.

The reference is computed independently by eager PyTorch and stored in
`conv2d-module-input.safetensors`. The graph loads it through a separate loader.
Two Tensor To Text nodes format the computed and reference tensors; the script
requires both strings to be nonempty and identical. This compares the displayed
values, shape and dtype. The [native captured-module test](development.md#native-checks)
provides the numerical comparison with explicit tolerances.

## Execute an existing graph directly

```powershell
& $cli node execute `
  --graph "C:/path/to/conv2d-module-cli.nodegraph.json" `
  --inputs-inline "{}" `
  --outputs "C:/path/to/output.json" `
  --report "C:/path/to/report.json" `
  --only-factory-serialize --output-names-type short --jit 0
```

`{}` is sufficient because this example's input tensor is already a resource.
Models with Generic Inputs need values supplied through the CLI's input mechanism;
direct tensor JSON input is not established by this example.

`--only-factory-serialize` writes output values even when the output file already
exists. Without it, the host treats an existing output file as a destination
template. `--output-names-type short` keeps the keys `Text` and `Expected`.

## What was verified

On 2026-11-02, an isolated copy of the local extension host and its runtime ran
the original tensor-output graph successfully with JIT disabled. The log showed
four Load Tensor nodes, Conv2d, Add and ReLU executing with status `Success`.
The default output serializer wrote `{}` for that tensor-valued result.

The text-output comparison graph makes the result inspectable without requiring
a tensor JSON serializer. It includes a fifth loader for the reference tensor.
The comparison passed with both JIT disabled (`0`) and enabled (`1`).
This output adapter belongs to the example; the core exporter continues to
preserve tensor outputs.

The source build's `gf.exe` launcher failed before graph import with an empty
exception message on this machine. Calling the working extension host directly
avoided that launcher issue. A zero exit code and `{}` alone are insufficient
evidence of successful model execution; check the stage log and comparison.

Keep the resource files where their loader Base Directory properties point.
Regenerate the graph after moving the program and archives.
