# GF Extensions

Project which will contain extensions for the Tora adapter.

## Getting started

To build extensions run `./bs.ps1 -Install` in pwsh.

## Load Tensor

The PyTorch extension includes `Load Tensor` in `PyTorch::IO`, with implementation
ID `3a6a2744-f014-45d9-bb93-3d96c9f6b152`.

| Port | Meaning |
| --- | --- |
| File | Path to a `.safetensors` archive |
| Tensor Name | Exact key, such as `linear.weight` |
| Base Directory | Optional absolute directory for resolving a relative File |
| Tensor (output) | Owned, contiguous CPU tensor with the stored shape and dtype |

Absolute file paths work directly. Relative paths require an absolute Base
Directory because the processor API does not supply the graph's directory.
No path is implicitly resolved against Visual Studio or the editor's working
directory. The graph saves the file reference and tensor name, not the weights.

The reader implements the
[Safetensors format](https://github.com/safetensors/safetensors/blob/main/README.md)
using the existing RapidJSON and libtorch dependencies. Supported archive dtypes
are BOOL, U8/U16/U32/U64, I8/I16/I32/I64, F16, BF16, F32 and F64. Scalars and empty
tensors are supported. Archives containing other dtypes are rejected. Header
size is limited to 100 MB; shapes, offsets, overlaps, duplicate keys and file
completeness are checked before reading tensor data.

Each evaluation reads the header and only the selected tensor payload into new
storage. There is no shared cache or device-transfer behavior in this first
version. Changes to the file take effect on the next evaluation. Missing files,
unknown tensor keys and invalid archives produce errors including the path and
requested tensor name.

From the repository root, create a small test archive:

```powershell
_build/capture-venv/Scripts/python.exe -m pip install -r scripts/requirements-loader-test.txt
_build/capture-venv/Scripts/python.exe examples/save_weights.py
```

Set File to the absolute path of `_build/example-weights.safetensors` and Tensor
Name to `linear.weight`. Connect the output to Tensor To Text to inspect the
Float32 tensor `[[1, 2], [-3, 4]]`. The same archive contains `linear.bias` with
values `[0.5, -0.5]`. Rebuild/reload the extension before using the new node.

`scripts/test-tensor-loader.cmd` compiles the C++ loader test executable and runs
Python interoperability tests against the official Safetensors writer. It uses
VS2026, `VCPKG_ROOT`, the project-local capture environment, and `LibTorchDir`
(default `C:\GitHub\libtorch\libtorch`). Tests include exact tensor bytes, Unicode
paths, relative-path resolution, missing tensors, unsupported dtypes and malformed
archives. These exercise the native loader; editor UI execution is a separate
manual check. Python capture does not yet emit weight resources or loader nodes.
