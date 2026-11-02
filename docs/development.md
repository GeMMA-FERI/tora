[← Tora](../README.md)

# Development and validation

## Project structure

- `src/cli.ts` - CLI entrypoint
- `src/generator.ts` - end-to-end generation pipeline
- `src/schemaParser.ts` - `func:` schema parser
- `src/subset.ts` - subset filtering logic
- `src/templates.ts` - per-argument/return template mapping
- `src/renderer.ts` - C++ node rendering
- `src/naming.ts` - naming + deterministic UUID generation
- `scripts/capture_program.py` - PyTorch capture and tensor snapshots
- `src/program.ts`, `src/binder.ts` - program representation and operation binding
- `src/graphDocuments.ts`, `src/graphCli.ts`, `src/graphExporter.ts` - document validation and graph export
- `src/tensorResources.ts` - shared tensor-loader identity and ports

Run the capture and full Python-to-JSON tests after building TypeScript:

```powershell
npm run build
_build/capture-venv/Scripts/python.exe -m unittest discover -s tests/python -v
```

These tests replay the captured operations in PyTorch and compare them with the
original module. They also exercise both CLIs, but do not execute GemmaFusionSuite.

To download libtorch headers, run:

```powershell
Invoke-WebRequest https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-2.12.1%2Bcu126.zip -OutFile libtorch.zip
Extract-Archive libtorch.zip -DestinationPath libtorch
```

The project defaults `LibTorchDir` to `C:/GitHub/libtorch/libtorch`; override
that MSBuild property for another installation.

## Native checks

The Windows test scripts use VS2026, `VCPKG_ROOT`, the capture environment,
`_build/native_functions.yaml`, and the installed Gemma Fusion SDK and libtorch.
Set `GemmaFusionSdkDir` and `LibTorchDir` when their defaults do not match your machine.

```powershell
npm test
scripts/test-tensor-loader.cmd
scripts/test-convolution.cmd
scripts/test-convolution-features.cmd
scripts/test-captured-module.cmd
```

These checks exercise different boundaries:

| Check | What it establishes |
| --- | --- |
| `npm test` | Parsing, generation, binding, compatibility, graph serialization and CLI validation |
| Python unit tests | Capture, tensor snapshots, operation replay against eager PyTorch, and Python-to-JSON integration |
| Native scripts | Real loader and generated processors through libtorch/SDK; not NodeEngine scheduling |
| Node Editor | Interactive graph import and execution; the captured Conv2d graph with file-backed inputs was confirmed by Mitko on 2026-11-02 |
| [NodeEngine CLI](nodeengine.md) | Graph import, extension discovery, scheduling, execution and formatted output comparison |

Generated files belong under the extension's `generated/` directory; change the
TypeScript generator rather than editing its output. Keep temporary artifacts in
`_build/`. Keep a registry together with the extension generated from the same
schema and operation configuration.
