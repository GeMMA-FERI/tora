# Tora

Tora is a cute little 💖 automatic transpiler for PyTorch function definition to proprietary C++ code with bindings to the libtorch C++ API functions.

1. Reads PyTorch `native_functions.yaml`
2. Parses function schemas into typed IR
3. Filters by a file-based allowlist
4. Emits C++ node definition files with deterministic IDs

## Project structure

- `src/cli.ts` - CLI entrypoint
- `src/generator.ts` - end-to-end generation pipeline
- `src/schemaParser.ts` - `func:` schema parser
- `src/subset.ts` - subset filtering logic
- `src/templates.ts` - per-argument/return template mapping
- `src/renderer.ts` - C++ node rendering
- `src/naming.ts` - naming + deterministic UUID generation

## Install & run

```bash
npm install
npm run gen
```

Default output: `extensions/GF.Extensions.PyTorch/generated/nodes/*.hpp` + `extensions/GF.Extensions.PyTorch/generated/manifest.txt`

Generation also writes `registry.json`, a versioned catalog for the graph compiler.
It contains only the operations selected for that generation, with exact ATen
overloads, implementation UUIDs, argument/return mappings to current terminal
keys, schema types, and C++ property types. Terminal keys include the existing
optional/list/keyword-only suffixes and are shared with C++ generation.
Defaults are recorded as `defaultExpression` schema text, not evaluated values.
The registry describes generated bindings; inclusion does not certify runtime
support. Each port includes `editorType` with the editor's numeric `ValueType`,
`ClassType`, and `InterfaceType`; the registry also records node names and
descriptions for graph snapshots.
Keep it with the extension generated from the same schema and configuration.

The `editorProfile` identifies the Windows/MSVC serialization mapping in
`src/editor.ts`. The initial `gemmafusionsuite-msvc-v1` profile was checked against
the SDK's `Data::Structures::ValueType` enum and property aliases, and against
the supplied `if.json` and `iter1.json` exports. Selected metadata from those
exports is retained in `tests/fixtures/editor-types.json` for regression tests.
This profile is explicit rather than assumed portable across SDKs or compilers.
It describes serialization types, not permission to convert between them.

`genericIO()` builds definitions and instance metadata for the existing Generic
Inputs and Generic Outputs nodes. Native types use the editor's short enum-name
form; tensor ports use the extended object form. Dynamic metadata uses
`ClassName`, whereas `NodeBases` snapshots use `ClassType`. Tensor metadata is
checked against the importer's implementation; import and execution in the
running editor remain to be verified.

## Graph compiler in progress

`src/program.ts` defines a versioned, straight-line program with typed inputs,
exact ATen operator names, positional/keyword arguments, and explicit result
value IDs. Operations appear in dependency order. An argument is either
`{ "value": "id" }` or `{ "literal": 1 }`; strings are literals only when wrapped
in `literal`. `examples/add-relu.program.json` represents `relu(x + y)`.

`bindProgram(program, registry)` from `src/binder.ts` resolves this representation
to implementation IDs and terminal keys. It binds keyword arguments, supplies
supported schema defaults, checks value references and types, and records each
result's producing operation and port. Diagnostics include the operation ID,
operator, and optional `source` location.

After `npm run build`, the library can be used from JavaScript:

```js
import { readFile } from 'node:fs/promises';
import { bindProgram } from './dist/binder.js';

const program = JSON.parse(await readFile('examples/add-relu.program.json', 'utf8'));
const registry = JSON.parse(await readFile('extensions/GF.Extensions.PyTorch/generated/registry.json', 'utf8'));
const bound = bindProgram(program, registry);
```

This API currently expects documents matching the TypeScript interfaces; it is
not an untrusted JSON parser. It supports Tensor, Scalar, Int, Float, Bool, and
Str values, with finite numeric, boolean, string, and optional null literals.
References require matching types. Numeric, `True`/`False`, `None`, and JSON-quoted
string defaults are supported; other expressions are rejected without evaluation.
An optional null literal represents an absent input value for future lowering.
Lists, mutation, optional outputs, conversions, tensor resources, control flow,
and symbolic constraints are not supported in this first binding pass.

The binder result is an intermediate plan. `exportNodeGraph()` in
`src/graphExporter.ts` lowers that plan to the editor's `Nodes`, `Edges`, and
`NodeBases` format. The Python adapter below supplies this representation from a
captured module.

## Exporting a NodeGraph

Generate a registry with the current generator, then export a program:

```powershell
npm run gen -- --source _build/native_functions.yaml
npm run graph -- --program examples/add-relu.program.json --registry extensions/GF.Extensions.PyTorch/generated/registry.json --out _build/add-relu.nodegraph.json
```

Use a local schema matching the installed PyTorch extension. To generate only
the sample's registry and C++ definitions in a separate directory, use:

```powershell
npm run gen -- --source _build/native_functions.yaml --out _build/graph-example-extension --include '^(add\.Tensor|relu)$'
npm run graph -- --program examples/add-relu.program.json --registry _build/graph-example-extension/registry.json --out _build/add-relu.nodegraph.json
```

The graph command validates document structure, registry profile, and binding
before writing output. Older registries without editor metadata must be
regenerated. A binding/validation failure leaves an existing output untouched.
The program and registry paths cannot also be the output path.

The example exports Generic Inputs (`x`, `y`), Add, ReLU, and Generic Outputs
(`result`). Add stores `alpha=1` in `Properties`. Connections use current terminal
keys; optional `None` values are omitted from saved properties. Nodes receive
fresh instance UUIDs on each export and a layout based on dependency depth.
Repeated operations share their implementation's `NodeBases` snapshot while
retaining separate instances.

Import `_build/add-relu.nodegraph.json` in GemmaFusionSuite with the matching
extension installed. The file defines an interface with tensor inputs; it does
not embed test tensors or model weights. Supply `x` and `y` through the graph's
input mechanism to execute it. With `x=[-2, 3]` and `y=[1, -1]`, the expected
result is `[0, 2]`. Automated tests check serialization, defaults, and connections;
they do not yet run the graph in the editor.

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
_build/capture-venv/Scripts/python.exe -m pip install -r scripts/requirements-capture.txt --index-url https://download.pytorch.org/whl/cpu
```

A factory supplies the module and its example tensors. It returns `(module,
args)` or `(module, args, kwargs)`, where `args` is a tuple. For example,
`examples/add_relu.py:example` returns an evaluation-mode module and two tensors.
The adapter uses the mode supplied by the factory; it does not switch modes.

```powershell
_build/capture-venv/Scripts/python.exe scripts/capture_program.py --factory examples/add_relu.py:example --out _build/add-relu.captured.program.json
npm run graph -- --program _build/add-relu.captured.program.json --registry _build/graph-example-extension/registry.json --out _build/add-relu.from-python.nodegraph.json
```

Use the registry-generation command above first. `--factory` executes the selected
Python factory and captures the returned module. Input IDs come from graph
placeholders; outputs are flattened in export order and named `output_0`,
`output_1`, etc. Original nested Python return structures are not reconstructed.
Tuple selection maps to the producer's return position rather than an editor
selection node. The graph compiler still decides whether each exact ATen overload
has an available extension binding.

The initial adapter accepts flat, static, strided tensor inputs and ATen
operations producing tensors or tuples of tensors. It rejects lifted parameters,
buffers, tensor constants, input mutation, symbolic dimensions, list arguments,
and control-flow operators. Primitive user inputs are rejected because export
can specialize them to the example values. Literal operator arguments remain
supported by the program format. Static Python branches may be specialized and
loops may be unrolled during capture; their original structure is not retained.

Capture records PyTorch version and input shapes, strides, dtypes, devices, and
gradient flags in `program.capture`. The exporter copies this to an advisory
`Tora.capture` field. Supply inputs consistent with the capture assumptions:
the editor does not enforce this metadata and may drop it when re-exporting.
This metadata is not a complete serialization of PyTorch export guards.
Tensor contents and model weights are not embedded in the generated graph.

Run the capture and full Python-to-JSON tests after building TypeScript:

```powershell
npm run build
_build/capture-venv/Scripts/python.exe -m unittest discover -s tests/python -v
```

These tests replay the captured operations in PyTorch and compare them with the
original module. They also exercise both CLIs, but do not execute GemmaFusionSuite.

To download libtorch headers, run:

```bash
Invoke-WebRequest https://download.pytorch.org/libtorch/cu126/libtorch-win-shared-with-deps-2.12.1%2Bcu126.zip -OutFile libtorch.zip
Extract-Archive libtorch.zip -DestinationPath libtorch
```

Currently the path to the libtorch headers is hardcoded in the vcxproj file as `C:/GitHub/libtorch`.

## CLI options

```bash
npm run gen -- --source <url_or_file> --out <dir> --allowlist <file> --max <n> --include <regex>
```

Example command which was ran:

```bash
npm run gen -- --max 200
```

Flags:

- `--include-internal`
- `--include-inplace`
- `--include-out`
- `--include-methods`

Example (small focused subset):

```bash
npm run gen -- --max 30 --include "^(add|sub|mul|div|matmul)$"
```

## Function selection and namespaces

- function selection is controlled by `config/operations.yaml`
- operations are keyed by exact name: `op` or `op.overload`
- each overload is configured independently; a bare base name does not select all overloads
- comments allowed via `#`; set `enabled: false` or remove an operation to exclude it
- generated order follows source YAML order
- all matching entries are generated by default; `--max` optionally caps the generated count

Default allowlist path is set in `src/config.ts`.

The catalog contains all 2,590 operations in the PyTorch `main` schema snapshot
downloaded on 2026-09-18. The original 50 operations have initial categories;
the remaining entries use `PyTorch::Uncategorized` for manual classification.
The catalog is a snapshot, so new upstream operations must be added explicitly.

```yaml
version: 1
defaults:
  namespace: PyTorch::Uncategorized
operations:
  add.Tensor:
    namespace: PyTorch::BasicOperators
  conv2d:
    namespace: PyTorch::Convolution
  relu:
    namespace: PyTorch::Activation
    displayName: ReLU
    description: Replace negative values with zero and retain nonnegative values.
  sigmoid:
    enabled: false
    reason: Disabled in this extension
  tanh: {} # Included, using the default namespace
```

Namespaces use identifier segments separated by `::`, with no spaces (use
`BasicOperators`, not `Basic Operators`). Each operation is a mapping; use `{}`
to inherit defaults. The optional `defaults.namespace` falls back to
`PyTorch::Uncategorized`. Operation namespaces override that default.

Supported operation fields are `namespace`, `displayName`, `description`, `enabled` (a boolean,
defaulting to true), and `reason` (a non-empty explanatory string, not emitted
into C++). `displayName` overrides the node's visible name without changing its
implementation ID. `description` supplies the node's user-facing help text;
without it the generator falls back to the PyTorch schema description. Both
strings are escaped when emitted into C++. Other exception fields must be implemented in the generator
before use: unknown fields, invalid values, and duplicate keys are rejected.
Validation errors identify the file and configuration path; YAML syntax and
duplicate-key errors also include source locations.

The former pipe-separated text format has been replaced by YAML. The `--allowlist`
option still accepts a custom path, now to a YAML file:

```powershell
npm run gen -- --allowlist config/operations.yaml
```

Generated nodes expose `create_metadata()` and call
`AddNamespace<"PyTorch::Convolution">(metadata)` using the configured value.
This requires an SDK with the `AddNamespace` bridge helper. Namespace changes do
not change implementation IDs. The host UI must interpret `[Namespace]` values
to display the hierarchy.

Generated class names and filenames include a stable signature UUID suffix to
keep operations such as `add` and `add_` distinct, including on Windows.

Catalog inclusion does not guarantee that every schema compiles with the current
C++ type mappings or installed libtorch version. Use a matching `--source`, edit
the allowlist, or use `--include`/`--max` to select the operations you need.

Run `npm test` to check allowlist parsing and namespace generation.

## Reviewing operations with Codex

`scripts/review-operations.mjs` is a local batch workflow. It makes no network or
model API calls: Codex reads its output using the terminal tool, writes decisions
to a JSON file, and submits that file back to the program. It does not invoke or
schedule Codex itself; ask Codex to work through the batches in an active session.

```powershell
npm.cmd run review:operations -- init --source _build/native_functions.yaml
node scripts/review-operations.mjs next --size 100
node scripts/review-operations.mjs inspect --id 0
node scripts/review-operations.mjs apply --answers _build/answers.json
node scripts/review-operations.mjs status
# After all batches are reviewed:
node scripts/review-operations.mjs finish
```

Use the `native_functions.yaml` matching the catalog. `init` reads that local
file; it does not download it. It groups overloads and in-place counterparts
into families to reduce review tokens, retaining exact signatures for `inspect`.
Decisions use compact groups of family IDs from the current batch:

```json
[
  { "namespace": "PyTorch::Activation", "ids": [0, 1] },
  { "namespace": "PyTorch::Internal", "ids": [2], "reason": "Internal helper; omit from the node palette." }
]
```

Every batch ID must occur exactly once. A group `reason` disables that family;
otherwise per-overload schema checks determine eligibility. These checks exclude
internal and backward operators, in-place/output-buffer mutations, method-only
schemas, no-return operations, list-valued ports, optional returns, and argument
types without a corresponding bridge conversion. This is a conservative
forward-processing palette, not a claim that those PyTorch operations are bad.
Even ordinary convolution and pooling nodes may be disabled until list-valued
ports are implemented. Remaining enabled nodes are candidates, not proof of
C++ compilation or runtime compatibility.

Checkpoints and decisions live in `_build/operation-review` (override with
`--work`). `next` repeats the pending batch after an interruption; `apply` saves
progress. `init` resets the checkpoint, so do not rerun it to resume. Only
`finish` updates the catalog; it requires a complete review and an unchanged
source catalog, and saves `operations.before.yaml` plus `summary.json` in the
checkpoint directory. Comments, display names, existing non-default namespaces,
and prior exclusions are preserved. Every disabled operation receives a reason.

To reconsider a node later, edit its `enabled` and `reason` fields directly.
The generator obeys the YAML; it does not rerun the review exclusions.

To review display text, initialize a separate checkpoint in `labels` mode:

```powershell
npm.cmd run review:operations -- init --mode labels --work _build/label-review --source _build/native_functions.yaml
node scripts/review-operations.mjs next --size 80 --work _build/label-review
node scripts/review-operations.mjs apply --answers _build/labels.json --work _build/label-review
node scripts/review-operations.mjs finish --work _build/label-review
```

This mode lists only enabled entries, one exact overload per ID, together with
its signature. Answer every batch with objects containing `ids`, `displayName`,
and `description`. For example:

```json
[
  {
    "ids": [0],
    "displayName": "Add Tensors",
    "description": "Add two tensors elementwise, broadcasting compatible shapes."
  }
]
```

Use `revisit --id <id>` to correct a completed item before finishing. Display-text
review changes only the two text fields; categories, enabled flags, exclusion
reasons, and disabled nodes stay unchanged. All 651 currently enabled operations
have reviewed display text. Exact schemas remain available in the generated
manifest for technical reference.

Reference documentation for text review:
[PyTorch operations](https://docs.pytorch.org/docs/stable/torch.html),
[linear algebra](https://docs.pytorch.org/docs/stable/linalg.html), and
[special functions](https://docs.pytorch.org/docs/stable/special.html).

## Deterministic IDs

Implementation IDs are generated from the canonical parsed signature (`schemaSignature`) using SHA-1-derived UUID bytes, then emitted in your `Core::Uuid(...)` constructor format.

## TODO: 

 - Add exclude list support
