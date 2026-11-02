[← Tora](../README.md)

# Program format and graph export

Run commands from the repository root. See [capture](capture.md) for the Python frontend
and [generation](generation.md) for the matching extension and registry.

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
Fixed integer sequences use `{ "items": [{ "literal": 2 }, { "literal": 1 }] }`.
Elements may also reference integer values using `{ "value": "id" }`. They bind
only to configured fixed-size scalar ports; nested or variable-length lists,
mutation, optional outputs, conversions, control flow,
and symbolic constraints are not supported in this first binding pass.

The binder result is an intermediate plan. `exportNodeGraph()` in
`src/graphExporter.ts` lowers that plan to the editor's `Nodes`, `Edges`, and
`NodeBases` format. The [Python adapter](capture.md) supplies this representation from a captured module.

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
