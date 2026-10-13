# Tora development guidelines

Keep these guidelines general. Feature details, commands, and examples belong in
the README or code rather than in this file.

## Responsibilities and design

- Tora owns PyTorch capture, the intermediate program representation, operation
  binding, C++ extension generation, and NodeGraph export. Keep these stages
  separate so they can be tested independently.
- The Node Editor in `gemmafusionsuite` owns graph import/export, execution, and
  terminal UI. Inspect its implementation when matching its format or behavior.
  Do not assume a change in Tora also changes the editor.
- Share operation IDs, terminal mappings, and type mappings between generated
  C++ and the registry. Do not maintain independent guesses in the exporter.
- Put per-operation choices in the YAML operation configuration when supported.
  Validate new configuration fields; do not silently ignore them.
- Keep identity separate from presentation. Preserve implementation IDs and
  current terminal keys. Edges reference keys through `UT` and `VT`; node
  instances have their own UUIDs. The terminal key/display-name migration is
  deferred, so current terminal names remain keys.
- Preserve compatibility with existing graphs. Changes to saved formats, IDs,
  or keys need an explicit compatibility strategy
- Make conversions and defaults explicit. Use the editor's verified native
  conversions where applicable and conversion nodes otherwise. Reject unsupported
  operations or values with useful diagnostics rather than approximating them.
- Store model weights in external tensor files; graph JSON stores references.
  Do not assume processors know the graph directory or resolve paths against an
  incidental process working directory.

## Coding style

- Match the surrounding file's formatting. Avoid unrelated reformatting, line
  ending changes, renaming, or cleanup.
- Use descriptive names and small functions with one clear responsibility.
  Comments explain intent, constraints, or non-obvious behavior rather than
  repeating the code. Keep user-visible names and errors plain and specific.
- TypeScript: use strict typing, ES modules, two-space indentation, semicolons,
  and double-quoted strings in source files. Use `.js` extensions in relative
  imports and `import type` for type-only imports. Use `camelCase` for functions
  and variables, `PascalCase` for types, and `UPPER_SNAKE_CASE` for module constants.
  Validate external data as `unknown` before treating it as a typed value; avoid
  `any` and unchecked casts at input boundaries.
- JavaScript tests and example scripts: use ES modules, two-space indentation,
  semicolons, and the existing single-quote convention.
- Python: use four-space indentation, `snake_case` for functions and variables,
  and `PascalCase` for classes. Preserve the file's quote style. Keep command-line
  execution behind a main guard; avoid work merely from importing a module.
- C++: target the project's C++20 conventions. Use four-space indentation in new
  processor headers; preserve tabs in existing files that use them. Follow nearby
  brace placement. Use `PascalCase` for processor types and public helper methods,
  and existing SDK names/signatures for bridge methods such as `name()`,
  `implementation_id()`, and `Process()`. Make ownership and lifetime explicit;
  array views must not outlive their backing storage. Prefer RAII containers and
  owned tensor storage over manual allocation.
- YAML: use two-space indentation and exact operation/overload keys. Metadata
  namespaces use C++ syntax, for example `PyTorch::Convolution`.

## Workflow and validation

- Work in small, reviewable steps. Preserve unrelated user edits. Ask when a
  material design decision is unresolved; handle routine implementation choices
  using the existing conventions.
- Change generator sources rather than hand-editing generated files. Keep
  reproducible examples in `examples/` and temporary output in `_build/`.
- Run checks appropriate to the change. Use `npm test` for TypeScript changes,
  capture tests for the Python frontend, and native builds/tests for C++ changes.
  Test compatibility boundaries, defaults, and failure cases where relevant.
- Distinguish serialization tests, numerical checks in PyTorch, native extension
  tests, and actual execution in the editor. Report what was verified and what
  remains unverified.
- Commit completed, tested steps when requested. Match the recent imperative
  commit style, keep messages concise, and always end the subject with a period.
  Prefer wording with `on`, `to`, `from`, or `in` over long adjective-heavy names.
  Example: `Add capture from PyTorch modules to programs.`
