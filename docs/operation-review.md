[← Tora](../README.md)

# Reviewing operation categories and display text

Run commands from the repository root.

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
schemas, no-return operations, unmapped list-valued ports, optional returns, and argument
types without a corresponding bridge conversion. This is a conservative
forward-processing palette, not a claim that those PyTorch operations are bad.
Configured Conv1d/Conv2d scalar mappings are retained during review; other
convolution and pooling variants may remain disabled until their argument types
are supported. Remaining enabled nodes are candidates, not proof of
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
