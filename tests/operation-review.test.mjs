import assert from 'node:assert/strict';
import test from 'node:test';
import YAML from 'yaml';
import { acceptBatch, createReview, exclusion, renderReview } from '../scripts/review-operations.mjs';
import { parseFuncSchema } from '../dist/schemaParser.js';

const catalog = `# Keep this comment
version: 1
defaults:
  namespace: PyTorch::Uncategorized
operations:
  add.Tensor:
    namespace: PyTorch::MyArithmetic
    displayName: My Add
  add.out: {}
  add_.Tensor: {}
  relu:
    enabled: false
    reason: My manual exclusion
`;
const source = `
- func: add.Tensor(Tensor self, Tensor other) -> Tensor
- func: add.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
- func: add_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
- func: relu(Tensor self) -> Tensor
`;

test('review groups overloads but excludes unsupported variants independently', () => {
  const state = createReview(catalog, source);
  assert.equal(state.families.length, 2);
  assert.equal(state.families[0].operations.length, 3);
  assert.equal(state.families[0].operations[0].reason, undefined);
  assert.match(state.families[0].operations[1].reason, /output-buffer/);
  assert.match(state.families[0].operations[2].reason, /In-place/);
  assert.throws(() => createReview(catalog, '- func: relu(Tensor self) -> Tensor'), /Schema missing/);
});

test('batch responses require exact coverage and do not mutate state on rejection', () => {
  const state = { ...createReview(catalog, source), pending: [0, 1] };
  for (const groups of [
    [{ namespace: 'PyTorch::Math', ids: [0] }],
    [{ namespace: 'PyTorch::Math', ids: [0, 0] }],
    [{ namespace: 'PyTorch::Math', ids: [0, 2] }],
    [{ namespace: 'PyTorch::Uncategorized', ids: [0, 1] }],
    [{ namespace: 'PyTorch::Math', ids: [0, 1], enabled: false }],
  ]) assert.throws(() => acceptBatch(state, groups));
  assert.deepEqual(state.decisions, {});
  const accepted = acceptBatch(state, [{ namespace: 'PyTorch::Math', ids: [0, 1] }]);
  assert.equal(Object.keys(accepted.decisions).length, 2);
  assert.deepEqual(accepted.pending, []);
});

test('finish preserves manual metadata and comments, requires complete review and unchanged source', () => {
  const state = { ...createReview(catalog, source), pending: [0, 1] };
  assert.throws(() => renderReview(state, catalog), /incomplete/);
  const complete = acceptBatch(state, [{ namespace: 'PyTorch::BasicOperators', ids: [0, 1] }]);
  assert.throws(() => renderReview(complete, catalog + '\n'), /changed/);
  const { text, summary } = renderReview(complete, catalog);
  const result = YAML.parse(text).operations;
  assert.ok(text.includes('# Keep this comment'));
  assert.equal(result['add.Tensor'].namespace, 'PyTorch::MyArithmetic');
  assert.equal(result['add.Tensor'].displayName, 'My Add');
  assert.equal(result['add.Tensor'].enabled, undefined);
  assert.equal(result['add.out'].enabled, false);
  assert.equal(result['add_.Tensor'].enabled, false);
  assert.equal(result.relu.reason, 'My manual exclusion');
  assert.equal(summary.disabled, 3);
  assert.equal(summary.total, 4);
});

test('schema exclusions detect list, method-only, no-return and unsupported typed ports', () => {
  for (const [schema, variants, expected] of [
    ['cat(Tensor[] tensors) -> Tensor', ['function'], /List-valued/],
    ['foo(Tensor self) -> Tensor', ['method'], /Method-only/],
    ['foo(Tensor self) -> ()', ['function'], /No return/],
    ['foo(Tensor self, Device device) -> Tensor', ['function'], /Device/],
    ['foo(Tensor self) -> Tensor?', ['function'], /Optional return/],
  ]) assert.match(exclusion(parseFuncSchema(schema, [], variants)), expected);
});

test('reviewer can disable a whole family with an explicit reason', () => {
  const state = { ...createReview(catalog, source), pending: [0, 1] };
  const complete = acceptBatch(state, [{ namespace: 'PyTorch::Math', ids: [0, 1], reason: 'Not appropriate for this palette.' }]);
  const result = YAML.parse(renderReview(complete, catalog).text).operations;
  assert.equal(result['add.Tensor'].enabled, false);
  assert.equal(result['add.Tensor'].reason, 'Not appropriate for this palette.');
  assert.equal(result.relu.reason, 'My manual exclusion');
});

test('label review covers enabled overloads individually and preserves all other metadata', () => {
  const state = createReview(catalog, source, 'labels');
  assert.equal(state.families.length, 3);
  assert.ok(state.families.every(item => item.operations.length === 1));
  state.pending = state.families.map(item => item.id);
  const groups = state.families.map(item => ({ ids: [item.id], displayName: `Name ${item.name}`, description: `Description for ${item.name}.` }));
  const complete = acceptBatch(state, groups);
  const { text, summary } = renderReview(complete, catalog);
  const before = YAML.parse(catalog).operations;
  const after = YAML.parse(text).operations;
  assert.deepEqual(summary, { labeled: 3 });
  assert.deepEqual(after.relu, before.relu);
  for (const op of ['add.Tensor', 'add.out', 'add_.Tensor']) {
    assert.equal(after[op].displayName, `Name ${op}`);
    assert.equal(after[op].description, `Description for ${op}.`);
    assert.equal(after[op].namespace, before[op].namespace);
    assert.equal(after[op].enabled, before[op].enabled);
    assert.equal(after[op].reason, before[op].reason);
  }
  assert.ok(text.includes('# Keep this comment'));
});

test('label review rejects missing text, invalid fields, and partial coverage', () => {
  const state = createReview(catalog, source, 'labels');
  state.pending = [0];
  for (const group of [
    { ids: [0], displayName: 'Name' },
    { ids: [0], displayName: ' ', description: 'Description' },
    { ids: [0], displayName: 'Name', description: 42 },
    { ids: [0], displayName: 'Name', description: 'Description', enabled: false },
    { ids: [], displayName: 'Name', description: 'Description' },
  ]) assert.throws(() => acceptBatch(state, [group]));
  assert.deepEqual(state.decisions, {});
});
