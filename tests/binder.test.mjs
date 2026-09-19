import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { bindProgram } from '../dist/binder.js';
import { buildNodeRegistry } from '../dist/registry.js';
import { parseFuncSchema } from '../dist/schemaParser.js';

const registry = buildNodeRegistry([
  'add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor',
  'relu(Tensor self) -> Tensor',
].map(schema => parseFuncSchema(schema, [], [])));
const example = JSON.parse(await readFile(new URL('../examples/add-relu.program.json', import.meta.url), 'utf8'));

test('add/relu binds defaults and producer terminals without changing its input', () => {
  const original = structuredClone(example);
  const bound = bindProgram(example, registry);
  assert.deepEqual(bound.operations[0].inputs, [
    { key: 'Input', argument: 'self', source: { input: 'x', type: 'Tensor' } },
    { key: 'Other', argument: 'other', source: { input: 'y', type: 'Tensor' } },
    { key: 'Alpha (kw_only)', argument: 'alpha', source: { literal: 1 } },
  ]);
  assert.equal(bound.operations[0].implementationId, registry.operations[0].implementationId);
  assert.deepEqual(bound.operations[1].inputs[0].source, { operation: 'add', key: 'Output 0', type: 'Tensor' });
  assert.deepEqual(bound.outputs, [{ name: 'result', source: { operation: 'relu', key: 'Output 0', type: 'Tensor' } }]);
  assert.deepEqual(example, original);
  assert.deepEqual(bindProgram(example, registry), bound);
});

test('keyword arguments can supply ordinary arguments and override defaults', () => {
  const program = structuredClone(example);
  program.operations[0].args = [];
  program.operations[0].kwargs = { self: { value: 'x' }, other: { value: 'y' }, alpha: { literal: 2 } };
  assert.deepEqual(bindProgram(program, registry).operations[0].inputs[2].source, { literal: 2 });
});

test('invalid bindings fail with an operation and source diagnostic', () => {
  const cases = [
    [p => p.operations[0].operator = 'aten.add.Scalar', /not present/],
    [p => p.operations[0].args.pop(), /missing argument other/],
    [p => p.operations[0].args.push({ literal: 1 }), /too many positional/],
    [p => p.operations[0].kwargs = { other: { value: 'y' } }, /supplied twice/],
    [p => p.operations[0].kwargs = { typo: { literal: 1 } }, /unknown keyword/],
    [p => p.operations[0].args[0] = { value: 'sum' }, /unknown or forward/],
    [p => p.operations[0].args[0] = { literal: null }, /None is not allowed/],
    [p => p.operations[0].args[0] = { literal: 1 }, /invalid literal/],
    [p => p.inputs[0].type = 'Int', /conversion required/],
    [p => p.operations[0].kwargs = { alpha: { literal: Infinity } }, /invalid literal/],
    [p => p.operations[0].kwargs = { alpha: { literal: 9007199254740992 } }, /invalid literal/],
    [p => p.operations[0].outputs = [], /expected 1 output/],
  ];
  for (const [mutate, expected] of cases) {
    const program = structuredClone(example);
    program.operations[0].source = 'model.py:7';
    mutate(program);
    assert.throws(() => bindProgram(program, registry), error => {
      assert.match(error.message, /add \(aten\.add\./);
      assert.match(error.message, /model\.py:7/);
      assert.match(error.message, expected);
      return true;
    });
  }
});

test('duplicate identities and missing graph outputs are rejected', () => {
  const cases = [
    [p => p.version = 2, /version/],
    [p => p.inputs.push(p.inputs[0]), /duplicate value/],
    [p => p.operations[1].id = 'add', /duplicate operation/],
    [p => p.operations[0].outputs = ['x'], /duplicate value/],
    [p => p.outputs[0].value = 'missing', /Unknown output/],
    [p => p.outputs.push(p.outputs[0]), /duplicate output/],
  ];
  for (const [mutate, expected] of cases) {
    const program = structuredClone(example);
    mutate(program);
    assert.throws(() => bindProgram(program, registry), expected);
  }
  assert.throws(() => bindProgram(example, { version: 1, operations: [...registry.operations, registry.operations[0]] }), /Duplicate registry/);
  const ambiguous = structuredClone(registry);
  ambiguous.operations[0].inputs[1].key = 'Input';
  assert.throws(() => bindProgram(example, ambiguous), /ambiguous terminal/);
});

test('named multi-output ports preserve return positions and optional None defaults', () => {
  const registry = buildNodeRegistry([parseFuncSchema('example(Tensor self, Tensor? other=None) -> (Tensor values, Tensor indices)', [], [])]);
  const bound = bindProgram({ version: 1, inputs: [{ id: 'x', type: 'Tensor' }], operations: [
    { id: 'split', operator: 'aten.example.default', args: [{ value: 'x' }], outputs: ['v', 'i'] },
  ], outputs: [{ name: 'indices', value: 'i' }, { name: 'values', value: 'v' }] }, registry);
  assert.deepEqual(bound.operations[0].inputs[1].source, { literal: null });
  assert.deepEqual(bound.outputs.map(o => o.source.key), ['Indices', 'Values']);
});

test('unsupported schemas and default expressions fail rather than approximate', () => {
  for (const [schema, expected] of [
    ['example(Tensor self, int[] dims=[]) -> Tensor', /unsupported default|unsupported port/],
    ['example(Tensor self, Scalar alpha=some_expression) -> Tensor', /unsupported default/],
    ['example(Tensor(a!) self) -> Tensor', /unsupported port/],
    ['example(Tensor self) -> Tensor?', /optional output/],
  ]) {
    const registry = buildNodeRegistry([parseFuncSchema(schema, [], [])]);
    assert.throws(() => bindProgram({ version: 1, inputs: [{ id: 'x', type: 'Tensor' }], operations: [
      { id: 'op', operator: 'aten.example.default', args: [{ value: 'x' }], outputs: ['y'] },
    ], outputs: [{ name: 'y', value: 'y' }] }, registry), expected);
  }
});
