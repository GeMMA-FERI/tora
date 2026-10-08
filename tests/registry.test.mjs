import assert from 'node:assert/strict';
import test from 'node:test';
import { parseFuncSchema } from '../dist/schemaParser.js';
import { buildNodeRegistry } from '../dist/registry.js';
import { renderNodeDefinition } from '../dist/renderer.js';
import { uuidToCppCtor } from '../dist/naming.js';

test('registry binds overloads, defaults, and exact generated terminal keys', () => {
  const fn = parseFuncSchema('add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor', [], []);
  const registry = buildNodeRegistry([fn]);
  assert.equal(registry.version, 2);
  const op = registry.operations[0];
  assert.equal(op.operator, 'aten.add.Tensor');
  assert.deepEqual(op.inputs.map(p => [p.argument, p.key]), [
    ['self', 'Input'], ['other', 'Other'], ['alpha', 'Alpha (kw_only)'],
  ]);
  assert.equal(op.inputs[2].defaultExpression, '1');
  assert.equal(op.inputs[2].keywordOnly, true);
  assert.equal(op.outputs[0].key, 'Output 0');
  const cpp = renderNodeDefinition(fn);
  assert.ok(cpp.includes(uuidToCppCtor(op.implementationId)));
  for (const port of [...op.inputs, ...op.outputs]) {
    assert.ok(cpp.includes(port.cppPropertyType));
    assert.ok(port.cppPropertyType.includes(`"${port.key}"`));
  }
});

test('registry preserves optional lists, named returns, and default overloads', () => {
  const fn = parseFuncSchema('example(Tensor self, *, int[]? dims=None) -> (Tensor values, Tensor indices)', [], []);
  const op = buildNodeRegistry([fn]).operations[0];
  assert.equal(op.operator, 'aten.example.default');
  assert.equal(op.inputs[1].key, 'Dimensions (optional,list,kw_only)');
  assert.equal(op.inputs[1].defaultExpression, 'None');
  assert.equal(op.inputs[1].optional, true);
  assert.equal(op.inputs[1].list, true);
  assert.deepEqual(op.outputs.map(p => [p.index, p.returnName, p.key]), [
    [0, 'values', 'Values'], [1, 'indices', 'Indices'],
  ]);
  assert.deepEqual(buildNodeRegistry([fn]), buildNodeRegistry([fn]));
});
