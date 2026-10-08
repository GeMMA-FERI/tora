import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import YAML from 'yaml';
import { parseFuncSchema } from '../dist/schemaParser.js';
import { filterSubset } from '../dist/subset.js';
import { defaultConfig } from '../dist/config.js';
import { buildNodeRegistry } from '../dist/registry.js';
import { renderNodeDefinition } from '../dist/renderer.js';
import { bindProgram } from '../dist/binder.js';
import { exportNodeGraph } from '../dist/graphExporter.js';
import { parseProgramDocument, parseRegistryDocument } from '../dist/graphDocuments.js';
import { signatureUuid } from '../dist/naming.js';
import { createReview } from '../scripts/review-operations.mjs';

const schemas = [1, 2].map(n => `conv${n}d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[${n}] stride=1, SymInt[${n}] padding=0, SymInt[${n}] dilation=1, SymInt groups=1) -> Tensor`);
const functions = await filterSubset(schemas.map(s => parseFuncSchema(s, [], [])), defaultConfig.subset, undefined, 'config/operations.yaml');
const registry = buildNodeRegistry(functions);
const base = { version: 1, inputs: [{ id: 'x', type: 'Tensor' }, { id: 'w', type: 'Tensor' }], operations: [
  { id: 'conv', operator: 'aten.conv2d.default', args: [{ value: 'x' }, { value: 'w' }], outputs: ['y'] },
], outputs: [{ name: 'y', value: 'y' }] };

test('fixed list sizes and configured integer ports survive generation', () => {
  assert.equal(functions[0].args[3].listLength, 1);
  assert.equal(functions[1].args[3].listLength, 2);
  assert.equal(parseFuncSchema('example(int[] x) -> Tensor', [], []).args[0].listLength, undefined);
  const op = registry.operations[1];
  assert.equal(registry.version, 2);
  assert.deepEqual(op.inputs.map(p => p.key), ['Input', 'Weight', 'Bias (optional)', 'Stride Height', 'Stride Width',
    'Padding Height', 'Padding Width', 'Dilation Height', 'Dilation Width', 'Groups']);
  assert.deepEqual(op.inputs.filter(p => p.argument === 'stride').map(p => p.split), [{ index: 0, length: 2 }, { index: 1, length: 2 }]);
  assert.equal(op.implementationId, signatureUuid(functions[1].schemaSignature));
  const cpp = renderNodeDefinition(functions[1]);
  assert.match(cpp, /std::array<int64_t, 2> torch_stride/);
  assert.match(cpp, /RegisterInput<In_stride_0>\(::Data::Variant\(static_cast<Core::Int64>\(1\)\)\)/);
  assert.match(cpp, /RegisterInput<In_padding_1>\(::Data::Variant\(static_cast<Core::Int64>\(0\)\)\)/);
  assert.match(cpp, /RegisterInput<In_groups>\(::Data::Variant\(static_cast<Core::Int64>\(1\)\)\)/);
  for (const p of op.inputs) assert.ok(cpp.includes(p.cppPropertyType));
  parseRegistryDocument(JSON.parse(JSON.stringify(registry)));
});

test('scalar expansion, exact sequences and defaults bind to the same ordered ports', () => {
  const p = structuredClone(base);
  p.operations[0].kwargs = { stride: { literal: 2 }, padding: { items: [{ literal: 1 }, { literal: 3 }] } };
  const bound = bindProgram(parseProgramDocument(p), registry);
  const props = Object.fromEntries(bound.operations[0].inputs.filter(p => 'literal' in p.source).map(p => [p.key, p.source.literal]));
  assert.deepEqual(props, { 'Bias (optional)': null, 'Stride Height': 2, 'Stride Width': 2,
    'Padding Height': 1, 'Padding Width': 3, 'Dilation Height': 1, 'Dilation Width': 1, Groups: 1 });
  const graph = exportNodeGraph(p, registry);
  assert.equal(graph.Nodes['1'].Properties['Stride Width'], 2);
  assert.equal(graph.NodeBases[graph.Nodes['1'].NodeBase].Inputs['Stride Height'].ValueType, 5);
  const one = structuredClone(base);
  one.operations[0].operator = 'aten.conv1d.default';
  assert.equal(bindProgram(one, registry).operations[0].inputs.find(p => p.key === 'Stride').source.literal, 1);
});

test('each sequence element can connect to its own integer value', () => {
  const p = structuredClone(base);
  p.inputs.push({ id: 'height', type: 'Int' });
  p.operations[0].kwargs = { stride: { items: [{ value: 'height' }, { literal: 1 }] } };
  const graph = exportNodeGraph(p, registry);
  assert.ok(Object.values(graph.Edges).some(e => e.UT === 'height' && e.VT === 'Stride Height'));
  assert.equal(graph.Nodes['1'].Properties['Stride Width'], 1);
});

test('wrong lengths, element types, nested sequences and bad registry mappings fail', () => {
  for (const stride of [{ items: [{ literal: 1 }] }, { items: [{ literal: 1 }, { literal: 2.5 }] }, { literal: null }, { literal: true }]) {
    const p = structuredClone(base); p.operations[0].kwargs = { stride };
    assert.throws(() => bindProgram(p, registry), /elements|invalid literal|None/);
  }
  const p = structuredClone(base);
  p.operations[0].kwargs = { stride: { items: [{ items: [] }] } };
  assert.throws(() => parseProgramDocument(p), /flat items/);
  const r = structuredClone(registry); r.operations[1].inputs[4].split.index = 0;
  assert.throws(() => bindProgram(base, r), /invalid split-port/);
  r.version = 1;
  assert.throws(() => parseRegistryDocument(r), /invalid split mapping/);
});

test('YAML mappings reject typos, invalid targets, lengths and duplicate keys', async t => {
  const dir = await mkdtemp(path.join(tmpdir(), 'tora-split-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  for (const argumentsValue of [
    { typo: { ports: [{ name: 'X' }] } },
    { input: { ports: [{ name: 'X' }] } },
    { stride: { ports: [{ name: 'X' }] } },
    { stride: { ports: [{ name: 'X' }, { name: 'X' }] } },
    { stride: { ports: [{ name: 'Input' }, { name: 'Width' }] } },
    { stride: { ports: [{ label: 'X' }, { name: 'Y' }] } },
  ]) {
    const file = path.join(dir, 'ops.yaml');
    await writeFile(file, YAML.stringify({ version: 1, operations: { conv2d: { arguments: argumentsValue } } }));
    await assert.rejects(filterSubset([parseFuncSchema(schemas[1], [], [])], defaultConfig.subset, undefined, file), /ops.yaml/);
  }
});

test('operation review retains configured convolution support', async () => {
  const catalog = YAML.parse(await readFile('config/operations.yaml', 'utf8'));
  const selected = { version: 1, operations: { conv1d: catalog.operations.conv1d, conv2d: catalog.operations.conv2d } };
  const review = createReview(YAML.stringify(selected), YAML.stringify(schemas.map(func => ({ func }))));
  assert.ok(review.families.every(f => f.operations.every(op => op.reason === undefined)));
});

test('legacy registries without split mappings still bind ordinary nodes', () => {
  const legacy = buildNodeRegistry([parseFuncSchema('relu(Tensor self) -> Tensor', [], [])]);
  legacy.version = 1;
  const p = { version: 1, inputs: [{ id: 'x', type: 'Tensor' }], operations: [
    { id: 'relu', operator: 'aten.relu.default', args: [{ value: 'x' }], outputs: ['y'] },
  ], outputs: [{ name: 'y', value: 'y' }] };
  assert.equal(bindProgram(p, parseRegistryDocument(legacy)).operations.length, 1);
});
