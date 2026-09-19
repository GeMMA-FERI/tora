import assert from 'node:assert/strict';
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { exportNodeGraph } from '../dist/graphExporter.js';
import { parseProgramDocument, parseRegistryDocument } from '../dist/graphDocuments.js';
import { buildNodeRegistry } from '../dist/registry.js';
import { parseFuncSchema } from '../dist/schemaParser.js';

const registry = buildNodeRegistry([
  'add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor',
  'relu(Tensor self) -> Tensor',
  'example(Tensor self, Tensor? other=None) -> Tensor',
].map(schema => parseFuncSchema(schema, [], [])));
const program = JSON.parse(await readFile(new URL('../examples/add-relu.program.json', import.meta.url), 'utf8'));

function checkEdges(graph) {
  for (const edge of Object.values(graph.Edges)) {
    const from = graph.Nodes[edge.U];
    const to = graph.Nodes[edge.V];
    assert.ok(Object.hasOwn(graph.NodeBases[from.NodeBase].Outputs, edge.UT));
    assert.ok(Object.hasOwn(graph.NodeBases[to.NodeBase].Inputs, edge.VT));
    assert.equal(Object.hasOwn(to.Properties, edge.VT), false);
  }
}

test('add/relu emits editor nodes, defaults, typed snapshots, and exact edge keys', () => {
  const graph = exportNodeGraph(program, registry);
  assert.equal(Object.keys(graph.Nodes).length, 4);
  assert.equal(Object.keys(graph.NodeBases).length, 4);
  assert.deepEqual(Object.values(graph.Edges), [
    { U: '0', V: '1', UT: 'x', VT: 'Input' },
    { U: '0', V: '1', UT: 'y', VT: 'Other' },
    { U: '1', V: '2', UT: 'Output 0', VT: 'Input' },
    { U: '2', V: '3', UT: 'Output 0', VT: 'result' },
  ]);
  assert.deepEqual(graph.Nodes['1'].Properties, { 'Alpha (kw_only)': 1 });
  assert.equal(graph.Nodes['0'].Metadata.Properties.x.ClassName, 'struct Data::Tensor');
  assert.equal(graph.NodeBases[graph.Nodes['1'].NodeBase].Inputs.Input.ValueType, 17);
  assert.equal(new Set(Object.values(graph.Nodes).map(n => n.InstanceUuid)).size, 4);
  assert.deepEqual(Object.values(graph.Nodes).map(n => n.Bounds[0]), [0, 320, 640, 960]);
  checkEdges(graph);
  assert.deepEqual(JSON.parse(JSON.stringify(graph)), graph);
});

test('repeated implementations retain separate instances and share snapshots', () => {
  const p = structuredClone(program);
  p.operations.push({ id: 'relu_again', operator: 'aten.relu.default', args: [{ value: 'result' }], outputs: ['again'] });
  p.outputs = [{ name: 'first', value: 'result' }, { name: 'second', value: 'again' }];
  const graph = exportNodeGraph(p, registry);
  assert.equal(Object.keys(graph.Nodes).length, 5);
  assert.equal(Object.keys(graph.NodeBases).length, 4);
  assert.equal(graph.Nodes['2'].NodeBase, graph.Nodes['3'].NodeBase);
  assert.notEqual(graph.Nodes['2'].InstanceUuid, graph.Nodes['3'].InstanceUuid);
  checkEdges(graph);
});

test('optional None is omitted and direct graph input/output wiring works', () => {
  const p = { version: 1, inputs: [{ id: 'x', type: 'Tensor' }], operations: [
    { id: 'op', operator: 'aten.example.default', args: [{ value: 'x' }], outputs: ['y'] },
  ], outputs: [{ name: 'y', value: 'y' }] };
  assert.deepEqual(exportNodeGraph(p, registry).Nodes['1'].Properties, {});
  p.operations = [];
  p.outputs = [{ name: 'identity', value: 'x' }];
  const graph = exportNodeGraph(p, registry);
  assert.deepEqual(graph.Edges['0'], { U: '0', V: '1', UT: 'x', VT: 'identity' });
  checkEdges(graph);
});

test('document validation catches malformed programs and stale registries', () => {
  assert.equal(parseProgramDocument(program), program);
  assert.equal(parseRegistryDocument(registry), registry);
  for (const mutate of [
    p => p.operations[0].args[0] = { value: 'x', literal: 1 },
    p => p.operations[0].kwargs = [],
    p => p.operations[0].outputs = 'sum',
    p => p.operations[0].typo = true,
    p => p.inputs[0].type = 'Object',
  ]) {
    const p = structuredClone(program); mutate(p);
    assert.throws(() => parseProgramDocument(p), /program\./);
  }
  const r = structuredClone(registry);
  r.operations[0].inputs[0].editorType.ValueType = 5;
  assert.throws(() => parseRegistryDocument(r), /does not match editor profile/);
  delete r.editorProfile;
  assert.throws(() => parseRegistryDocument(r), /regenerate/);
  assert.throws(() => exportNodeGraph(program, r), /regenerate/);
});

test('CLI exports JSON and preserves existing output on binding failure', async t => {
  const dir = await mkdtemp(path.join(tmpdir(), 'tora-graph-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const programPath = path.join(dir, 'program.json');
  const registryPath = path.join(dir, 'registry.json');
  const outputPath = path.join(dir, 'graph.json');
  await writeFile(programPath, JSON.stringify(program));
  await writeFile(registryPath, JSON.stringify(registry));
  const args = ['dist/graphCli.js', '--program', programPath, '--registry', registryPath, '--out', outputPath];
  const run = spawnSync(process.execPath, args, { encoding: 'utf8' });
  assert.equal(run.status, 0, run.stderr);
  const text = await readFile(outputPath, 'utf8');
  checkEdges(JSON.parse(text));
  const bad = structuredClone(program);
  bad.operations[0].operator = 'aten.missing.default';
  await writeFile(programPath, JSON.stringify(bad));
  const failed = spawnSync(process.execPath, args, { encoding: 'utf8' });
  assert.equal(failed.status, 1);
  assert.match(failed.stderr, /not present in the generated registry/);
  assert.equal(await readFile(outputPath, 'utf8'), text);
  args[args.length - 1] = registryPath;
  const overwrite = spawnSync(process.execPath, args, { encoding: 'utf8' });
  assert.equal(overwrite.status, 1);
  assert.match(overwrite.stderr, /must not overwrite/);
});
