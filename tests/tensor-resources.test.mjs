import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { bindProgram } from '../dist/binder.js';
import { parseProgramDocument, parseRegistryDocument } from '../dist/graphDocuments.js';
import { exportNodeGraph } from '../dist/graphExporter.js';
import { buildNodeRegistry } from '../dist/registry.js';
import { parseFuncSchema } from '../dist/schemaParser.js';
import { renderTensorLoaderBinding } from '../dist/tensorResources.js';
import { uuidToCppCtor } from '../dist/naming.js';

const registry = buildNodeRegistry([
  parseFuncSchema('add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor', [], []),
]);
const program = {
  version: 2,
  inputs: [{ id: 'x', type: 'Tensor' }],
  resources: [{ id: 'weight', type: 'Tensor', file: 'model.safetensors', key: 'layer.weight' }],
  operations: [{ id: 'add', operator: 'aten.add.Tensor', args: [{ value: 'x' }, { value: 'weight' }], outputs: ['y'] }],
  outputs: [{ name: 'result', value: 'y' }, { name: 'weight', value: 'weight' }],
};
const directory = path.resolve('_build');

test('resource binding and export use shared loader metadata and explicit paths', () => {
  const p = structuredClone(program);
  assert.equal(parseProgramDocument(p), p);
  const bound = bindProgram(p, registry);
  assert.deepEqual(bound.operations[0].inputs[1].source, { resource: 'weight', type: 'Tensor' });
  const graph = exportNodeGraph(p, registry, { resourceDirectory: directory });
  const loader = graph.Nodes['1'];
  assert.equal(loader.NodeBase, registry.tensorLoader.implementationId);
  assert.deepEqual(loader.Properties, { File: 'model.safetensors', 'Tensor Name': 'layer.weight', 'Base Directory': directory });
  assert.ok(Object.values(graph.Edges).some(e => e.U === '1' && e.UT === 'Tensor' && e.VT === 'Other'));
  assert.ok(Object.values(graph.Edges).some(e => e.U === '1' && e.UT === 'Tensor' && e.VT === 'weight'));
  assert.deepEqual(Object.keys(graph.Nodes['0'].Metadata.Properties), ['x']);
  assert.deepEqual(p, program);
  for (const edge of Object.values(graph.Edges)) {
    assert.ok(Object.hasOwn(graph.NodeBases[graph.Nodes[edge.U].NodeBase].Outputs, edge.UT));
    assert.ok(Object.hasOwn(graph.NodeBases[graph.Nodes[edge.V].NodeBase].Inputs, edge.VT));
  }
  assert.throws(() => exportNodeGraph(p, registry), /absolute resourceDirectory/);
  assert.throws(() => exportNodeGraph(p, registry, { resourceDirectory: 'relative' }), /absolute resourceDirectory/);
  p.resources[0].file = path.join(directory, 'model.safetensors');
  assert.equal(exportNodeGraph(p, registry).Nodes['1'].Properties.File, p.resources[0].file);
  if (process.platform === 'win32') {
    for (const file of ['C:model.safetensors', '\\model.safetensors']) {
      p.resources[0].file = file;
      assert.throws(() => exportNodeGraph(p, registry, { resourceDirectory: directory }), /drive\/root prefix/);
    }
  }
});

test('resources can be the only values and multiple loads share their base', () => {
  const p = structuredClone(program);
  p.inputs = [];
  p.operations = [];
  p.resources.push({ ...p.resources[0], id: 'bias', key: 'layer.bias' });
  p.outputs = p.resources.map(r => ({ name: r.id, value: r.id }));
  const graph = exportNodeGraph(p, registry, { resourceDirectory: directory });
  assert.equal(Object.keys(graph.Nodes).length, 3);
  assert.equal(graph.Nodes['0'].NodeBase, graph.Nodes['1'].NodeBase);
  assert.notEqual(graph.Nodes['0'].InstanceUuid, graph.Nodes['1'].InstanceUuid);
});

test('resource validation rejects malformed documents, collisions and stale registries', () => {
  for (const mutate of [
    p => p.version = 1,
    p => p.resources = {},
    p => p.resources[0].type = 'Float',
    p => p.resources[0].key = '',
    p => p.resources[0].file = 'bad\0file',
    p => p.resources[0].typo = true,
  ]) {
    const p = structuredClone(program); mutate(p);
    assert.throws(() => parseProgramDocument(p), /program.resources/);
  }
  for (const id of ['x', 'y', 'weight']) {
    const p = structuredClone(program);
    p.resources.push({ ...p.resources[0], id });
    assert.throws(() => bindProgram(p, registry), /duplicate value/);
  }
  const oldRegistry = structuredClone(registry);
  delete oldRegistry.tensorLoader;
  assert.equal(parseRegistryDocument(oldRegistry), oldRegistry);
  assert.throws(() => exportNodeGraph(program, oldRegistry), /regenerate the registry/);
  for (const mutate of [
    r => r.tensorLoader.implementationId = 'wrong',
    r => r.tensorLoader.inputs.file.key = 'Renamed',
    r => r.tensorLoader.outputs.tensor.editorType.ValueType = 5,
  ]) {
    const r = structuredClone(registry); mutate(r);
    assert.throws(() => parseRegistryDocument(r), /registry.tensorLoader/);
  }
});

test('loader C++ definitions and registry share the existing identity and ports', () => {
  const cpp = renderTensorLoaderBinding();
  assert.ok(cpp.includes(uuidToCppCtor('3a6a2744-f014-45d9-bb93-3d96c9f6b152')));
  for (const port of [...Object.values(registry.tensorLoader.inputs), ...Object.values(registry.tensorLoader.outputs)]) {
    assert.ok(cpp.includes(port.cppPropertyType));
  }
});

test('CLI resolves resources beside the program, regardless of output location, and protects tensors', async t => {
  const root = await mkdtemp(path.join(tmpdir(), 'tora-resources-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const programPath = path.join(root, 'program.json');
  const registryPath = path.join(root, 'registry.json');
  const outputPath = path.join(root, 'elsewhere', 'graph.json');
  const tensorPath = path.join(root, 'model.safetensors');
  await writeFile(programPath, JSON.stringify(program));
  await writeFile(registryPath, JSON.stringify(registry));
  await writeFile(tensorPath, 'preserve');
  const args = ['dist/graphCli.js', '--program', programPath, '--registry', registryPath, '--out', outputPath];
  const result = spawnSync(process.execPath, args, { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const graph = JSON.parse(await readFile(outputPath, 'utf8'));
  assert.equal(graph.Nodes['1'].Properties['Base Directory'], root);
  args[args.length - 1] = tensorPath;
  const failed = spawnSync(process.execPath, args, { encoding: 'utf8' });
  assert.equal(failed.status, 1);
  assert.match(failed.stderr, /must not overwrite a tensor resource/);
  assert.equal(await readFile(tensorPath, 'utf8'), 'preserve');
});
