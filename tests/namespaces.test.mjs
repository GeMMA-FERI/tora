import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import test from 'node:test';
import YAML from 'yaml';
import { defaultConfig } from '../dist/config.js';
import { runGenerator } from '../dist/generator.js';
import { parseFuncSchema } from '../dist/schemaParser.js';
import { filterSubset } from '../dist/subset.js';
import { makeNodeClassName } from '../dist/naming.js';
import { renderNodeDefinition } from '../dist/renderer.js';

async function fixture(t, allowlist) {
  const dir = await mkdtemp(path.join(tmpdir(), 'tora-namespaces-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const allowlistPath = path.join(dir, 'ops.yaml');
  await writeFile(allowlistPath, allowlist);
  return { dir, allowlistPath };
}

test('exact overload namespaces, defaults, exclusions, and limits', async t => {
  const { allowlistPath } = await fixture(t, `# Catalog
version: 1
defaults:
  namespace: PyTorch::Default
operations:
  add.Tensor:
    namespace: PyTorch::BasicOperators # Inline comment
  add.Scalar:
    namespace: PyTorch::ScalarOperators
  relu: {}
  sigmoid:
    enabled: false
    reason: Disabled for this test
`);
  const functions = [
    'add.Tensor(Tensor self, Tensor other) -> Tensor',
    'add.Scalar(Tensor self, Scalar other) -> Tensor',
    'relu(Tensor self) -> Tensor',
    'sigmoid(Tensor self) -> Tensor',
  ].map(schema => parseFuncSchema(schema, [], []));
  const selected = await filterSubset(functions, defaultConfig.subset, undefined, allowlistPath);
  assert.deepEqual(selected.map(fn => fn.namespace), [
    'PyTorch::BasicOperators', 'PyTorch::ScalarOperators', 'PyTorch::Default',
  ]);
  assert.equal(functions[0].namespace, undefined);
  assert.equal(makeNodeClassName(selected[0]), makeNodeClassName(functions[0]));
  const limited = await filterSubset(functions, { ...defaultConfig.subset, maxOps: 1 }, /^add\./, allowlistPath);
  assert.equal(limited.length, 1);
});

test('invalid configurations and duplicate keys are rejected', async t => {
  const invalid = [
    '', 'version: 2\noperations: {}', 'version: 1\noperations: []',
    'version: 1\nunknown: true\noperations: {}',
    'version: 1\ndefaults: {enabled: false}\noperations: {}',
    'version: 1\ndefaults: {namespace: null}\noperations: {}',
    ...['{namespace: PyTorch/Activation}', '{namespace: "PyTorch::"}',
      '{namespace: Basic Operators}', '{namespace: null}', '{enabled: "false"}',
      '{displayName: 1}', '{description: 1}', '{description: " "}', '{description: null}', '{reason: false}', '{namespce: Typo}', 'null', '[]',
      '{namespace: A, namespace: B}'].map(value => `version: 1\noperations:\n  relu: ${value}`),
    'version: 1\noperations:\n  relu: {}\n  relu: {}',
  ];
  for (const entry of invalid) {
    const { allowlistPath } = await fixture(t, entry);
    await assert.rejects(filterSubset([], defaultConfig.subset, undefined, allowlistPath), /ops\.yaml:/);
  }
});

test('generation emits namespace metadata without the old 250-node cap', async t => {
  const ops = [...Array.from({ length: 251 }, (_, i) => `op${i}`), 'add', 'add_', '_add', 'ADD'];
  const { dir, allowlistPath } = await fixture(t, YAML.stringify({
    version: 1,
    defaults: { namespace: 'PyTorch::Tests' },
    operations: Object.fromEntries(ops.map(op => [op, { displayName: 'Custom "name"', description: 'Use "quoted" values.\nPath: C:\\data' }])),
  }));
  const source = path.join(dir, 'native_functions.yaml');
  await writeFile(source, ops.map(op => `- func: ${op}(Tensor self) -> Tensor`).join('\n'));
  const outDir = path.join(dir, 'generated');
  await runGenerator({ ...defaultConfig, source, outDir, allowlistPath });
  const nodes = await readdir(path.join(outDir, 'nodes'));
  assert.equal(nodes.length, ops.length);
  assert.equal(new Set(nodes.map(name => name.toLowerCase())).size, ops.length);
  const registry = JSON.parse(await readFile(path.join(outDir, 'registry.json'), 'utf8'));
  assert.equal(registry.version, 1);
  assert.equal(registry.operations.length, ops.length);
  assert.deepEqual(registry.operations.map(op => `${op.className}.hpp`).sort(), nodes.sort());
  const node = await readFile(path.join(outDir, 'nodes', nodes[0]), 'utf8');
  assert.match(node, /static ::Data::Properties create_metadata\(\)/);
  assert.match(node, /AddNamespace<"PyTorch::Tests">\(metadata\);/);
  assert.ok(node.includes(`std::string_view(${JSON.stringify('Custom "name"')})`));
  assert.ok(node.includes(`description() { return std::string_view(${JSON.stringify('Use "quoted" values.\nPath: C:\\data')})`));
});

test('omitted defaults use Uncategorized and absent operations stay excluded', async t => {
  const { allowlistPath } = await fixture(t, 'version: 1\noperations:\n  relu: {}');
  const functions = ['relu', 'sigmoid'].map(op => parseFuncSchema(`${op}(Tensor self) -> Tensor`, [], []));
  const result = await filterSubset(functions, defaultConfig.subset, undefined, allowlistPath);
  assert.equal(result.length, 1);
  assert.equal(result[0].namespace, 'PyTorch::Uncategorized');
});

test('schema description fallback escapes quoted defaults', () => {
  const schema = 'linalg_eigh(Tensor self, str UPLO="L") -> (Tensor eigenvalues, Tensor eigenvectors)';
  const node = renderNodeDefinition(parseFuncSchema(schema, [], []));
  assert.ok(node.includes(`description() { return std::string_view(${JSON.stringify(`Auto-generated from PyTorch signature: ${schema}`)})`));
});
