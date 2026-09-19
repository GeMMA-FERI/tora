import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { editorTypeForProperty, genericIO, EDITOR_PROFILE } from '../dist/editor.js';
import { buildNodeRegistry } from '../dist/registry.js';
import { parseFuncSchema } from '../dist/schemaParser.js';

const fixture = JSON.parse(await readFile(new URL('./fixtures/editor-types.json', import.meta.url), 'utf8'));

test('property types match snapshots exported by the editor', () => {
  for (const [name, { Description, ...type }] of Object.entries(fixture.properties)) {
    assert.deepEqual(editorTypeForProperty(`Data::${name}<"Example", "">`), type);
  }
  assert.throws(() => editorTypeForProperty('Data::UnknownProperty<"", "">'), /No editor type/);
});

test('generic IO definitions and metadata match exported graphs', () => {
  const inputs = genericIO('inputs', [{ key: 'A', type: 'Int' }, { key: 'B', type: 'Int' }]);
  assert.equal(inputs.implementationId, '86e1a1e0-2ad2-4096-a707-0ed6302b9bac');
  assert.deepEqual(inputs.metadata, fixture.genericInputs);
  assert.deepEqual(inputs.nodeBase, fixture.genericInputBase);
  const outputs = genericIO('outputs', [{ key: 'Result', type: 'Str' }]);
  assert.equal(outputs.implementationId, 'e036c9af-ae18-4b4c-bd6d-c4b77691ba33');
  assert.deepEqual(outputs.metadata, fixture.genericOutputs);
  assert.deepEqual(outputs.nodeBase, fixture.genericOutputBase);
});

test('tensor IO uses the extended metadata accepted by NodeFactory', () => {
  const io = genericIO('inputs', [{ key: 'x', type: 'Tensor' }]);
  const meta = io.metadata.Properties.x;
  assert.equal(meta.ValueType, 17);
  assert.equal(meta.ClassName, fixture.properties.TensorProperty.ClassType);
  assert.equal(meta.InterfaceType, fixture.properties.TensorProperty.InterfaceType);
  assert.equal(meta.ClassType, undefined);
  assert.equal(io.nodeBase.Outputs.x.ClassType, meta.ClassName);
  assert.deepEqual(io.nodeBase.Inputs, {});
});

test('IO definitions reject collisions and keep arbitrary keys as own properties', () => {
  assert.throws(() => genericIO('outputs', [{ key: 'x', type: 'Int' }, { key: 'x', type: 'Str' }]), /duplicate IO/);
  const io = genericIO('inputs', [{ key: '__proto__', type: 'Bool' }]);
  assert.equal(Object.hasOwn(io.metadata.Properties, '__proto__'), true);
  assert.equal(io.metadata.Properties.__proto__, 'Bool');
});

test('registry records editor profile and types without changing property identities', () => {
  const registry = buildNodeRegistry([parseFuncSchema('add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor', [], [])]);
  assert.equal(registry.editorProfile, EDITOR_PROFILE);
  const operation = registry.operations[0];
  assert.equal(operation.name, 'Add Tensor');
  assert.equal(operation.description, `Auto-generated from PyTorch signature: ${operation.schema}`);
  assert.equal(operation.inputs[0].editorType.InterfaceType, 'struct Data::Interfaces::ITensor');
  assert.equal(operation.inputs[2].key, 'Alpha (kw_only)');
  assert.equal(operation.inputs[2].editorType.ValueType, 13);
});
