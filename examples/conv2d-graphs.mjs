// Run after save_conv2d_tensors.py and npm run build, from the repository root.
import { readFile, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { exportNodeGraph } from '../dist/graphExporter.js';
import { parseRegistryDocument } from '../dist/graphDocuments.js';
import { genericIO, editorTypeForProperty } from '../dist/editor.js';

const registry = parseRegistryDocument(JSON.parse(await readFile('extensions/GF.Extensions.PyTorch/generated/registry.json', 'utf8')));
const tensors = JSON.parse(await readFile(new URL('./conv2d-tensors.json', import.meta.url), 'utf8'));
const ids = {
  text: '8d9be809-5b45-4c53-91ed-b7923b005317',
  load: '3a6a2744-f014-45d9-bb93-3d96c9f6b152',
  format: 'd420b414-433b-46f4-84d7-a986a4f27e86',
};
const port = (kind, Description) => ({ Description, ...editorTypeForProperty(`Data::${kind}Property<"", "">`) });
const bases = {
  text: { Name: 'Tensor From Text', Description: 'Creates a Float64 CPU tensor from comma- or whitespace-separated values and dimensions.',
    Inputs: { Values: port('String', 'Comma- or whitespace-separated floating-point values.'), Dimensions: port('String', 'Comma- or whitespace-separated integral dimensions.') },
    Outputs: { Tensor: port('Tensor', 'Created Float64 CPU tensor.') } },
  load: { Name: 'Load Tensor', Description: 'Loads a named tensor from a Safetensors file into owned CPU memory.',
    Inputs: { File: port('String', 'Safetensors file. Relative paths require Base Directory.'), 'Tensor Name': port('String', 'Exact tensor key in the file, for example linear.weight.'), 'Base Directory': port('String', 'Optional absolute directory used to resolve a relative File.') },
    Outputs: { Tensor: port('Tensor', 'Loaded contiguous CPU tensor with its original shape and dtype.') } },
  format: { Name: 'Tensor To Text', Description: 'Copies a tensor to CPU when necessary and formats its values, shape and dtype for inspection.',
    Inputs: { Tensor: port('Tensor', 'Tensor to inspect.') }, Outputs: { Text: port('String', 'Formatted PyTorch tensor.') } },
};

for (const mode of ['text', 'mixed', 'safetensors']) {
  const graph = exportNodeGraph({ version: 1,
    inputs: Object.keys(tensors).map(id => ({ id, type: 'Tensor' })),
    operations: [{ id: 'conv', operator: 'aten.conv2d.default', args: Object.keys(tensors).map(value => ({ value })), outputs: ['result'] }],
    outputs: [{ name: 'Result', value: 'result' }],
  }, registry);
  const originalInput = graph.Nodes['0'].NodeBase;
  delete graph.Nodes['0'];
  delete graph.NodeBases[originalInput];
  let nextNode = 3;
  const add = (kind, properties, bounds) => {
    const id = String(nextNode++);
    graph.Nodes[id] = { NodeBase: ids[kind], InstanceUuid: randomUUID(), Metadata: {}, Properties: properties, Bounds: bounds };
    graph.NodeBases[ids[kind]] = bases[kind];
    return id;
  };
  Object.entries(tensors).forEach(([name, tensor], index) => {
    const text = mode === 'text' || (mode === 'mixed' && name === 'input');
    const properties = text ? { Values: tensor.values.join(', '), Dimensions: tensor.shape.join(', ') }
      : { File: 'conv2d-tensors.safetensors', 'Tensor Name': name, 'Base Directory': path.resolve('_build') };
    const node = add(text ? 'text' : 'load', properties, [0, index * 240, 260, index * 240 + 190]);
    for (const edge of Object.values(graph.Edges)) if (edge.U === '0' && edge.UT === name) { edge.U = node; edge.UT = 'Tensor'; }
  });
  graph.Nodes['1'].Bounds = [380, 160, 660, 630];
  graph.Nodes['2'].Bounds = [1120, 250, 1380, 440];
  const outputs = genericIO('outputs', [{ key: 'Result', type: 'Tensor' }, { key: 'Text', type: 'Str' }]);
  graph.Nodes['2'].Metadata = outputs.metadata;
  graph.NodeBases[outputs.implementationId] = outputs.nodeBase;
  const formatter = add('format', {}, [760, 350, 1020, 500]);
  const result = Object.values(graph.Edges).find(edge => edge.V === '2');
  graph.Edges['4'] = { U: result.U, V: formatter, UT: result.UT, VT: 'Tensor' };
  graph.Edges['5'] = { U: formatter, V: '2', UT: 'Text', VT: 'Text' };
  // Verify every reference against the snapshots before producing examples.
  for (const edge of Object.values(graph.Edges)) {
    const from = graph.NodeBases[graph.Nodes[edge.U]?.NodeBase]?.Outputs;
    const to = graph.NodeBases[graph.Nodes[edge.V]?.NodeBase]?.Inputs;
    if (!from?.[edge.UT] || !to?.[edge.VT]) throw new Error(`Invalid edge: ${JSON.stringify(edge)}`);
  }
  const target = `_build/conv2d-${mode}.nodegraph.json`;
  await writeFile(target, JSON.stringify(graph, null, 2) + '\n');
  console.log(target);
}
