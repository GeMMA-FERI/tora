// Run capture_conv2d_module_input.py and npm run build first, from the repository root.
import { readFile, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { exportNodeGraph } from '../dist/graphExporter.js';
import { parseProgramDocument, parseRegistryDocument } from '../dist/graphDocuments.js';
import { genericIO, editorTypeForProperty } from '../dist/editor.js';

const program = parseProgramDocument(JSON.parse(await readFile('_build/conv2d-module-with-input.program.json', 'utf8')));
const registry = parseRegistryDocument(JSON.parse(await readFile('extensions/GF.Extensions.PyTorch/generated/registry.json', 'utf8')));
program.resources.push({ id: 'eager_reference', type: 'Tensor', file: 'conv2d-module-input.safetensors', key: 'expected' });
program.outputs.push({ name: 'expected', value: 'eager_reference' });
const graph = exportNodeGraph(program, registry, { resourceDirectory: path.resolve('_build') });
const outputs = genericIO('outputs', [{ key: 'Text', type: 'Str' }, { key: 'Expected', type: 'Str' }]);
const output = Object.entries(graph.Nodes).find(([, node]) => node.NodeBase === outputs.implementationId);
if (!output) throw new Error('Missing Generic Outputs node');
const [outputId, outputNode] = output;
outputNode.Metadata = outputs.metadata;
graph.NodeBases[outputs.implementationId] = outputs.nodeBase;

// Formatting is an example-level output adapter. Model capture keeps tensors.
const formatterId = 'd420b414-433b-46f4-84d7-a986a4f27e86';
const port = (kind, Description) => ({ Description, ...editorTypeForProperty(`Data::${kind}Property<"", "">`) });
graph.NodeBases[formatterId] = {
  Name: 'Tensor To Text',
  Description: 'Copies a tensor to CPU when necessary and formats its values, shape and dtype for inspection.',
  Inputs: { Tensor: port('Tensor', 'Tensor to inspect.') },
  Outputs: { Text: port('String', 'Formatted PyTorch tensor.') },
};
for (const edge of Object.values(graph.Edges).filter(edge => edge.V === outputId)) {
  const destination = edge.VT === 'expected' ? 'Expected' : 'Text';
  const id = String(Object.keys(graph.Nodes).length);
  graph.Nodes[id] = { NodeBase: formatterId, InstanceUuid: randomUUID(), Metadata: {}, Properties: {},
    Bounds: [960, destination === 'Text' ? 0 : 240, 1200, destination === 'Text' ? 180 : 420] };
  edge.V = id;
  edge.VT = 'Tensor';
  graph.Edges[String(Object.keys(graph.Edges).length)] = { U: id, V: outputId, UT: 'Text', VT: destination };
}
outputNode.Bounds = [1280, 0, 1520, 200];
for (const edge of Object.values(graph.Edges)) {
  const from = graph.NodeBases[graph.Nodes[edge.U]?.NodeBase]?.Outputs;
  const to = graph.NodeBases[graph.Nodes[edge.V]?.NodeBase]?.Inputs;
  if (!from?.[edge.UT] || !to?.[edge.VT]) throw new Error(`Invalid edge: ${JSON.stringify(edge)}`);
}
const target = '_build/conv2d-module-cli.nodegraph.json';
await writeFile(target, JSON.stringify(graph, null, 2) + '\n');
console.log(target);
