// Run save_conv2d_features.py and npm run build first, from the repository root.
import { readFile, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { exportNodeGraph } from '../dist/graphExporter.js';
import { parseRegistryDocument, parseProgramDocument } from '../dist/graphDocuments.js';
import { editorTypeForProperty } from '../dist/editor.js';

const config = JSON.parse(await readFile(new URL('./conv2d-features.json', import.meta.url), 'utf8'));
const registry = parseRegistryDocument(JSON.parse(await readFile('extensions/GF.Extensions.PyTorch/generated/registry.json', 'utf8')));
const names = ['input', ...config.stages.flatMap(stage => [`${stage.name}.weight`, `${stage.name}.bias`])];
const operations = [];
const outputs = [{ name: 'Input patterns', value: 'input' }];
let input = 'input';
for (const stage of config.stages) {
  const conv = `${stage.name}.conv`;
  const relu = `${stage.name}.relu`;
  operations.push({ id: conv, operator: 'aten.conv2d.default',
    args: [input, `${stage.name}.weight`, `${stage.name}.bias`].map(value => ({ value })),
    kwargs: Object.fromEntries(['stride', 'padding', 'dilation', 'groups'].map(key => [key,
      Array.isArray(stage[key]) ? { items: stage[key].map(literal => ({ literal })) } : { literal: stage[key] }])),
    outputs: [conv] });
  operations.push({ id: relu, operator: 'aten.relu.default', args: [{ value: conv }], outputs: [relu] });
  outputs.push({ name: `${stage.name} features`, value: relu });
  input = relu;
}
const program = parseProgramDocument({ version: 1, inputs: names.map(id => ({ id, type: 'Tensor' })), operations, outputs });
const graph = exportNodeGraph(program, registry);
const inputBase = graph.Nodes['0'].NodeBase;
delete graph.Nodes['0'];
delete graph.NodeBases[inputBase];
// The loader is a handwritten extension processor, not an ATen registry operation.
const loaderId = '3a6a2744-f014-45d9-bb93-3d96c9f6b152';
const port = (kind, Description) => ({ Description, ...editorTypeForProperty(`Data::${kind}Property<"", "">`) });
graph.NodeBases[loaderId] = {
  Name: 'Load Tensor', Description: 'Loads a named tensor from a Safetensors file into owned CPU memory.',
  Inputs: { File: port('String', 'Safetensors file.'), 'Tensor Name': port('String', 'Exact tensor key.'),
    'Base Directory': port('String', 'Absolute directory resolving the tensor file.') },
  Outputs: { Tensor: port('Tensor', 'Loaded CPU tensor with its original shape and dtype.') },
};
let nextNode = Math.max(...Object.keys(graph.Nodes).map(Number)) + 1;
for (const [index, name] of names.entries()) {
  const id = String(nextNode++);
  const stage = index === 0 ? 0 : Math.floor((index - 1) / 2);
  const x = index === 0 ? 0 : 360 + stage * 780;
  const y = index === 0 ? 80 : 600 + ((index - 1) % 2) * 230;
  graph.Nodes[id] = { NodeBase: loaderId, InstanceUuid: randomUUID(), Metadata: {},
    Properties: { File: 'conv2d-features.safetensors', 'Tensor Name': name, 'Base Directory': path.resolve('_build') },
    Bounds: [x, y, x + 290, y + 200] };
  for (const edge of Object.values(graph.Edges)) {
    if (edge.U === '0' && edge.UT === name) { edge.U = id; edge.UT = 'Tensor'; }
  }
}
for (const [index] of config.stages.entries()) {
  const x = 360 + index * 780;
  graph.Nodes[String(index * 2 + 1)].Bounds = [x, 40, x + 300, 530];
  graph.Nodes[String(index * 2 + 2)].Bounds = [x + 390, 160, x + 650, 330];
}
const outputId = String(operations.length + 1);
graph.Nodes[outputId].Bounds = [360 + config.stages.length * 780, 100, 720 + config.stages.length * 780, 420];
for (const edge of Object.values(graph.Edges)) {
  const from = graph.NodeBases[graph.Nodes[edge.U]?.NodeBase]?.Outputs;
  const to = graph.NodeBases[graph.Nodes[edge.V]?.NodeBase]?.Inputs;
  if (!from?.[edge.UT] || !to?.[edge.VT]) throw new Error(`Invalid edge: ${JSON.stringify(edge)}`);
}
await writeFile('_build/conv2d-features.program.json', JSON.stringify(program, null, 2) + '\n');
await writeFile('_build/conv2d-features.nodegraph.json', JSON.stringify(graph, null, 2) + '\n');
console.log(`_build/conv2d-features.nodegraph.json (${Object.keys(graph.Nodes).length} nodes, ${Object.keys(graph.Edges).length} edges)`);
