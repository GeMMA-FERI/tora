import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { runGenerator } from '../dist/generator.js';
import { defaultConfig } from '../dist/config.js';
import { parseProgramDocument, parseRegistryDocument } from '../dist/graphDocuments.js';
import { exportNodeGraph } from '../dist/graphExporter.js';

const directory = '_build/captured-module-test';
await runGenerator({ ...defaultConfig, source: '_build/native_functions.yaml',
  outDir: `${directory}/generated`, includeNameRegex: /^(conv2d|add\.Tensor|relu)$/ });
const registry = parseRegistryDocument(JSON.parse(await readFile(`${directory}/generated/registry.json`, 'utf8')));
const program = parseProgramDocument(JSON.parse(await readFile(`${directory}/model.program.json`, 'utf8')));
const graph = exportNodeGraph(program, registry, { resourceDirectory: path.resolve(directory) });
await writeFile(`${directory}/model.nodegraph.json`, JSON.stringify(graph, null, 2) + '\n');
const headers = ['conv2d.default', 'add.Tensor', 'relu.default'].map((name, index) => {
  const operation = registry.operations.find(op => op.operator === `aten.${name}`);
  if (!operation) throw new Error(`Missing generated processor ${name}`);
  return `#include "generated/nodes/${operation.className}.hpp"\nusing ${['Conv2d', 'Add', 'Relu'][index]} = ${operation.className};`;
});
await writeFile(`${directory}/Processors.hpp`, headers.join('\n') + '\n');
