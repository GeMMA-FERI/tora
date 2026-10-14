import { mkdir, readFile, writeFile } from 'node:fs/promises';

const registry = JSON.parse(await readFile('extensions/GF.Extensions.PyTorch/generated/registry.json', 'utf8'));
const directory = '_build/convolution-features-test';
await mkdir(directory, { recursive: true });
const headers = ['conv2d', 'relu'].map(name => {
  const operation = registry.operations.find(item => item.operator === `aten.${name}.default`);
  if (!operation) throw new Error(`Missing generated processor: ${name}`);
  return `#include "generated/nodes/${operation.className}.hpp"\nusing ${name === 'conv2d' ? 'Conv2d' : 'Relu'} = ${operation.className};`;
});
await writeFile(`${directory}/Processors.hpp`, headers.join('\n') + '\n');
