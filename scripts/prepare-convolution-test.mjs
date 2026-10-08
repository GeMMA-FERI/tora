import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { runGenerator } from '../dist/generator.js';
import { defaultConfig } from '../dist/config.js';

const directory = '_build/convolution-test';
await mkdir(directory, { recursive: true });
await runGenerator({ ...defaultConfig, source: '_build/native_functions.yaml', outDir: `${directory}/generated`, includeNameRegex: /^conv[12]d$/ });
const registry = JSON.parse(await readFile(`${directory}/generated/registry.json`, 'utf8'));
const headers = registry.operations.map(op => `#include "generated/nodes/${op.className}.hpp"\nusing ${op.name} = ${op.className};`).join('\n');
await writeFile(`${directory}/Processors.hpp`, headers + '\n');
