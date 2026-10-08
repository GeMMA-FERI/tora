// Local Codex workflow: print compact batches, accept decisions, then update YAML.
// No network or model API calls. Run `node scripts/review-operations.mjs help`.
import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import YAML from 'yaml';
import { parseFuncSchema } from '../dist/schemaParser.js';
import { resolveInputPorts } from '../dist/inputPorts.js';

const digest = text => createHash('sha256').update(text).digest('hex');
const namespacePattern = /^PyTorch(?:::[A-Za-z_][A-Za-z0-9_]*)+$/;

export function exclusion(fn) {
  if (fn.baseName.startsWith('_')) return 'Internal ATen operator; excluded from the public node palette.';
  if (fn.baseName.includes('backward')) return 'Explicit backward kernel; excluded from the forward-processing node palette.';
  if (fn.baseName.endsWith('_')) return 'In-place mutation; prefer the functional variant in a dataflow graph.';
  if (fn.args.some(arg => arg.rawType.includes('!'))) return 'Writes into supplied tensors; output-buffer variants are excluded from the node palette.';
  if (fn.variants.length && !fn.variants.includes('function')) return 'Method-only schema; the renderer currently emits free at:: function calls.';
  if (!fn.returns.length) return 'No return value; the renderer currently requires a result.';
  if (fn.arguments) resolveInputPorts(fn);
  if (fn.returns.some(value => value.list) || fn.args.some(value => value.list && !Object.hasOwn(fn.arguments ?? {}, value.name))) return 'List-valued ports are not implemented by the current property mappings.';
  if (fn.returns.some(value => value.optional)) return 'Optional return values are not handled by the current output conversion.';
  const unsupported = new Set(['Device', 'Layout', 'MemoryFormat', 'Generator', 'Dimname', 'Stream', 'QScheme', 'Unknown']);
  const kinds = [...new Set([...fn.args, ...fn.returns].map(value => value.kind).filter(kind => unsupported.has(kind)))];
  if (kinds.length) return `Current bridge lacks typed conversion for: ${kinds.join(', ')}.`;
  return undefined;
}

export function createReview(catalogText, schemaText, mode = 'categories') {
  if (!['categories', 'labels'].includes(mode)) throw new Error('Mode must be categories or labels');
  const catalog = YAML.parse(catalogText);
  if (catalog.version !== 1 || !catalog.operations || typeof catalog.operations !== 'object') throw new Error('Invalid operation catalog');
  const schemas = new Map(YAML.parse(schemaText).filter(row => typeof row.func === 'string').map(row => {
    const variants = Array.isArray(row.variants) ? row.variants : (row.variants ?? '').split(',').map(s => s.trim()).filter(Boolean);
    const fn = parseFuncSchema(row.func, [], variants);
    return [fn.overloadName ? `${fn.baseName}.${fn.overloadName}` : fn.baseName, fn];
  }));
  const families = new Map();
  for (const [op, options] of Object.entries(catalog.operations)) {
    if (mode === 'labels' && options.enabled === false) continue;
    const fn = schemas.get(op);
    if (!fn) throw new Error(`Schema missing for ${op}; use the catalog's matching native_functions.yaml`);
    const family = mode === 'labels' ? op : fn.baseName.endsWith('_') && !fn.baseName.endsWith('__') ? fn.baseName.slice(0, -1) : fn.baseName;
    if (!families.has(family)) families.set(family, []);
    families.get(family).push({ op, schema: fn.rawFunc, namespace: options.namespace, reason: options.enabled === false ? options.reason ?? 'Previously disabled by the user.' : exclusion({ ...fn, arguments: options.arguments }) });
  }
  return {
    version: 1, mode, originalHash: digest(catalogText),
    families: [...families].sort(([a], [b]) => a.localeCompare(b)).map(([name, operations], id) => ({ id, name, operations })),
    decisions: {}, pending: [],
  };
}

export function acceptBatch(state, groups) {
  if (!state.pending.length) throw new Error('Run next before apply');
  if (!Array.isArray(groups)) throw new Error('Decisions must be an array of {namespace, ids, reason?} groups');
  const decisions = {};
  for (const group of groups) {
    if (state.mode === 'labels') {
      if (!group || Object.keys(group).some(key => !['displayName', 'description', 'ids'].includes(key)) ||
          !Array.isArray(group.ids) || !['displayName', 'description'].every(key => typeof group[key] === 'string' && group[key].trim())) {
        throw new Error('Label decisions require displayName, description, and ids');
      }
    } else if (!group || Object.keys(group).some(key => !['namespace', 'ids', 'reason'].includes(key)) ||
        !namespacePattern.test(group.namespace) || group.namespace === 'PyTorch::Uncategorized' || !Array.isArray(group.ids) ||
        (group.reason !== undefined && (typeof group.reason !== 'string' || !group.reason.trim()))) {
      throw new Error('Invalid decision group; reason, when present, disables the entire family');
    }
    for (const id of group.ids) {
      if (!Number.isInteger(id) || !state.pending.includes(id) || Object.hasOwn(decisions, id)) throw new Error(`Unexpected or duplicate family ID: ${id}`);
      decisions[id] = state.mode === 'labels'
        ? { displayName: group.displayName, description: group.description }
        : { namespace: group.namespace, ...(group.reason ? { reason: group.reason } : {}) };
    }
  }
  if (Object.keys(decisions).length !== state.pending.length) throw new Error('Every pending family must have exactly one decision');
  return { ...state, decisions: { ...state.decisions, ...decisions }, pending: [] };
}

export function renderReview(state, original) {
  if (digest(original) !== state.originalHash) throw new Error('Catalog changed since init; reinitialize before applying decisions');
  if (state.pending.length || state.families.some(family => !state.decisions[family.id])) throw new Error('Review is incomplete');
  const doc = YAML.parseDocument(original);
  let disabled = 0;
  const counts = {};
  for (const family of state.families) {
    const decision = state.decisions[family.id];
    for (const op of family.operations) {
      const base = ['operations', op.op];
      if (state.mode === 'labels') {
        doc.setIn([...base, 'displayName'], decision.displayName);
        doc.setIn([...base, 'description'], decision.description);
        continue;
      }
      const current = doc.getIn([...base, 'namespace']) ?? doc.getIn(['defaults', 'namespace']);
      const namespace = current && current !== 'PyTorch::Uncategorized' ? current : decision.namespace;
      doc.setIn([...base, 'namespace'], namespace);
      counts[namespace] = (counts[namespace] ?? 0) + 1;
      const reason = op.reason ?? decision.reason;
      if (reason) {
        doc.setIn([...base, 'enabled'], false);
        doc.setIn([...base, 'reason'], reason);
        disabled++;
      }
    }
  }
  return { text: doc.toString(), summary: state.mode === 'labels'
    ? { labeled: state.families.length }
    : { total: state.families.reduce((sum, f) => sum + f.operations.length, 0), disabled, counts } };
}

async function atomicWrite(file, text) {
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(`${file}.tmp`, text);
  await rename(`${file}.tmp`, file);
}

async function main() {
  const [command = 'help', ...args] = process.argv.slice(2);
  const option = (key, fallback) => {
    const index = args.indexOf(key);
    if (index === -1) return fallback;
    if (!args[index + 1] || args[index + 1].startsWith('--')) throw new Error(`Missing value for ${key}`);
    return args[index + 1];
  };
  const directory = option('--work', '_build/operation-review');
  const statePath = path.join(directory, 'state.json');
  if (command === 'help') {
    console.log(`Local Codex operation review (no API calls):
  init --source <native_functions.yaml> [--catalog config/operations.yaml] [--mode categories|labels]
  next [--size 100]           Print compact families; repeat returns same batch
  inspect --id <family-id>    Print exact schemas and exclusion reasons
  revisit --id <item-id>      Reopen one completed item for a corrected decision
  apply --answers <file.json> Accept [{namespace:"PyTorch::Category",ids:[0,1],reason?:"disable reason"}]
  status                     Show completed and remaining families
  finish                     Back up and update the catalog after full review
All commands accept --work <checkpoint-directory>. init intentionally resets that checkpoint.
For labels mode use a separate --work directory. next prints enabled-node schemas;
apply accepts [{displayName:"Add Tensors",description:"Add two tensors elementwise.",ids:[0]}].
finish updates only the names and descriptions of the enabled nodes reviewed.
Existing manual namespaces and disabled flags are preserved. A family decision sets the category
for all its overloads; schema-based exclusions still apply independently to each overload.`);
    return;
  }
  if (command === 'init') {
    const catalogPath = path.resolve(option('--catalog', 'config/operations.yaml'));
    const source = option('--source', '_build/native_functions.yaml');
    const state = createReview(await readFile(catalogPath, 'utf8'), await readFile(source, 'utf8'), option('--mode', 'categories'));
    state.catalogPath = catalogPath;
    await atomicWrite(statePath, JSON.stringify(state));
    console.log(`Prepared ${state.families.length} review items in ${state.mode} mode.`);
    return;
  }
  let state = JSON.parse(await readFile(statePath, 'utf8'));
  if (command === 'next') {
    const size = Number(option('--size', '100'));
    if (!Number.isInteger(size) || size < 1 || size > 500) throw new Error('Batch size must be 1..500');
    if (!state.pending.length) state.pending = state.families.filter(f => !state.decisions[f.id]).slice(0, size).map(f => f.id);
    await atomicWrite(statePath, JSON.stringify(state));
    console.log(state.mode === 'labels' ? 'id schema' : 'id family (eligible/total); overloads are grouped to reduce tokens');
    for (const id of state.pending) {
      const f = state.families[id];
      console.log(state.mode === 'labels' ? `${id} ${f.operations[0].schema}` : `${id} ${f.name} (${f.operations.filter(o => !o.reason).length}/${f.operations.length})`);
    }
  } else if (command === 'revisit') {
    const id = Number(option('--id', '-1'));
    if (!Number.isInteger(id) || !state.families[id] || !state.decisions[id]) throw new Error('Select a completed item to revisit');
    if (state.pending.length) throw new Error('Finish the pending batch before revisiting an item');
    state.pending = [id];
    await atomicWrite(statePath, JSON.stringify(state));
    console.log(`Reopened ${id}: ${state.families[id].name}`);
  } else if (command === 'inspect') {
    const family = state.families[Number(option('--id', '-1'))];
    if (!family) throw new Error('Invalid family ID');
    console.log(JSON.stringify(family, null, 2));
  } else if (command === 'apply') {
    state = acceptBatch(state, JSON.parse(await readFile(option('--answers', ''), 'utf8')));
    await atomicWrite(statePath, JSON.stringify(state));
    console.log(`Reviewed ${Object.keys(state.decisions).length}/${state.families.length} families.`);
  } else if (command === 'status') {
    console.log(`Reviewed ${Object.keys(state.decisions).length}/${state.families.length} families; pending batch: ${state.pending.length}.`);
  } else if (command === 'finish') {
    const original = await readFile(state.catalogPath, 'utf8');
    const result = renderReview(state, original);
    await atomicWrite(path.join(directory, 'operations.before.yaml'), original);
    await atomicWrite(path.join(directory, 'summary.json'), JSON.stringify(result.summary, null, 2));
    await atomicWrite(state.catalogPath, result.text);
    console.log(JSON.stringify(result.summary, null, 2));
  } else throw new Error(`Unknown command: ${command}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
