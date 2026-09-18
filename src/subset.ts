import { readFile } from "node:fs/promises";
import YAML from "yaml";
import type { SubsetConfig, TorchFunction } from "./types.js";

function opIdentifier(fn: TorchFunction): string {
  return fn.overloadName ? `${fn.baseName}.${fn.overloadName}` : fn.baseName;
}

interface OperationOptions {
  namespace: string;
  displayName?: string;
  description?: string;
}

function mapping(value: unknown, location: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${location}: expected a mapping`);
  }
  return value as Record<string, unknown>;
}

function fields(value: Record<string, unknown>, allowed: string[], location: string): void {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) throw new Error(`${location}: unknown field '${key}'`);
  }
}

function namespaceValue(value: unknown, location: string): string {
  if (typeof value !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*(::[A-Za-z_][A-Za-z0-9_]*)*$/.test(value)) {
    throw new Error(`${location}: expected namespace identifiers separated by ::`);
  }
  return value;
}

async function loadAllowlist(allowlistPath: string): Promise<Map<string, OperationOptions>> {
  const text = await readFile(allowlistPath, "utf-8");
  const document = YAML.parseDocument(text);
  if (document.errors.length) {
    throw new Error(`${allowlistPath}: ${document.errors.map(error => error.message).join("; ")}`);
  }
  const root = mapping(document.toJS(), allowlistPath);
  fields(root, ["version", "defaults", "operations"], allowlistPath);
  if (root.version !== 1) throw new Error(`${allowlistPath}: version must be 1`);
  const defaults = root.defaults === undefined ? {} : mapping(root.defaults, `${allowlistPath}:defaults`);
  fields(defaults, ["namespace"], `${allowlistPath}:defaults`);
  const defaultNamespace = namespaceValue(
    defaults.namespace === undefined ? "PyTorch::Uncategorized" : defaults.namespace,
    `${allowlistPath}:defaults.namespace`
  );
  const operations = mapping(root.operations, `${allowlistPath}:operations`);
  const entries = new Map<string, OperationOptions>();
  for (const [op, value] of Object.entries(operations)) {
    const location = `${allowlistPath}:operations.${op}`;
    if (!/^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$/.test(op)) {
      throw new Error(`${location}: invalid operation identifier`);
    }
    const options = mapping(value, location);
    fields(options, ["namespace", "enabled", "reason", "displayName", "description"], location);
    const namespace = namespaceValue(options.namespace === undefined ? defaultNamespace : options.namespace, `${location}.namespace`);
    if (options.enabled !== undefined && typeof options.enabled !== "boolean") {
      throw new Error(`${location}.enabled: expected a boolean`);
    }
    for (const key of ["reason", "displayName", "description"]) {
      if (options[key] !== undefined && (typeof options[key] !== "string" || !(options[key] as string).trim())) {
        throw new Error(`${location}.${key}: expected a non-empty string`);
      }
    }
    if (options.enabled === false) continue;
    entries.set(op, { namespace, displayName: options.displayName as string | undefined, description: options.description as string | undefined });
  }
  return entries;
}

export async function filterSubset(
  functions: TorchFunction[],
  subset: SubsetConfig,
  includeNameRegex: RegExp | undefined,
  allowlistPath: string
): Promise<TorchFunction[]> {
  const allowlist = await loadAllowlist(allowlistPath);
  const out: TorchFunction[] = [];

  for (const fn of functions) {
    const opId = opIdentifier(fn);
    const options = allowlist.get(opId);
    if (options === undefined) continue;
    if (includeNameRegex && !includeNameRegex.test(opId)) continue;
    out.push({ ...fn, ...options });
    if (out.length >= subset.maxOps) break;
  }

  return out;
}
