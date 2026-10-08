import { inputPortKey, renderInputAliasType } from "./templates.js";
import type { TorchArgument, TorchFunction } from "./types.js";

export interface InputPort {
  argument: TorchArgument;
  alias: string;
  key: string;
  cppType: string;
  split?: { index: number; length: number };
}

export function safeAliasName(input: string): string {
  return input.replace(/[^a-zA-Z0-9]+/g, "_").replace(/^([0-9])/, "_$1");
}

// Shared by the C++ renderer and registry: the YAML mapping is the only source
// of split-port order and terminal keys.
export function resolveInputPorts(fn: TorchFunction): InputPort[] {
  for (const name of Object.keys(fn.arguments ?? {})) {
    if (!fn.args.some(arg => arg.name === name)) throw new Error(`${fn.baseName}.arguments.${name}: unknown argument`);
  }
  const ports = fn.args.flatMap((arg): InputPort[] => {
    const mapping = fn.arguments && Object.hasOwn(fn.arguments, arg.name) ? fn.arguments[arg.name] : undefined;
    const alias = safeAliasName(arg.name);
    if (!mapping) return [{ argument: arg, alias, key: inputPortKey(arg), cppType: renderInputAliasType(arg) }];
    if (!arg.list || !arg.listLength || arg.optional || !["Int", "SymInt"].includes(arg.kind)
      || mapping.ports.length !== arg.listLength) {
      throw new Error(`${fn.baseName}.arguments.${arg.name}: ports must match a non-optional fixed-length integer list`);
    }
    return mapping.ports.map((port, index) => ({
      argument: arg, alias: `${alias}_${index}`, key: port.name,
      cppType: `Data::IntegralProperty<${JSON.stringify(port.name)}, "">`,
      split: { index, length: arg.listLength! },
    }));
  });
  for (const field of ["key", "alias"] as const) {
    if (new Set(ports.map(port => port[field])).size !== ports.length) throw new Error(`${fn.baseName}: duplicate input ${field}`);
  }
  return ports;
}

export function integerDefault(arg: TorchArgument, index?: number): number | undefined {
  if (arg.defaultValue === undefined) return undefined;
  let value: unknown;
  try { value = JSON.parse(arg.defaultValue); } catch { return undefined; }
  if (index !== undefined && Array.isArray(value)) {
    if (value.length !== arg.listLength) throw new Error(`${arg.name}: default list length does not match schema`);
    value = value[index];
  }
  return typeof value === "number" && Number.isSafeInteger(value) ? value : undefined;
}
