import type { Argument, Operation, Program, ValueType } from "./program.js";
import type { NodeRegistry } from "./registry.js";

type Binding = NodeRegistry["operations"][number];
type InputPort = Binding["inputs"][number];
type Port = InputPort | Binding["outputs"][number];
export type ValueSource =
  | { input: string; type: ValueType }
  | { operation: string; key: string; type: ValueType };
export type BoundArgument = ValueSource | { literal: number | boolean | string | null };
export interface BoundOperation {
  id: string;
  operator: string;
  implementationId: string;
  inputs: { key: string; argument: string; source: BoundArgument }[];
  outputs: { value: string; key: string; type: ValueType }[];
  source?: string;
}
export interface BoundProgram {
  version: 1;
  inputs: Program["inputs"];
  operations: BoundOperation[];
  outputs: { name: string; source: ValueSource }[];
}

const supportedTypes = new Set<string>(["Tensor", "Scalar", "Int", "Float", "Bool", "Str"]);

function fail(operation: Operation, message: string): never {
  throw new Error(`${operation.id} (${operation.operator})${operation.source ? ` at ${operation.source}` : ""}: ${message}`);
}

function portType(port: Port, operation: Operation): ValueType {
  if (port.list || !supportedTypes.has(port.kind) || port.type.includes("!")) {
    fail(operation, `unsupported port type ${port.type} on ${port.key}`);
  }
  return port.kind as ValueType;
}

function defaultArgument(port: InputPort, operation: Operation): Argument {
  const expression = port.defaultExpression;
  if (expression === undefined) fail(operation, `missing argument ${port.argument}`);
  if (expression === "None") return { literal: null };
  if (expression === "True" || expression === "False") return { literal: expression === "True" };
  if (/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(expression)) {
    return { literal: Number(expression) };
  }
  if (expression.startsWith('"')) {
    try {
      const value: unknown = JSON.parse(expression);
      if (typeof value === "string") return { literal: value };
    } catch { /* Report the unsupported expression below. */ }
  }
  fail(operation, `unsupported default ${expression} for ${port.argument}`);
}

function bindArgument(argument: Argument, port: InputPort, operation: Operation,
  values: Map<string, ValueSource>): BoundArgument {
  const expected = portType(port, operation);
  if ("value" in argument) {
    const source = values.get(argument.value);
    if (!source) fail(operation, `unknown or forward value ${argument.value} for ${port.argument}`);
    // Conversion nodes and the editor's native conversion rules are a later
    // lowering pass. Do not silently assume a conversion is available.
    if (source.type !== expected) fail(operation, `expected ${expected} for ${port.argument}, got ${source.type}; conversion required`);
    return { ...source };
  }
  const value = argument.literal;
  if (value === null) {
    if (!port.optional) fail(operation, `None is not allowed for ${port.argument}`);
  } else {
    const numeric = typeof value === "number" && Number.isFinite(value);
    const valid = expected === "Scalar" || expected === "Float" ? numeric
      : expected === "Int" ? numeric && Number.isSafeInteger(value)
      : expected === "Bool" ? typeof value === "boolean"
      : expected === "Str" ? typeof value === "string" : false;
    if (!valid || (numeric && Number.isInteger(value) && !Number.isSafeInteger(value))) {
      fail(operation, `invalid literal for ${port.argument} (${expected})`);
    }
  }
  return { literal: value };
}

export function bindProgram(program: Program, registry: NodeRegistry): BoundProgram {
  if (program.version !== 1 || registry.version !== 1) throw new Error("Unsupported program or registry version");
  const bindings = new Map<string, Binding>();
  for (const binding of registry.operations) {
    if (bindings.has(binding.operator)) throw new Error(`Duplicate registry operator ${binding.operator}`);
    bindings.set(binding.operator, binding);
  }
  const values = new Map<string, ValueSource>();
  function addValue(id: string, source: ValueSource) {
    if (!id || values.has(id)) throw new Error(`Empty or duplicate value ID: ${id}`);
    values.set(id, source);
  }
  for (const input of program.inputs) {
    if (!supportedTypes.has(input.type)) throw new Error(`Unsupported input type ${input.type} on ${input.id}`);
    addValue(input.id, { input: input.id, type: input.type });
  }
  const ids = new Set<string>();
  const operations: BoundOperation[] = [];
  for (const operation of program.operations) {
    if (!operation.id || ids.has(operation.id)) fail(operation, "empty or duplicate operation ID");
    ids.add(operation.id);
    const binding = bindings.get(operation.operator);
    if (!binding) fail(operation, "operator is not present in the generated registry");
    for (const ports of [binding.inputs, binding.outputs]) {
      if (new Set(ports.map(port => port.key)).size !== ports.length) fail(operation, "ambiguous terminal keys in registry");
    }
    if (operation.outputs.length !== binding.outputs.length) fail(operation, `expected ${binding.outputs.length} output values`);
    const positional = binding.inputs.filter(port => !port.keywordOnly);
    if (operation.args.length > positional.length) fail(operation, "too many positional arguments");
    const argumentsByName = new Map<string, Argument>();
    operation.args.forEach((arg, index) => argumentsByName.set(positional[index].argument, arg));
    for (const [name, argument] of Object.entries(operation.kwargs ?? {})) {
      if (!binding.inputs.some(port => port.argument === name)) fail(operation, `unknown keyword argument ${name}`);
      if (argumentsByName.has(name)) fail(operation, `argument ${name} supplied twice`);
      argumentsByName.set(name, argument);
    }
    const inputs = binding.inputs.map(port => ({
      key: port.key,
      argument: port.argument,
      source: bindArgument(argumentsByName.get(port.argument) ?? defaultArgument(port, operation), port, operation, values),
    }));
    const outputs = binding.outputs.map((port, index) => {
      const type = portType(port, operation);
      if (port.optional) fail(operation, `optional output ${port.key} is not supported yet`);
      const value = operation.outputs[index];
      addValue(value, { operation: operation.id, key: port.key, type });
      return { value, key: port.key, type };
    });
    operations.push({ id: operation.id, operator: operation.operator,
      implementationId: binding.implementationId, inputs, outputs, source: operation.source });
  }
  const names = new Set<string>();
  const outputs = program.outputs.map(output => {
    if (!output.name || names.has(output.name)) throw new Error(`Empty or duplicate output name: ${output.name}`);
    names.add(output.name);
    const source = values.get(output.value);
    if (!source) throw new Error(`Unknown output value ${output.value}`);
    return { name: output.name, source: { ...source } };
  });
  return { version: 1, inputs: program.inputs.map(input => ({ ...input })), operations, outputs };
}
