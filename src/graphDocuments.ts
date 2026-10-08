import { EDITOR_PROFILE, editorTypeForProperty } from "./editor.js";
import type { Program } from "./program.js";
import type { NodeRegistry } from "./registry.js";

type JsonObject = Record<string, unknown>;
function object(value: unknown, path: string): JsonObject {
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error(`${path}: expected an object`);
  return value as JsonObject;
}
function array(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${path}: expected an array`);
  return value;
}
function string(value: unknown, path: string, allowEmpty = false): string {
  if (typeof value !== "string" || (!allowEmpty && !value.length)) throw new Error(`${path}: expected a string${allowEmpty ? "" : " that is not empty"}`);
  return value;
}
function boolean(value: unknown, path: string) {
  if (typeof value !== "boolean") throw new Error(`${path}: expected a boolean`);
}
function keys(value: JsonObject, allowed: string[], path: string) {
  for (const key of Object.keys(value)) if (!allowed.includes(key)) throw new Error(`${path}: unknown field ${key}`);
}
function argument(value: unknown, path: string, allowItems = true) {
  const arg = object(value, path);
  if (Object.keys(arg).length !== 1) throw new Error(`${path}: expected exactly one of value, literal or items`);
  if (Object.hasOwn(arg, "value")) string(arg.value, `${path}.value`);
  else if (Object.hasOwn(arg, "literal")) {
    const literal = arg.literal;
    if (literal !== null && typeof literal !== "string" && typeof literal !== "boolean" &&
      !(typeof literal === "number" && Number.isFinite(literal))) throw new Error(`${path}: unsupported literal`);
  } else if (allowItems && Object.hasOwn(arg, "items")) {
    array(arg.items, `${path}.items`).forEach((item, index) => argument(item, `${path}.items[${index}]`, false));
  } else throw new Error(`${path}: expected value, literal or a flat items sequence`);
}

export function parseProgramDocument(value: unknown): Program {
  const root = object(value, "program");
  keys(root, ["version", "inputs", "operations", "outputs", "capture"], "program");
  if (root.version !== 1) throw new Error("program.version: expected 1");
  for (const [i, entry] of array(root.inputs, "program.inputs").entries()) {
    const path = `program.inputs[${i}]`;
    const input = object(entry, path);
    keys(input, ["id", "type"], path);
    string(input.id, `${path}.id`);
    if (!["Tensor", "Scalar", "Int", "Float", "Bool", "Str"].includes(string(input.type, `${path}.type`))) throw new Error(`${path}: unsupported type`);
  }
  for (const [i, entry] of array(root.operations, "program.operations").entries()) {
    const path = `program.operations[${i}]`;
    const op = object(entry, path);
    keys(op, ["id", "operator", "args", "kwargs", "outputs", "source"], path);
    string(op.id, `${path}.id`);
    string(op.operator, `${path}.operator`);
    array(op.args, `${path}.args`).forEach((arg, j) => argument(arg, `${path}.args[${j}]`));
    if (op.kwargs !== undefined) for (const [key, arg] of Object.entries(object(op.kwargs, `${path}.kwargs`))) argument(arg, `${path}.kwargs.${key}`);
    array(op.outputs, `${path}.outputs`).forEach((output, j) => string(output, `${path}.outputs[${j}]`));
    if (op.source !== undefined) string(op.source, `${path}.source`);
  }
  for (const [i, entry] of array(root.outputs, "program.outputs").entries()) {
    const path = `program.outputs[${i}]`;
    const output = object(entry, path);
    keys(output, ["name", "value"], path);
    string(output.name, `${path}.name`);
    string(output.value, `${path}.value`);
  }
  if (root.capture !== undefined) {
    const capture = object(root.capture, "program.capture");
    keys(capture, ["framework", "version", "inputs"], "program.capture");
    if (capture.framework !== "pytorch") throw new Error("program.capture: expected pytorch framework");
    string(capture.version, "program.capture.version");
    const inputs = array(capture.inputs, "program.capture.inputs");
    if (inputs.length !== (root.inputs as unknown[]).length) throw new Error("program.capture: input count mismatch");
    for (const [i, item] of inputs.entries()) {
      const path = `program.capture.inputs[${i}]`;
      const input = object(item, path);
      keys(input, ["value", "shape", "stride", "dtype", "device", "requiresGrad"], path);
      if (input.value !== (root.inputs as JsonObject[])[i].id || (root.inputs as JsonObject[])[i].type !== "Tensor") throw new Error(`${path}: input identity mismatch`);
      for (const key of ["shape", "stride"]) {
        for (const dimension of array(input[key], `${path}.${key}`)) {
          if (!Number.isSafeInteger(dimension) || (dimension as number) < 0) throw new Error(`${path}.${key}: expected nonnegative integer dimensions`);
        }
      }
      if ((input.shape as unknown[]).length !== (input.stride as unknown[]).length) throw new Error(`${path}: shape/stride rank mismatch`);
      string(input.dtype, `${path}.dtype`);
      string(input.device, `${path}.device`);
      boolean(input.requiresGrad, `${path}.requiresGrad`);
    }
  }
  return value as Program;
}

// Registries are generated artifacts. Check the fields used by binding/export,
// allowing additive metadata so future generators can attach extra information.
export function parseRegistryDocument(value: unknown): NodeRegistry {
  const root = object(value, "registry");
  if (![1, 2].includes(root.version as number) || root.editorProfile !== EDITOR_PROFILE) throw new Error(`Registry version/profile mismatch; regenerate with this Tora version (${EDITOR_PROFILE})`);
  for (const [i, entry] of array(root.operations, "registry.operations").entries()) {
    const path = `registry.operations[${i}]`;
    const op = object(entry, path);
    for (const key of ["operator", "schema", "schemaSignature", "className", "name"]) string(op[key], `${path}.${key}`);
    string(op.description, `${path}.description`, true);
    if (!/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/i.test(string(op.implementationId, `${path}.implementationId`))) throw new Error(`${path}: invalid implementation UUID`);
    for (const side of ["inputs", "outputs"]) {
      for (const [j, entry] of array(op[side], `${path}.${side}`).entries()) {
        const location = `${path}.${side}[${j}]`;
        const port = object(entry, location);
        for (const key of ["key", "type", "kind", "cppPropertyType"]) string(port[key], `${location}.${key}`);
        boolean(port.optional, `${location}.optional`);
        boolean(port.list, `${location}.list`);
        if (side === "inputs") {
          string(port.argument, `${location}.argument`);
          boolean(port.keywordOnly, `${location}.keywordOnly`);
          if (port.defaultExpression !== undefined) string(port.defaultExpression, `${location}.defaultExpression`);
          if (port.split !== undefined) {
            const split = object(port.split, `${location}.split`);
            keys(split, ["index", "length"], `${location}.split`);
            if (root.version !== 2 || !Number.isSafeInteger(split.index) || !Number.isSafeInteger(split.length)
              || (split.length as number) < 1 || (split.index as number) < 0 || (split.index as number) >= (split.length as number)
              || port.kind !== "Int" || port.list || port.optional) throw new Error(`${location}: invalid split mapping`);
          }
        } else {
          if (port.index !== j) throw new Error(`${location}: invalid return index`);
          if (port.returnName !== undefined) string(port.returnName, `${location}.returnName`);
        }
        const expected = editorTypeForProperty(port.cppPropertyType as string);
        const actual = object(port.editorType, `${location}.editorType`);
        for (const [key, value] of Object.entries(expected)) if (actual[key] !== value) throw new Error(`${location}.editorType.${key}: does not match editor profile`);
      }
    }
  }
  return value as NodeRegistry;
}
