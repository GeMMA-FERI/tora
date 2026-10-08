import { makeDisplayName, makeNodeClassName, signatureUuid } from "./naming.js";
import { EDITOR_PROFILE, editorTypeForProperty } from "./editor.js";
import { outputPortKey, renderReturnAliasType } from "./templates.js";
import type { TorchFunction } from "./types.js";
import { resolveInputPorts } from "./inputPorts.js";

// This describes generated bindings, not verified runtime capabilities. Defaults
// remain schema text until the compiler converts them to typed graph constants.
export function buildNodeRegistry(functions: TorchFunction[]) {
  return {
    version: 2 as 1 | 2,
    editorProfile: EDITOR_PROFILE,
    operations: functions.map(fn => ({
      operator: `aten.${fn.baseName}.${fn.overloadName || "default"}`,
      schema: fn.rawFunc,
      schemaSignature: fn.schemaSignature,
      implementationId: signatureUuid(fn.schemaSignature),
      className: makeNodeClassName(fn),
      name: fn.displayName ?? makeDisplayName(fn),
      description: fn.description ?? `Auto-generated from PyTorch signature: ${fn.rawFunc}`,
      inputs: resolveInputPorts(fn).map(({ argument: arg, key, cppType, split }) => ({
        argument: arg.name,
        key,
        type: split ? "int" : arg.rawType,
        kind: split ? "Int" as const : arg.kind,
        optional: arg.optional,
        list: split ? false : arg.list,
        split,
        keywordOnly: arg.kwOnly,
        defaultExpression: arg.defaultValue,
        cppPropertyType: cppType,
        editorType: editorTypeForProperty(cppType),
      })),
      outputs: fn.returns.map((ret, index) => ({
        index,
        returnName: ret.name,
        key: outputPortKey(ret, index),
        type: ret.rawType,
        kind: ret.kind,
        optional: ret.optional,
        list: ret.list,
        cppPropertyType: renderReturnAliasType(ret, index),
        editorType: editorTypeForProperty(renderReturnAliasType(ret, index)),
      })),
    })),
  };
}

export type NodeRegistry = ReturnType<typeof buildNodeRegistry>;
