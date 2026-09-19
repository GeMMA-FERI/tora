import { makeNodeClassName, signatureUuid } from "./naming.js";
import { inputPortKey, outputPortKey, renderInputAliasType, renderReturnAliasType } from "./templates.js";
import type { TorchFunction } from "./types.js";

// This describes generated bindings, not verified runtime capabilities. Defaults
// remain schema text until the compiler converts them to typed graph constants.
export function buildNodeRegistry(functions: TorchFunction[]) {
  return {
    version: 1 as const,
    operations: functions.map(fn => ({
      operator: `aten.${fn.baseName}.${fn.overloadName || "default"}`,
      schema: fn.rawFunc,
      schemaSignature: fn.schemaSignature,
      implementationId: signatureUuid(fn.schemaSignature),
      className: makeNodeClassName(fn),
      inputs: fn.args.map(arg => ({
        argument: arg.name,
        key: inputPortKey(arg),
        type: arg.rawType,
        kind: arg.kind,
        optional: arg.optional,
        list: arg.list,
        keywordOnly: arg.kwOnly,
        defaultExpression: arg.defaultValue,
        cppPropertyType: renderInputAliasType(arg),
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
      })),
    })),
  };
}

export type NodeRegistry = ReturnType<typeof buildNodeRegistry>;
