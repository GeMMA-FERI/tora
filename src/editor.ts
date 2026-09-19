import type { ValueType } from "./program.js";

// Windows/MSVC profile verified against FrameworkData/Include/Data/Interfaces.h,
// Property.hpp and the editor's NodeFactory.cpp plus exported graph samples.
// These are serialized SDK values, not PyTorch dtype codes.
export const EDITOR_PROFILE = "gemmafusionsuite-msvc-v1";
export interface EditorType {
  ValueType: number;
  ClassType: string;
  InterfaceType: string;
}
export interface EditorPort extends EditorType { Description: string }
export interface EditorNodeBase {
  Name: string;
  Description: string;
  Inputs: Record<string, EditorPort>;
  Outputs: Record<string, EditorPort>;
}

const PROPERTY_TYPES: Record<string, EditorType> = {
  TensorProperty: { ValueType: 17, ClassType: "struct Data::Tensor", InterfaceType: "struct Data::Interfaces::ITensor" },
  NumericProperty: { ValueType: 13, ClassType: "double", InterfaceType: "double" },
  IntegralProperty: { ValueType: 5, ClassType: "__int64", InterfaceType: "__int64" },
  BooleanProperty: { ValueType: 15, ClassType: "struct Core::Bool", InterfaceType: "struct Core::Bool" },
  StringProperty: { ValueType: 19, ClassType: "struct Data::String", InterfaceType: "struct Data::Interfaces::IString" },
  EnumProperty: { ValueType: 17, ClassType: "struct Data::Enum", InterfaceType: "struct Data::Interfaces::IEnum" },
};

export function editorTypeForProperty(cppPropertyType: string): EditorType {
  const name = /^Data::([A-Za-z]+Property)</.exec(cppPropertyType)?.[1];
  const type = name && Object.hasOwn(PROPERTY_TYPES, name) ? PROPERTY_TYPES[name] : undefined;
  if (!type) throw new Error(`No editor type mapping for ${cppPropertyType}`);
  return { ...type };
}

export const GENERIC_INPUTS_ID = "86e1a1e0-2ad2-4096-a707-0ed6302b9bac";
export const GENERIC_OUTPUTS_ID = "e036c9af-ae18-4b4c-bd6d-c4b77691ba33";
type DynamicPortMetadata = string | {
  Description: string; ValueType: number; ClassName: string; InterfaceType: string;
};

// Match NodeFactory's short primitive form and extended object form exactly.
// In particular, dynamic object metadata uses ClassName, while snapshots use
// ClassType. Primitive snapshots use the enum name supplied to RegisterProperty.
export function genericIO(direction: "inputs" | "outputs", ports: { key: string; type: ValueType }[]) {
  const definitions: [string, EditorPort][] = [];
  const metadata: [string, DynamicPortMetadata][] = [];
  const seen = new Set<string>();
  for (const port of ports) {
    if (!port.key || seen.has(port.key)) throw new Error(`Empty or duplicate IO key: ${port.key}`);
    seen.add(port.key);
    if (port.type === "Tensor") {
      const type = editorTypeForProperty('Data::TensorProperty<"", "">');
      const Description = "Generic IO of type Tensor";
      definitions.push([port.key, { Description, ...type }]);
      metadata.push([port.key, { Description, ValueType: type.ValueType, ClassName: type.ClassType, InterfaceType: type.InterfaceType }]);
    } else {
      const primitive = { Scalar: ["Double", 13], Float: ["Double", 13], Int: ["Int64", 5], Bool: ["Bool", 15], Str: ["String", 19] } as const;
      if (!Object.hasOwn(primitive, port.type)) throw new Error(`Unsupported IO type ${port.type}`);
      const [name, valueType] = primitive[port.type];
      definitions.push([port.key, { Description: `Generic IO of type ${name}`, ValueType: valueType, ClassType: name, InterfaceType: name }]);
      metadata.push([port.key, name]);
    }
  }
  const propertyMap = Object.fromEntries(definitions);
  const nodeBase: EditorNodeBase = {
    Name: direction === "inputs" ? "Generic Inputs" : "Generic Outputs",
    Description: "",
    Inputs: direction === "outputs" ? propertyMap : {},
    Outputs: direction === "inputs" ? propertyMap : {},
  };
  return {
    implementationId: direction === "inputs" ? GENERIC_INPUTS_ID : GENERIC_OUTPUTS_ID,
    metadata: { Properties: Object.fromEntries(metadata) },
    nodeBase,
  };
}
