export type ArgKind =
  | "Tensor"
  | "Scalar"
  | "ScalarType"
  | "SymInt"
  | "Int"
  | "Float"
  | "Bool"
  | "Str"
  | "Layout"
  | "Device"
  | "MemoryFormat"
  | "Generator"
  | "Dimname"
  | "Stream"
  | "QScheme"
  | "Unknown";

export interface TorchArgument {
  name: string;
  rawType: string;
  kind: ArgKind;
  optional: boolean;
  list: boolean;
  kwOnly: boolean;
  defaultValue?: string;
}

export interface TorchReturn {
  name?: string;
  rawType: string;
  kind: ArgKind;
  optional: boolean;
  list: boolean;
}

export interface TorchFunction {
  namespace?: string;
  displayName?: string;
  description?: string;
  rawFunc: string;
  baseName: string;
  overloadName?: string;
  schemaSignature: string;
  args: TorchArgument[];
  returns: TorchReturn[];
  tags: string[];
  variants: string[];
}

export interface SubsetConfig {
  includeInternal: boolean;
  includeInplace: boolean;
  includeOut: boolean;
  includeMethods: boolean;
  requireTensorInput: boolean;
  requireTensorOutput: boolean;
  maxOps: number;
}

export interface GeneratorConfig {
  source: string;
  outDir: string;
  allowlistPath: string;
  subset: SubsetConfig;
  includeNameRegex?: RegExp;
}
