// Programs are straight-line operations in execution order. Version 2 adds
// external tensor resources; version 1 remains valid without resources. IDs are local
// to the program; they are neither editor instance UUIDs nor terminal keys.
export type ValueType = "Tensor" | "Scalar" | "Int" | "Float" | "Bool" | "Str";

export type ScalarArgument =
  | { value: string }
  | { literal: number | boolean | string | null };
export type Argument = ScalarArgument | { items: ScalarArgument[] };

export interface Operation {
  id: string;
  operator: string;
  args: Argument[];
  kwargs?: Record<string, Argument>;
  outputs: string[];
  source?: string;
}

export interface Program {
  version: 1 | 2;
  inputs: { id: string; type: ValueType }[];
  resources?: { id: string; type: "Tensor"; file: string; key: string }[];
  operations: Operation[];
  outputs: { name: string; value: string }[];
  capture?: {
    framework: "pytorch";
    version: string;
    inputs: { value: string; shape: number[]; stride: number[]; dtype: string; device: string; requiresGrad: boolean }[];
  };
}
