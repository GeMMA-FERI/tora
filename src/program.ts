// Version 1 is a straight-line program in execution order. Value IDs are local
// to the program; they are neither editor instance UUIDs nor terminal keys.
export type ValueType = "Tensor" | "Scalar" | "Int" | "Float" | "Bool" | "Str";

export type Argument =
  | { value: string }
  | { literal: number | boolean | string | null };

export interface Operation {
  id: string;
  operator: string;
  args: Argument[];
  kwargs?: Record<string, Argument>;
  outputs: string[];
  source?: string;
}

export interface Program {
  version: 1;
  inputs: { id: string; type: ValueType }[];
  operations: Operation[];
  outputs: { name: string; value: string }[];
}
