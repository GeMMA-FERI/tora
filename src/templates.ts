import type { ArgKind, TorchArgument, TorchReturn } from "./types.js";
import { makePortLabel } from "./portLabels.js";

const INPUT_TYPE_TEMPLATES: Record<ArgKind, string> = {
  Tensor: 'Data::TensorProperty<"{label}", "">',
  Scalar: 'Data::NumericProperty<"{label}", "">',
  ScalarType: 'Data::EnumProperty<"{label}", "">',
  SymInt: 'Data::IntegralProperty<"{label}", "">',
  Int: 'Data::IntegralProperty<"{label}", "">',
  Float: 'Data::NumericProperty<"{label}", "">',
  Bool: 'Data::BooleanProperty<"{label}", "">',
  Str: 'Data::StringProperty<"{label}", "">',
  Layout: 'Data::EnumProperty<"{label}", "">',
  Device: 'Data::EnumProperty<"{label}", "">',
  MemoryFormat: 'Data::EnumProperty<"{label}", "">',
  Generator: 'Data::TensorProperty<"{label}", "">', // TODO
  Dimname: 'Data::StringProperty<"{label}", "">',
  Stream: 'Data::StringProperty<"{label}", "">',    // TODO
  QScheme: 'Data::EnumProperty<"{label}", "">',
  Unknown: 'Data::StringProperty<"{label}", "">'    // TODO
};

function withLabel(template: string, label: string): string {
  return template.replaceAll("{label}", label);
}

export function renderInputAliasType(arg: TorchArgument): string {
  const base = INPUT_TYPE_TEMPLATES[arg.kind];
  const labels: string[] = [];
  if (arg.optional) labels.push("optional");
  if (arg.list) labels.push("list");
  if (arg.kwOnly) labels.push("kw_only");
  const suffix = labels.length ? ` (${labels.join(",")})` : "";
  return withLabel(base, `${makePortLabel(arg.name, "input")}${suffix}`);
}

export function renderReturnAliasType(ret: TorchReturn, index: number): string {
  const base = INPUT_TYPE_TEMPLATES[ret.kind];
  const name = ret.name ?? `output_${index}`;
  const labels: string[] = [];
  if (ret.optional) labels.push("optional");
  if (ret.list) labels.push("list");
  const suffix = labels.length ? ` (${labels.join(",")})` : "";
  return withLabel(base, `${makePortLabel(name, "output")}${suffix}`);
}
