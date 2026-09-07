type PortDirection = "input" | "output";

const LABEL_MAP: Record<string, string> = {
  alpha: "Alpha",
  beta: "Beta",
  bias: "Bias",
  ceil_mode: "Ceil Mode",
  count_include_pad: "Count Include Padding",
  descending: "Descending",
  dilation: "Dilation",
  dim: "Dimension",
  dims: "Dimensions",
  divisor_override: "Divisor Override",
  dtype: "Data Type",
  groups: "Groups",
  ignore_index: "Ignore Index",
  indices: "Indices",
  input: "Input",
  keepdim: "Keep Dimensions",
  k: "K",
  kernel_size: "Kernel Size",
  largest: "Largest",
  mat1: "Matrix 1",
  mat2: "Matrix 2",
  mode: "Mode",
  other: "Other",
  output: "Output",
  output_0: "Output 0",
  output_1: "Output 1",
  output_2: "Output 2",
  output_3: "Output 3",
  output_4: "Output 4",
  output_padding: "Output Padding",
  output_size: "Output Size",
  padding: "Padding",
  padding_idx: "Padding Index",
  per_sample_weights: "Per-Sample Weights",
  p: "Probability",
  pos_weight: "Positive Weight",
  reduction: "Reduction",
  scale_grad_by_freq: "Scale Gradient By Frequency",
  self: "Input",
  shape: "Shape",
  sorted: "Sorted",
  sparse: "Sparse",
  stride: "Stride",
  target: "Target",
  total_weight: "Total Weight",
  train: "Training",
  weight: "Weight",
  grad_input: "Gradient Input",
  values: "Values",
  running_mean: "Running Mean",
  running_var: "Running Variance",
  training: "Training",
  momentum: "Momentum",
  eps: "Epsilon",
  cudnn_enabled: "Enable CuDNN",
  cudnn_enable: "Enable CuDNN",
  normalized_shape: "Normalized Shape",
  approximate: "Approximation Method",
  tensors: "Tensors",
  offsets: "Offsets",
  include_last_offset: "Include Last Offset",
  start_dim: "Start Dimension",
  end_dim: "End Dimension",
  num_groups: "Number of Groups",
  label_smoothing: "Label Smoothing",
  grad_output: "Gradient Output",
};

const ACRONYMS = new Set(["fft", "nll"]);

function titleCaseToken(token: string): string {
  if (!token) return token;
  const lower = token.toLowerCase();
  if (ACRONYMS.has(lower)) return lower.toUpperCase();
  return `${lower[0].toUpperCase()}${lower.slice(1)}`;
}

function toTitleCase(name: string): string {
  return name
    .split(/[_\s]+/g)
    .filter(Boolean)
    .map(titleCaseToken)
    .join(" ");
}

export function makePortLabel(name: string, direction: PortDirection): string {
  if (LABEL_MAP[name]) {
    return LABEL_MAP[name];
  } else {
    console.log(`Warning: No label found for port name "${name}" in direction "${direction}". Using title case as fallback.`);
  }
  return toTitleCase(name);
}
