"""Capture a static PyTorch module and its tensor resources into Tora's program format."""
import argparse
import importlib.util
import json
import math
import operator
import os
from pathlib import Path
import sys
import tempfile


class CaptureError(ValueError):
    pass


def capture_program(module, args, kwargs=None, *, resource_file=None):
    """Write model tensors to resource_file; place the program beside that file."""
    import torch
    from torch.export.graph_signature import InputKind, OutputKind, TensorArgument

    kwargs = kwargs or {}
    if not isinstance(module, torch.nn.Module) or not isinstance(args, tuple):
        raise CaptureError("Expected a torch.nn.Module and a tuple of example inputs")
    if not all(isinstance(value, torch.Tensor) for value in (*args, *kwargs.values())):
        raise CaptureError("Only flat tensor inputs are supported; primitive inputs can be specialized during export")
    exported = torch.export.export(module, args, kwargs, strict=True)
    if any(spec.kind == InputKind.CONSTANT_TENSOR for spec in exported.graph_signature.input_specs):
        # Tensor construction can emit detach_ on a fresh lifted copy. Use
        # PyTorch's functionalization without requesting ATen decompositions.
        # Observable input/buffer mutations remain explicit output specs below.
        exported = exported.run_decompositions({})
    if exported.range_constraints:
        raise CaptureError("Symbolic shape constraints are not supported")
    resource_specs = {}
    for spec in exported.graph_signature.input_specs:
        if not isinstance(spec.arg, TensorArgument) or spec.kind not in (
                InputKind.USER_INPUT, InputKind.PARAMETER, InputKind.BUFFER, InputKind.CONSTANT_TENSOR):
            raise CaptureError(f"Unsupported input {spec.arg.name}: {spec.kind.name}")
        if spec.kind != InputKind.USER_INPUT:
            if resource_file is None:
                raise CaptureError(f"Input {spec.arg.name} ({spec.kind.name}) requires resource_file for tensor storage")
            resource_specs[spec.arg.name] = spec
    for spec in exported.graph_signature.output_specs:
        if spec.kind != OutputKind.USER_OUTPUT:
            raise CaptureError(f"Unsupported output {spec.kind.name}; mutation is not supported")

    program = {"version": 1, "inputs": [], "operations": [], "outputs": []}
    constraints = []
    values = {}
    tensors = {}

    def fail(node, message):
        source = node.meta.get("stack_trace", "").strip()
        raise CaptureError(f"{node.name} ({node.target}): {message}" + (f"\n{source}" if source else ""))

    def argument(value, node):
        if isinstance(value, torch.fx.Node):
            result = values.get(value)
            if not isinstance(result, str):
                fail(node, "Expected one tensor value; select a tuple result explicitly")
            return {"value": result}
        if isinstance(value, (tuple, list)):
            if not all(type(item) is int for item in value):
                fail(node, "Only concrete integer sequences are supported")
            return {"items": [argument(item, node) for item in value]}
        if value is None or type(value) in (str, bool):
            return {"literal": value}
        if type(value) in (int, float):
            if not math.isfinite(value) or (float(value).is_integer() and abs(value) > 2**53 - 1):
                fail(node, "Numeric literal cannot be represented safely in JSON")
            return {"literal": value}
        fail(node, f"Unsupported argument type {type(value).__name__}; dtype/device objects and symbolic values need lowering")

    for node in exported.graph.nodes:
        if node.op == "placeholder":
            if node.name in resource_specs:
                spec = resource_specs[node.name]
                key = spec.target
                source = exported.state_dict if key in exported.state_dict else exported.constants
                tensor = source.get(key)
                if not isinstance(key, str) or not key or "\0" in key or not isinstance(tensor, torch.Tensor):
                    fail(node, "Cannot resolve the captured tensor resource")
                supported = {"torch.bool", "torch.uint8", "torch.int8", "torch.int16", "torch.int32", "torch.int64",
                             "torch.uint16", "torch.uint32", "torch.uint64", "torch.float16", "torch.bfloat16",
                             "torch.float32", "torch.float64"}
                if tensor.layout != torch.strided or tensor.is_quantized or tensor.device.type != "cpu" or str(tensor.dtype) not in supported:
                    fail(node, f"Unsupported tensor resource layout, device or dtype: {tensor.layout}, {tensor.device}, {tensor.dtype}")
                if key not in tensors:
                    # Independent CPU snapshots also allow tied weights and views
                    # to be saved without Safetensors storage-alias restrictions.
                    tensors[key] = tensor.detach().resolve_conj().resolve_neg().contiguous().clone()
                program.setdefault("resources", []).append({"id": node.name, "type": "Tensor",
                                                           "file": Path(resource_file).name, "key": key})
                values[node] = node.name
                continue
            tensor = node.meta.get("val")
            if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
                fail(node, "Only strided tensor inputs are supported")
            if resource_specs and tensor.device.type != "cpu":
                fail(node, "Modules with tensor resources currently require CPU inputs; device placement needs explicit lowering")
            shape, stride = list(tensor.shape), list(tensor.stride())
            if any(type(size) is not int for size in shape + stride):
                fail(node, "Dynamic tensor dimensions or strides are not supported")
            program["inputs"].append({"id": node.name, "type": "Tensor"})
            constraints.append({"value": node.name, "shape": shape, "stride": stride,
                                "dtype": str(tensor.dtype), "device": str(tensor.device),
                                "requiresGrad": tensor.requires_grad})
            values[node] = node.name
        elif node.op == "call_function" and node.target is operator.getitem:
            if len(node.args) != 2 or type(node.args[1]) is not int:
                fail(node, "Only static tuple selection is supported")
            source = values.get(node.args[0])
            if not isinstance(source, tuple) or not -len(source) <= node.args[1] < len(source):
                fail(node, "Expected an operation's tuple result")
            values[node] = source[node.args[1]]
        elif node.op == "call_function":
            if not isinstance(node.target, torch._ops.OpOverload):
                fail(node, "Only ATen operators are supported; control flow and custom calls need lowering")
            schema = node.target._schema
            if not schema.name.startswith("aten::") or schema.is_mutable:
                fail(node, "Custom or mutable operators are not supported")
            result = node.meta.get("val")
            results = result if isinstance(result, tuple) else (result,)
            if not results or not all(isinstance(item, torch.Tensor) for item in results):
                fail(node, "Only tensor or tuple-of-tensor results are supported")
            if len(results) != len(schema.returns):
                fail(node, "Result count does not match the operator schema")
            outputs = [f"{node.name}:{i}" for i in range(len(results))]
            operation = {"id": node.name,
                         "operator": f"{schema.name.replace('::', '.')}.{schema.overload_name or 'default'}",
                         "args": [argument(value, node) for value in node.args],
                         "kwargs": {key: argument(value, node) for key, value in node.kwargs.items()},
                         "outputs": outputs}
            if node.meta.get("stack_trace"):
                operation["source"] = node.meta["stack_trace"].strip()
            program["operations"].append(operation)
            values[node] = tuple(outputs) if isinstance(result, tuple) else outputs[0]
        elif node.op == "output":
            for i, value in enumerate(node.args[0]):
                resolved = argument(value, node)
                if "value" not in resolved:
                    fail(node, "Literal outputs are not supported")
                program["outputs"].append({"name": f"output_{i}", "value": resolved["value"]})
        else:
            fail(node, f"Unsupported graph node kind {node.op}")
    program["capture"] = {"framework": "pytorch", "version": torch.__version__, "inputs": constraints}
    if tensors:
        from safetensors.torch import save_file
        program["version"] = 2
        target = Path(resource_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".safetensors", delete=False) as stream:
                temporary = Path(stream.name)
            save_file(tensors, temporary)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="Python file:callable returning (module, args) or (module, args, kwargs)")
    parser.add_argument("--out", required=True, type=Path)
    options = parser.parse_args()
    file, separator, name = options.factory.rpartition(":")
    if not separator or not file or not name:
        parser.error("--factory must be a Python file:callable")
    source = Path(file).resolve()
    if source == options.out.resolve():
        parser.error("Output must not overwrite the Python source")
    # Explicitly selected user code: load it only when running the capture CLI.
    spec = importlib.util.spec_from_file_location("tora_capture_model", source)
    if spec is None or spec.loader is None:
        raise CaptureError(f"Cannot load factory from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    factory = getattr(module, name, None)
    if not callable(factory):
        raise CaptureError(f"Factory {name} is not callable")
    example = factory()
    if not isinstance(example, tuple) or len(example) not in (2, 3):
        raise CaptureError("Factory must return (module, args) or (module, args, kwargs)")
    resource_file = options.out.with_suffix(".safetensors")
    if resource_file.resolve() in (source, options.out.resolve()):
        parser.error("Tensor output must not overwrite the Python source or program")
    program = capture_program(*example, resource_file=resource_file)
    serialized = json.dumps(program, indent=2, allow_nan=False) + "\n"
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_text(serialized, encoding="utf-8")
    print(f"Captured {len(program['operations'])} operations to {options.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Capture failed: {error}", file=sys.stderr)
        sys.exit(1)
