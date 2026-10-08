"""Capture a static, stateless PyTorch module into Tora's program format."""
import argparse
import importlib.util
import json
import math
import operator
from pathlib import Path
import sys


class CaptureError(ValueError):
    pass


def capture_program(module, args, kwargs=None):
    import torch
    from torch.export.graph_signature import InputKind, OutputKind, TensorArgument

    kwargs = kwargs or {}
    if not isinstance(module, torch.nn.Module) or not isinstance(args, tuple):
        raise CaptureError("Expected a torch.nn.Module and a tuple of example inputs")
    if not all(isinstance(value, torch.Tensor) for value in (*args, *kwargs.values())):
        raise CaptureError("Only flat tensor inputs are supported; primitive inputs can be specialized during export")
    exported = torch.export.export(module, args, kwargs, strict=True)
    if exported.range_constraints:
        raise CaptureError("Symbolic shape constraints are not supported")
    for spec in exported.graph_signature.input_specs:
        if spec.kind != InputKind.USER_INPUT or not isinstance(spec.arg, TensorArgument):
            raise CaptureError(f"Unsupported input {spec.arg.name}: {spec.kind.name}; weights, buffers and constants need resource support")
    for spec in exported.graph_signature.output_specs:
        if spec.kind != OutputKind.USER_OUTPUT:
            raise CaptureError(f"Unsupported output {spec.kind.name}; mutation is not supported")

    program = {"version": 1, "inputs": [], "operations": [], "outputs": []}
    constraints = []
    values = {}

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
            tensor = node.meta.get("val")
            if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
                fail(node, "Only strided tensor inputs are supported")
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
    program = capture_program(*example)
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
