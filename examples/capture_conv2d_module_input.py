"""Capture the convolution module with a file-backed input for editor testing."""
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from capture_program import capture_program
    from conv2d_module import example

    directory = root / "_build"
    model, args = example()
    program = capture_program(model, args, resource_file=directory / "conv2d-module-with-input.safetensors")
    input_id = program["inputs"][0]["id"]
    input_file = directory / "conv2d-module-input.safetensors"
    with torch.no_grad():
        expected = model(*args)
    save_file({"input": args[0].contiguous(), "expected": expected.contiguous()}, input_file)

    # Turn the example input into a resource before graph export, using the
    # same value ID so the captured operation arguments remain unchanged.
    program["resources"].append({"id": input_id, "type": "Tensor", "file": input_file.name, "key": "input"})
    program["inputs"] = []
    program["capture"]["inputs"] = []
    target = directory / "conv2d-module-with-input.program.json"
    target.write_text(json.dumps(program, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(target)
    print(f"Input shape: {list(args[0].shape)}; output shape: {list(expected.shape)}")
    print(f"Expected output range: {expected.min().item():.6f} to {expected.max().item():.6f}")


if __name__ == "__main__":
    main()
