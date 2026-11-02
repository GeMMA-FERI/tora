"""Capture the example and save independent eager inputs/outputs for native checks."""
import json
from pathlib import Path
import sys

from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from conv2d_module import example
from capture_program import capture_program


def main():
    directory = ROOT / "_build/captured-module-test"
    directory.mkdir(parents=True, exist_ok=True)
    model, args = example()
    program = capture_program(model, args, resource_file=directory / "model.safetensors")
    (directory / "model.program.json").write_text(json.dumps(program, indent=2) + "\n", encoding="utf8")
    save_file({"input": args[0], "expected": model(*args).detach().contiguous()}, directory / "reference.safetensors")


if __name__ == "__main__":
    main()
