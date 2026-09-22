"""Write a small archive for testing the Load Tensor node."""
from pathlib import Path
import torch
from safetensors.torch import save_file

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "_build/example-weights.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file({
        "linear.weight": torch.tensor([[1., 2.], [-3., 4.]], dtype=torch.float32),
        "linear.bias": torch.tensor([0.5, -0.5], dtype=torch.float32),
    }, target)
    print(target)
