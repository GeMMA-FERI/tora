"""Write matching Float64 tensors and a PyTorch reference result for the graphs."""
import json
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file

root = Path(__file__).resolve().parents[1]
definitions = json.loads((root / 'examples/conv2d-tensors.json').read_text())
tensors = {name: torch.tensor(item['values'], dtype=torch.float64).reshape(item['shape'])
           for name, item in definitions.items()}
target = root / '_build/conv2d-tensors.safetensors'
target.parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, target)
loaded = load_file(target)
reference = torch.nn.functional.conv2d(tensors['input'], tensors['weight'], tensors['bias'])
for mode in ['text', 'mixed', 'safetensors']:
    selected = {key: tensors[key] if mode == 'text' or (mode == 'mixed' and key == 'input') else loaded[key]
                for key in tensors}
    actual = torch.nn.functional.conv2d(selected['input'], selected['weight'], selected['bias'])
    torch.testing.assert_close(actual, reference)
expected = {'dtype': str(reference.dtype), 'shape': list(reference.shape), 'values': reference.tolist(),
            'stride': [1, 1], 'padding': [0, 0], 'dilation': [1, 1], 'groups': 1}
(root / '_build/conv2d-expected.json').write_text(json.dumps(expected, indent=2) + '\n')
print(target)
print(json.dumps(expected))
