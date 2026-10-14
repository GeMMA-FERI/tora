"""Create a deterministic convolution demo with external weights and reference maps."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / 'examples/conv2d-features.json').read_text())
    target = root / '_build'
    target.mkdir(exist_ok=True)
    generator = torch.Generator().manual_seed(config['seed'])
    batch, channels, height, width = config['inputShape']
    y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing='ij')
    images = []
    for index in range(batch):
        shift = index * 0.25
        radius = ((x - shift) ** 2 + (y + shift / 2) ** 2).sqrt()
        blob = torch.exp(-8 * radius ** 2)
        rings = 0.5 + 0.5 * torch.cos(22 * radius)
        stripes = 0.5 + 0.5 * torch.sin(13 * x + 7 * y + index)
        checker = ((torch.floor((x + 1) * 6) + torch.floor((y + 1) * 6)) % 2).float()
        image = torch.stack([0.7 * blob + 0.3 * stripes, 0.6 * rings + 0.4 * blob, 0.6 * checker + 0.4 * stripes])
        images.append(image + 0.025 * torch.randn(image.shape, generator=generator))
    tensors = {'input': torch.stack(images).contiguous()}
    current_channels = channels
    for stage in config['stages']:
        kh, kw = stage['kernel']
        count = stage['channels']
        per_group = current_channels // stage['groups']
        weights = torch.randn((count, per_group, kh, kw), generator=generator)
        if stage['name'] == 'stem':
            ky, kx = torch.meshgrid(torch.arange(-2, 3), torch.arange(-2, 3), indexing='ij')
            gaussian = torch.exp(-(kx.square() + ky.square()) / 2.0)
            kernels = torch.stack([gaussian, kx * gaussian, ky * gaussian,
                                   (kx + ky) * gaussian, (kx - ky) * gaussian,
                                   (kx.square() + ky.square() - 2) * gaussian,
                                   torch.cos(kx * 2.0) * gaussian, torch.cos(ky * 2.0) * gaussian])
            weights = kernels[:, None] * torch.tensor([1.0, 0.7, 0.4])[None, :, None, None]
        weights /= weights.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        tensors[f'{stage["name"]}.weight'] = weights.contiguous()
        tensors[f'{stage["name"]}.bias'] = torch.linspace(0.02, 0.16, count)
        current_channels = count
    tensor_file = target / 'conv2d-features.safetensors'
    save_file(tensors, tensor_file)
    loaded = load_file(tensor_file)
    for name, tensor in tensors.items():
        torch.testing.assert_close(loaded[name], tensor, rtol=0, atol=0)
    value = loaded['input']
    reference = {}
    for stage in config['stages']:
        name = stage['name']
        value = torch.nn.functional.conv2d(value, loaded[f'{name}.weight'], loaded[f'{name}.bias'],
                                          stride=stage['stride'], padding=stage['padding'],
                                          dilation=stage['dilation'], groups=stage['groups'])
        reference[f'{name}.conv'] = value.contiguous()
        value = torch.relu(value)
        reference[f'{name}.relu'] = value.contiguous()
        assert torch.isfinite(value).all(), name
    assert list(value.shape) == [2, 6, 48, 64]
    save_file(reference, target / 'conv2d-features-reference.safetensors')
    summary = {name: {'shape': list(value.shape), 'minimum': value.min().item(),
                      'maximum': value.max().item(), 'mean': value.mean().item()}
               for name, value in reference.items()}
    (target / 'conv2d-features-expected.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(tensor_file)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
