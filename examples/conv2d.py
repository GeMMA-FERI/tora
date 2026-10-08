import torch


class Convolution(torch.nn.Module):
    def forward(self, x, weight, bias):
        return torch.nn.functional.conv2d(x, weight, bias, stride=(2, 1), padding=(1, 0), dilation=(1, 2))


def example():
    return Convolution().eval(), (
        torch.arange(72, dtype=torch.float32).reshape(1, 1, 8, 9),
        torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2),
        torch.tensor([0.5]),
    )
