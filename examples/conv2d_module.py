"""A stateful convolution module for automatic capture with external weights."""
import torch


class ConvolutionModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, (3, 2), stride=(2, 1), padding=(1, 0))
        with torch.no_grad():
            self.conv.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(2, 1, 3, 2) / 12 - 0.5)
            self.conv.bias.copy_(torch.tensor([0.25, -0.5]))
        self.register_buffer("offset", torch.tensor([[[[0.1]], [[-0.2]]]]))

    def forward(self, x):
        return torch.relu(self.conv(x) + self.offset)


def example():
    return ConvolutionModule().eval(), (torch.arange(72, dtype=torch.float32).reshape(1, 1, 8, 9) / 24 - 1,)
