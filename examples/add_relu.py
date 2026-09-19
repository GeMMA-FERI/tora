import torch


class AddReLU(torch.nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y)


def example():
    return AddReLU().eval(), (torch.tensor([-2.0, 3.0]), torch.tensor([1.0, -1.0]))
