import torch.nn as nn


# Fully Connected (FC) network
class Simple(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, output_dimension)
        )

    def forward(self, x):
        output = x.view(x.size(0), -1)
        return self.layers(output)


# Fully Connected (FC) little deep network
class Deep(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dimension)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

