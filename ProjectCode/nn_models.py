# Here we define our network classes

import torch
import torch.nn as nn

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# Simple Linear NN
class simpleANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(36 * 36, 2, bias=True),
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)
        return self.layers(input)


# Deep NN. Not so deep though.
class deepANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(36 * 36, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 2, bias=True)
        )

    def forward(self, input):
        input = input.view(input.size(0), -1)
        return self.layers(input)


# Model selecter, SGD parameter set.
def get_nn_model(model_name):
    if model_name == 'simple':
        model = simpleANN()
    elif model_name == 'deep':
        model = deepANN()
    else:
        print('Error at slecting model, choosing simple model instead.')
        model = simpleANN()
    print(dev, "will be used for training of this model.")
    if (dev == torch.device("cuda")) and (torch.cuda.device_count() > 1):
        print("You have", torch.cuda.device_count(), "GPUs for training!")
        model = nn.DataParallel(model)

    model = model.to(dev)
    return model
