# Here we define our network classes
import torch
import torch.nn as nn


# Fully Connected (FC) network
class SimpleFC(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dimension, output_dimension)
        )

    def forward(self, x):
        output = x.view(x.size(0), -1)
        return self.layers(output)


# Fully Connected (FC) little deep network
class DeepFC(nn.Module):
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


# Model selection
def get_nn_model(model_name, input_dimension, output_dimension):
    if model_name == 'SimpleFC':
        model = SimpleFC(input_dimension, output_dimension)
    elif model_name == 'DeepFC':
        model = DeepFC(input_dimension, output_dimension)
    else:
        raise Exception('Mismatch between nn_list in main.py and available names in get_nn_model')

    # Transfer model to gpu if possible
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device, "Will be used for training of this model.")
    if (device == torch.device("cuda")) and (torch.cuda.device_count() > 1):
        print("You have", torch.cuda.device_count(), "GPUs for training!")
        model = nn.DataParallel(model)
    model = model.to(device)

    return model
