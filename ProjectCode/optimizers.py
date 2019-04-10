# Here we define our optimizers
import torch


def get_optimizer(optimizer_name, learning_rate, model_parameters):
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model_parameters, lr=learning_rate)
    return optimizer

