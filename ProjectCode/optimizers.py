# Here we define our optimizers
from torch import optim
from braindecode.torch_ext.optimizers import AdamW

def get_optimizer(optimizer_name, learning_rate, model_parameters, sgd_momentum=0, weight_decay_factor=0):
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay_factor)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model_parameters, lr=learning_rate, weight_decay=weight_decay_factor)
    elif optimizer_name == 'SGDMomentum':
        optimizer = optim.SGD(model_parameters, lr=learning_rate, momentum=sgd_momentum,
                              weight_decay=weight_decay_factor)
    elif optimizer_name == 'AdamW':
        optimizer = AdamW(model_parameters, lr=learning_rate, weight_decay=weight_decay_factor)
    else:
        raise Exception('Mismatch between optimizer_list in main.py and available names in get_optimizer')
    return optimizer

