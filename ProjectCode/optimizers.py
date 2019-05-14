# Here we define our optimizers
from torch import optim
from braindecode.torch_ext.optimizers import AdamW

def get_optimizer(config, model_parameters):
    
    optimizer_name = config.optimizer_list[config.optimizer_selection_idx]
    learning_rate = config.learning_rate
    sgd_momentum = config.momentum
    weight_decay_factor = config.weight_decay

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

