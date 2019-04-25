from configs.defaultconfig import DefaultConfig


# Own configs follow here
class ConfigNo01(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 30

    # Give it a unique name and a brief description if you like
    config_name = 'simpleNN'
    config_remark = 'This is a simple NN test.. nothing serious'
    selected_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    
    show_events_distribution = True
    show_eeg_sample_plot = True
    subjectIdx_to_plot = 1
    seconds_to_plot = 3
    channels_to_plot = 5
    
    nn_list = ['SimpleFC', 'DeepFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0

    # Select optimizer parameters
    
    learning_rate = 1e-5
    weight_decay = 0
    momentum = 0  # Relevant only for SGDMomentum, else: ignored
    optimizer_list = ['Adam', 'SGD', 'SGDMomentum']  # Extend if you want more. Add them in the optimizers.py module
    optimizer_selection_idx = 1  # Idx corresponds to entry optimizer_list (find below)
    
    verbose = 'WARNING'


# Put them all in a list
list_of_configs = [ConfigNo01]