from configs.defaultconfig import DefaultConfig
# ==> Subjects 88, 89, 92 and 100 have overlapping events. Please exclude these subjects.
# ==> Make sure to pick enough subjects! Otherwise baseline has too few labels!

# Own configs follow here
class ConfigNo01(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 1

    # Give it a unique name and a brief description if you like
    config_name = 'simpleNN'
    config_remark = 'This is a simple NN test.. nothing serious'

class ConfigNo02(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 1

    # Give it a unique name and a brief description if you like
    config_name = 'example1'
    config_remark = 'Another example here'

class ConfigNo03(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 1

    # Give it a unique name and a brief description if you like
    config_name = 'example2'
    config_remark = 'Super crazy network tested with normal settings'
    
class ConfigConv3dKeras(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 5

    # Give it a unique name and a brief description if you like
    config_name = 'ConvNet3DFFT'
    config_remark = 'This is a cov NN test.. I am serious'
    selected_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    augment_with_gauss_noise = False
    
    show_events_distribution = False
    show_eeg_sample_plot = False
    subjectIdx_to_plot = 1
    seconds_to_plot = 3
    channels_to_plot = 5
    
    nn_list = ['ConvNet3DFFT']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0

    # Select optimizer parameters
    
    learning_rate = 1e-3
    weight_decay = 0.008
    momentum = 0.01  # Relevant only for SGDMomentum, else: ignored
    optimizer_list = ['SGD']  # Extend if you want more. Add them in the optimizers.py module
    optimizer_selection_idx = 0  # Idx corresponds to entry optimizer_list (find below)
    
    verbose = 'CRITICAL'
    
    batch_size = 128
    use_early_stopping = True
    
    print_dLoss = False
    
    es_patience = 20
    
    normalize = True
    

# Put them all in a list
list_of_configs = [ConfigConv3dKeras]