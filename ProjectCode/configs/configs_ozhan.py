from configs.defaultconfig import DefaultConfig
# ==> Make sure to pick enough subjects! Otherwise baseline has too few labels!

# Own configs follow here
class ConfigNo01(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 30

    # Give it a unique name and a brief description if you like
    config_name = 'simpleNN'
    config_remark = 'This is a simple NN test.. nothing serious'
    selected_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    
    show_events_distribution = False
    show_eeg_sample_plot = False
    subjectIdx_to_plot = 1
    seconds_to_plot = 3
    channels_to_plot = 5
    
    nn_list = ['DeepFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0

    # Select optimizer parameters
    
    learning_rate = 1e-3
    weight_decay = 0.005
    momentum = 0  # Relevant only for SGDMomentum, else: ignored
    optimizer_list = ['SGD']  # Extend if you want more. Add them in the optimizers.py module
    optimizer_selection_idx = 0  # Idx corresponds to entry optimizer_list (find below)
    
    verbose = 'CRITICAL'
    
    batch_size = 128
    
class ConfigNo02(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 30

    # Give it a unique name and a brief description if you like
    config_name = 'deepNN'
    config_remark = 'This is a deep fc NN test.. nothing serious'
    selected_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    
    show_events_distribution = False
    show_eeg_sample_plot = False
    subjectIdx_to_plot = 1
    seconds_to_plot = 3
    channels_to_plot = 5
    
    nn_list = ['DeepFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0

    # Select optimizer parameters
    
    learning_rate = 1e-3
    weight_decay = 0.005
    momentum = 0.0  # Relevant only for SGDMomentum, else: ignored
    optimizer_list = ['Adam']  # Extend if you want more. Add them in the optimizers.py module
    optimizer_selection_idx = 0  # Idx corresponds to entry optimizer_list (find below)
    
    verbose = 'CRITICAL'
    
    batch_size = 128
        
class EEGNet(DefaultConfig):
    verbose = 'CRITICAL'
    
    selected_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    config_name = 'EEGNet'
    config_remark = 'This is a EEGNet NN test.. I am serious'
    
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    optimizer_list = ['Adam']  # Extend if you want more. Add them in the optimizers.py module
    optimizer_selection_idx = 0  # Idx corresponds to entry optimizer_list (find below)
    
    # Select optimizer parameters
    learning_rate = 1e-3
    weight_decay = 0.01
    
    
    num_of_epochs = 15
    batch_size = 128
    use_early_stopping = True
    es_patience = 10
    normalize = True

    
class ConfigConv1D(DefaultConfig):
    verbose = 'CRITICAL'
    
    config_name = 'ConvNet1D'
    config_remark = 'This is a cov NN test.. I am serious'
    selected_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    
    nn_list = ['ConvNet1D']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    optimizer_list = ['Adam']  # Extend if you want more. Add them in the optimizers.py module
    optimizer_selection_idx = 0  # Idx corresponds to entry optimizer_list (find below)
    learning_rate = 1e-3
    weight_decay = 0.02
    
    
    num_of_epochs = 15
    batch_size = 128
    use_early_stopping = True
    es_patience = 10
    normalize = True
    
    
class ConfigConvOzhan(DefaultConfig):
    verbose = 'CRITICAL'
    
    config_name = 'ConvNetOzhan'
    config_remark = 'This is a cov NN test.. I am serious'
    selected_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

    
    nn_list = ['ConvNetOzhan']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    optimizer_list = ['Adam']  # Extend if you want more. Add them in the optimizers.py module
    optimizer_selection_idx = 0  # Idx corresponds to entry optimizer_list (find below)
    learning_rate = 1e-3
    weight_decay = 0.02
    
    
    num_of_epochs = 50
    batch_size = 128
    use_early_stopping = False
    es_patience = 10
    
    normalize = True
    augment_with_gauss_noise = True
    augmentation_factor = 4
    
    
# Put them all in a list
list_of_configs = [ConfigConvOzhan]