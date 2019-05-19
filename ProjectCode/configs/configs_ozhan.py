from configs.defaultconfig import DefaultConfig
# ==> Make sure to pick enough subjects! Otherwise baseline has too few labels!

# Own configs follow here
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
    selected_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                         20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                         30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                         40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                         50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                         60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                         70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                         80, 81, 82, 83, 84, 85, 86, 87,
                         90, 91, 93, 94, 95, 96, 97, 98, 99,
                         101, 102, 103, 104, 105, 106, 107, 108, 109]
    nSubj = 60
    selected_subjects = selected_subjects[:nSubj]
    
    
    nn_list = ['ConvNetOzhan3D']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    optimizer_list = ['Adam']  # Extend if you want more. Add them in the optimizers.py module
    optimizer_selection_idx = 0  # Idx corresponds to entry optimizer_list (find below)
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    scheduler = True  
    schStepSize = 50
    schGamma = 0.5
    
    num_of_epochs = 200
    batch_size = 256
    use_early_stopping = True
    es_patience = num_of_epochs
    
    normalize = True
    augment_with_gauss_noise = False
    augmentation_factor = 2
    augment_std_gauss = 0.2
    dropOut = False
    dropOutChOnly = False
    dropOutTimeOnly = False
    dropOutTilePerc = 0.5
    dropOutTimeTile = 40
    dropOutChannelTile = 8
    
    time_before_event_s = 0.0  # Epochsize parameter: Start time before event.
    time_after_event_s = 2.0  # Epochsize parameter: Time after event.
    downSample = 1
    
    show_events_distribution = False
    removeLastData = True
    
    Elec2D = True
    wSize = 10
    wCropped = True
    
# Put them all in a list
list_of_configs = [ConfigConvOzhan]