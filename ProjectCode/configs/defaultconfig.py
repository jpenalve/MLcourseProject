# config.py
"""USER SPECIFIC PRESETTING"""
import torch.nn as nn


# This class holds the attributes needed for EVERY setting. It is shared between all configs

# ==> PLEASE PUBLISH CHANGES CLEARLY TO ALL USERS. ALL USER INHERIT THIS CLASS
# (non overwritten attributes take the DefaultConfigs values)
class DefaultConfig:
    config_name = 'DEFAULT'
    config_remark = 'Default... just default'
    # Number of subjects to investigate (range from 1 to 109).
    # ==> Subjects 88, 92 and 100 have overlapping events. Please exclude these subjects.
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
    time_before_event_s = -1.1  # Epochsize parameter: Start time before event.
    time_after_event_s = 4.0  # Epochsize parameter: Time after event.
    # Number of channels to investigate (range from 1 to 64)
    selected_channels = range(1, 64)
    # Show sample plot of 1 subject
    show_eeg_sample_plot = False
    subjectIdx_to_plot = 0
    seconds_to_plot = 3
    channels_to_plot = 5
    # Show events distribution over selected_subjects
    show_events_distribution = False

    
    
    # Train / Test / Validation Split
    train_split = 0.9
    test_split = 0.1
    validation_split = 0.1  # This is the share of the train_split
    # Batch Size. Batch size should be powers of 2 for better utilization of GPUs.
    batch_size = 50
    # Select network architecture according to the nn_list(predefined in nn_models_getter.py)
    nn_selection_idx = 1
    nn_list = ['SimpleFC', 'DeepFC', 'EEGNet', 'ConvNet01']  # Extend if you want more. Add them in the nn_models_getter.py module

    # Select optimizer parameters
    optimizer_selection_idx = 0  # Idx corresponds to entry optimizer_list (find below)
    learning_rate = 0.001
    weight_decay = 0.000075
    momentum = 0  # Relevant only for SGDMomentum, else: ignored
    optimizer_list = ['Adam', 'SGD', 'SGDMomentum']  # Extend if you want more. Add them in the optimizers.py module
    # Adaption of learning rate?
    # TODO: Make scheduler module like the optimizer module
    scheduler = None  # if true: torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.25)
    schStepSize = 10
    schGamma = 0.25
    # Set loss function
    loss_fn = nn.CrossEntropyLoss() # TODO: Maybe we have to apply class weighting (if we care about the under represented classes..)
    # Set number of epochs for training
    num_of_epochs = 25
    use_early_stopping = False
    es_patience = 5
    # TODO : Make data augmentation module (e.g. add gaussian noise to channels)
    
    normalize = True  # Epoch normalization to mean=0.5, std=0.5

    augment_with_gauss_noise = True
    augment_std_gauss = 0.2  # (See EEG Review Roy et. al. 2019)
    augmentation_factor = 3

    # Warning messages for MNE related stuff
    verbose = None
    dropout_perc = 0 # default is no dropout
    
    curve_name = "NoName"
    
    #nClasses = 10 # Need to change other parts as well
    nClasses = 2 # Need to change other parts as well
    
    removeLastData = False
    downSample = 1 # Downsamples the epoch time samples with this factor. Simply selects each downSample'th  data.
    
    # Drops out square tiles of ChannelxTime for each epoch. Possible to drop whole channels (for all time parts) or same time parts (for all channels).
    dropOut = False
    dropOutChOnly = False
    dropOutTimeOnly = False
    
    dropOutTilePerc = 0.5
    dropOutTimeTile = 40
    dropOutChannelTile = 4
    
    Elec2D = False

# Here we can define more specific configurations. For example we need extra parameters or we have to

# overwrite parameters from the Config class

# Dummy Config
class MyDummyOwnConfig(DefaultConfig):
    num_of_epochs = 1   # Dummy




