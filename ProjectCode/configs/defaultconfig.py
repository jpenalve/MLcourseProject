# config.py
"""USER SPECIFIC PRESETTING"""
import torch.nn as nn


# This class holds the attributes needed for EVERY setting. It is shared between all configs

# ==> PLEASE PUBLISH CHANGES CLEARLY TO ALL USERS. ALL USER INHERIT THIS CLASS
# (non overwritten attributes take the DefaultConfigs values)
class DefaultConfig:
    # Number of subjects to investigate (range from 1 to 109).
    # ==> Subjects 88, 92 and 100 have overlapping events. Please exclude these subjects.
    selected_subjects = [1, 2, 3, 4]
    time_before_event_s = -1.1  # Epochsize parameter: Start time before event.
    time_after_event_s = 4.0  # Epochsize parameter: Start time before event.
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
    batch_size = 64 
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
    scheduler = None  # torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
    # Set loss function
    loss_fn = nn.CrossEntropyLoss()
    # Set number of epochs for training
    num_of_epochs = 3
    use_early_stopping = False
    es_patience = 5
    # TODO : Make data augmentation module (e.g. add gaussian noise to channels)
    
    normalize = True # Epoch normalization to mean=0.5, std=0.5
    
    
    # Warning messages for MNE related stuff
    verbose = None


# Here we can define more specific configurations. For example we need extra parameters or we have to
# overwrite parameters from the Config class

# Dummy Config
class MyDummyOwnConfig(DefaultConfig):
    num_of_epochs = 1   # Dummy




