# config.py
"""USER SPECIFIC PRESETTING"""
import torch.nn as nn


# This class holds the attributes needed for EVERY setting. It is shared between all configs
class Config:
    # Number of subjects to investigate (range from 1 to 109).
    selected_subjects = [1, 2, 3]
    # Select the experimental runs per subject (range from 1 to 14). Runs differ in tasks performed tasks!
    # Default: 1 to 14
    selected_runs = range(1, 14)
    # Select the event selection parameters --> See impact in mne_data_loader.py
    # TODO: Extract the correct classes (done via selection of runs), or we have to come to an agreement what to
    #  classify exactly.
    # TODO: Make all the config files a class with member variables or at least a structure. Put it in a config.py file
    # Remark:   This is a random pick.
    selected_classes = dict(both_hands_or_left_fist=2, both_feet_or_right_fist=3)
    time_before_event_s = -1.1  # Epochsize parameter: Start time before event.
    time_after_event_s = 4.0  # Epochsize parameter: Start time before event.
    # Number of channels to investigate (range from 1 to 64)
    selected_channels = range(1, 10)
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
    # Batch Size
    batch_size = 50
    # Select network architecture according to the nn_list(predefined in nn_models.py)
    nn_selection_idx = 1
    nn_list = ['SimpleFC', 'DeepFC']  # Extend if you want more. Add them in the nn_models.py module

    # Select optimizer parameters
    optimizer_selection_idx = 0  # Idx corresponds to entry optimizer_list (find below)
    learning_rate = 0.001
    weight_decay = 0
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
    # TODO : Make data augmentation module (e.g. add gaussian noise to channels)


# Here we can define more specific configurations. For example we need extra parameters or we have to
# overwrite parameters from the Config class

# Dummy Config
class MyDummyOwnConfig(Config):
    num_of_epochs = 1   # Dummy


# Dummy Config
class TimsConfig(Config):
    num_of_epochs = 1337   # Dummy

