from visualisations import eeg_sample_plot, events_distribution_plot

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from datasets import ChannelsVoltageDataset
from neural_nets.nn_models_getter import get_nn_model
from optimizers import get_optimizer
from utils_train import fit, test
from configs import configs_tim
from mne_data_loader import get_epoched_data


""" USER: SELECT THE CONFIGURATION YOU NEED """
myList = configs_tim.list_of_configs
#myList = configs_joaquin.list_of_configs
#myList = configs_oezhan.list_of_configs

for myCfg in myList:
    """LOAD RAW DATA"""
    epoched = get_epoched_data(myCfg)

    """SHOW DATA"""
    # Show some sample EEG data if desired
    if myCfg.show_eeg_sample_plot:
        eeg_sample_plot(myCfg.subjectIdx_to_plot, myCfg.seconds_to_plot, myCfg.channels_to_plot, myCfg.raw_EDF_list)
    if myCfg.show_events_distribution:
        events_distribution_plot(epoched.events)

    """DATA PREPARATION"""
    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    # TODO: Make this all in the dataset class
    event_start_sample_column = 0
    event_previous_class_column = 1
    event_current_class_column = 2

    data = (epoched.get_data() * 1e6).astype(np.float32)  # Get all epochs as a 3D array.
    labels = (epoched.events[:, event_current_class_column] - 2).astype(np.int64)  # -2 -> Classes must be 0 indexed
    assert len(data) == len(labels)  # Check format
    # Split data in train test and validation set
    train_data_temp, test_data, train_labels_temp, test_labels = train_test_split(data, labels, test_size=myCfg.test_split,
                                                                                  shuffle=True)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data_temp, train_labels_temp,
                                                                      test_size=myCfg.validation_split, shuffle=True)
    myTransforms = Compose([ToTensor()])  # TODO: This has to be more sophisticated
    # Define datasets
    train_ds = ChannelsVoltageDataset(train_data, train_labels, myTransforms)
    val_ds = ChannelsVoltageDataset(val_data, val_labels, myTransforms)
    test_ds = ChannelsVoltageDataset(test_data, test_labels, myTransforms)
    print("train_ds.shape", train_ds.data.shape)
    # Define data loader
    train_dl = DataLoader(train_ds, myCfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, myCfg.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, myCfg.batch_size, shuffle=False)
    input_dimension_ = train_ds.data.shape[1] * train_ds.data.shape[2]

    """CLASSIFICATION"""
    # Get the model
    model_untrained = get_nn_model(myCfg.nn_list[myCfg.nn_selection_idx], input_dimension=input_dimension_, #TODO: More sophisticated models needed
                                   output_dimension=len(myCfg.selected_classes))

    # Get the optimizer
    optimizer = get_optimizer(myCfg.optimizer_list[myCfg.optimizer_selection_idx], myCfg.learning_rate,
                              model_untrained.parameters(), myCfg.momentum, myCfg.weight_decay)

    # Train and show validation loss
    train_losses, train_accuracies, val_losses, val_accuracies, model_trained = fit(train_dl, val_dl, model_untrained,
                                                                                    optimizer, myCfg.loss_fn,
                                                                                    myCfg.num_of_epochs)
    # Test the net
    test(model_trained, test_dl, myCfg.loss_fn, print_loss=True)
    # TODO: Store the results of the training with all the parameters from the """PRESETTING""" section above.