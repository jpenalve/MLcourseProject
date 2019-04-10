from visualisations import eeg_sample_plot, events_distribution_plot
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types, find_events, events_from_annotations
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from datasets import ChannelsVoltageDataset
import torch.nn as nn
from nn_models import get_nn_model
from optimizers import get_optimizer
from utils_train import fit, test

"""
The data are provided here in EDF+ format (containing 64 EEG signals, each sampled at 160 samples per second, and an 
annotation channel).
The .event files and the annotation channels in the corresponding .edf files contain identical data.

Each annotation includes one of three codes (T0, T1, or T2):

Coded as label = 1:
    T0 corresponds to rest
    
Coded as label = 2:
    T1 corresponds to onset of motion (real or imagined of
        the left fist (in runs 3, 4, 7, 8, 11, and 12)
        both fists (in runs 5, 6, 9, 10, 13, and 14)
        
Coded as label = 3:        
    T2 corresponds to onset of motion (real or imagined) of
        the right fist (in runs 3, 4, 7, 8, 11, and 12)
        both feet (in runs 5, 6, 9, 10, 13, and 14)
"""


"""USER SPECIFIC PRESETTING"""
# Number of subjects to investigate (range from 1 to 109).
selected_subjects = [1, 2]
# Select the experimental runs per subject (range from 1 to 14). Runs differ in tasks performed tasks! Default: 1 to 14
selected_runs = range(1, 14)
# Select the event selection parameters
# TODO: Extract the correct classes (done via selection of runs), or we have to come to an agreement what to classify
#       exactly.
# TODO: Make all the config files a class with member variables or at least a structure. Put it in a config.py file
# Remark:   This is a random pick.
selected_classes = dict(both_hands_or_left_fist=2, both_feet_or_right_fist=3)
time_before_event_s = -1.1  # Epochsize parameter: Start time before event.
time_after_event_s = 4.0    # Epochsize parameter: Start time before event.
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
# Select network architecture (predefined in nn_models.py)
nn_selection_idx = 1
nn_list = ['SimpleFC', 'DeepFC']  # Extend if you want more. Add them in the nn_models.py module

# Select optimizer parameters
optimizer_selection_idx = 0
learning_rate = 0.001
weight_decay = 0
momentum = 0  # Relevant only for SGDMomentum, else: ignored
optimizer_list = ['Adam', 'SGD', 'SGDMomentum']  # Extend if you want more. Add them in the optimizers.py module
# Adaption of learning rate?
scheduler = None # torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
# Set loss function
loss_fn = nn.CrossEntropyLoss()
# Set number of epochs for training
num_of_epochs = 3
use_early_stopping = False

"""LOAD RAW DATA"""
# Load the data
subjects = selected_subjects
runs = selected_runs
raw_EDF_list = []
for subj in subjects:
    fileNames = eegbci.load_data(subj, runs, path='RawDataMNE')
    raw_EDF = [read_raw_edf(f, preload=True, stim_channel='auto', verbose='WARNING') for f in fileNames]
    raw_EDF_list.append(concatenate_raws(raw_EDF))

raw = concatenate_raws(raw_EDF_list)

# Pick the events and select the epochs from them
events = find_events(raw, shortest_event=0)
epoched = Epochs(raw, events, event_id=selected_classes, tmin=time_before_event_s, tmax=time_after_event_s, baseline=(None, 0), picks=None,
                 preload=False, reject=None, flat=None, proj=True, decim=1, reject_tmin=None, reject_tmax=None,
                 detrend=None, on_missing='error', reject_by_annotation=True, metadata=None, verbose=None)

"""SHOW DATA"""
# Show some sample EEG data if desired
if show_eeg_sample_plot:
    eeg_sample_plot(subjectIdx_to_plot, seconds_to_plot, channels_to_plot, raw_EDF_list)
if show_events_distribution:
    events_distribution_plot(events)

"""CLASSIFICATION"""
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
train_data_temp, test_data, train_labels_temp, test_labels = train_test_split(data, labels, test_size=test_split,
                                                                              shuffle=True)
train_data, val_data, train_labels, val_labels = train_test_split(train_data_temp, train_labels_temp,
                                                                  test_size=validation_split, shuffle=True)
myTransforms = Compose([ToTensor()])  # TODO: This has to be more sophisticated
# Define datasets
train_ds = ChannelsVoltageDataset(train_data, train_labels, myTransforms)
val_ds = ChannelsVoltageDataset(val_data, val_labels, myTransforms)
test_ds = ChannelsVoltageDataset(test_data, test_labels, myTransforms)
print("train_ds.shape", train_ds.data.shape)
# Define data loader
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size, shuffle=False)

# Get the model
input_dimension_ = train_ds.data.shape[1] * train_ds.data.shape[2]
model_untrained = get_nn_model(nn_list[nn_selection_idx], input_dimension=input_dimension_,
                               output_dimension=len(selected_classes))

# Get the optimizer
optimizer = get_optimizer(optimizer_list[optimizer_selection_idx], learning_rate, model_untrained.parameters(),
                          momentum, weight_decay)

# Train and show validation loss
train_losses, train_accuracies, val_losses, val_accuracies, model_trained = fit(train_dl, val_dl, model_untrained, optimizer, loss_fn, num_of_epochs)

# Test the net
test(model_trained, test_dl, loss_fn, print_loss=True)
# TODO: Store the results of the training with all the parameters from the """PRESETTING""" section above.