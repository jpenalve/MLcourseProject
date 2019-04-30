import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from datasets import ChannelsVoltageDataset
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, find_events
import os
from visualisations import eeg_sample_plot, events_distribution_plot
import torch

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

The runs correspond to:
run 	        task
1 	            Baseline, eyes open
2 	            Baseline, eyes closed
3, 7, 11 	    Motor execution: left vs right hand
4, 8, 12 	    Motor imagery: left vs right hand
5, 9, 13 	    Motor execution: hands vs feet
6, 10, 14 	    Motor imagery: hands vs feet

POSSIBLE LABELS                 APPEAR IN RUNS      ACTUAL LABEL IN RUNS    OUR OFFSET NEEDED
0 Baseline, eyes open           1                   T0(=1)                  0
1 Baseline, eyes closed         2                   T0(=1)                  -1
2 Motor Ex: Left Hand           3,7,11              T1(=2)                  0
3 Motor Ex: Right Hand          3,7,11              T2(=3)                  0
4 Motor Im: Left Hand           4,8,12              T1(=2)                  2
5 Motor Im: Right Hand          4,8,12              T2(=3)                  2
6 Motor Ex: Both Hands          5,9,13              T1(=2)                  4
7 Motor Ex: Both Feet           5,9,13              T2(=3)                  4
8 Motor Im: Both Hands          6,10,14             T1(=2)                  6
9 Motor Im: Both Feet           6,10,14             T2(=3)                  6


"""


def get_dataloader_objects(my_cfg):
    """LOAD RAW DATA"""
    epoched = get_epoched_data(my_cfg)

    """DATA PREPARATION"""
    # Convert data from volt to millivolt
    # Pytorch expects float32 for input and int64 for labels.
    event_current_class_column = 2  # event_previous_class_column = 1   event_start_sample_column = 0

    data = (epoched.get_data() * 1e6)  # Get all epochs as a 3D array.
    data = data[:, :-1, :]  # We do not want to feed in the labels as inputs

    # -offset_to_subtract -> Classes made matching to CX definition
    labels = epoched.events[:, event_current_class_column]

    # Split data in train test and validation set. Stratify makes sure the label distribution is the same
    temp_data, test_data, temp_labels, test_labels = train_test_split(data, labels, test_size=my_cfg.test_split,
                                                                      shuffle=True, stratify=labels)

    train_data, val_data, train_labels, val_labels = train_test_split(temp_data, temp_labels,
                                                                      test_size=my_cfg.validation_split, shuffle=True,
                                                                      stratify=temp_labels)

    # Convert them to Tensors already. torch.float is needed for GPU.
    train_data = torch.tensor(train_data, dtype=torch.float)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_data = torch.tensor(val_data, dtype=torch.float)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    myTransforms = None  # TODO: This has to be more sophisticated. Should also be list selectable like the optimizers

    # Define datasets
    train_ds = ChannelsVoltageDataset(train_data, train_labels,
                                      myTransforms)  # TODO: Should also be list selectable like the optimizers
    val_ds = ChannelsVoltageDataset(val_data, val_labels, myTransforms)
    test_ds = ChannelsVoltageDataset(test_data, test_labels, myTransforms)

    # Define data loader
    train_dl = DataLoader(train_ds, my_cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, my_cfg.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, my_cfg.batch_size, shuffle=False)
    input_dimension_ = train_ds.data.shape[1] * train_ds.data.shape[2]
    output_dimension_ = epoched.events.shape[1]

    return train_dl, val_dl, test_dl, input_dimension_, output_dimension_


def get_epoched_data(my_cfg):
    print("Data is being loaded using MNE...")
    # Experimental runs per subject (range from 1 to 14). Runs differ in tasks performed tasks!
    runs = range(1, 14)
    selected_classes = None  # if none all are selected

    # Load the data
    subjects = my_cfg.selected_subjects
    raw_EDF_list = []
    current_path = os.path.abspath(__file__)
    # print(current_path)
    if 'studi7/home/ProjectCode/' in current_path:
        data_path = '../../var/tmp/RawDataMNE'
        print('We are on the cluster...')
        data_path = '../../var/tmp/RawDataMNE'
    else:
        print('We are not on the cluster...')
        data_path = 'RawDataMNE'

    for subj in subjects:
        fileNames = eegbci.load_data(subj, runs, path=data_path)
        raw_EDF = [read_raw_edf(f, preload=True, stim_channel='auto', verbose='WARNING') for f in fileNames]
        raw_EDF_list.append(concatenate_raws(raw_EDF))

    raw = concatenate_raws(raw_EDF_list)

    # Pick the events and select the epochs from them
    events = find_events(raw, shortest_event=0)
    epoched = Epochs(raw, events, event_id=selected_classes, tmin=my_cfg.time_before_event_s,
                     tmax=my_cfg.time_after_event_s, baseline=(None, 0), picks=None,
                     preload=False, reject=None, flat=None, proj=True, decim=1, reject_tmin=None, reject_tmax=None,
                     detrend=None, on_missing='error', reject_by_annotation=True, metadata=None, verbose=my_cfg.verbose)

    epoched.events[:, 2] = epoched.events[:, 2] - 1
    """SHOW DATA"""
    # Show some sample EEG data if desired
    if my_cfg.show_eeg_sample_plot:
        eeg_sample_plot(my_cfg.subjectIdx_to_plot, my_cfg.seconds_to_plot, my_cfg.channels_to_plot, raw_EDF_list)
    if my_cfg.show_events_distribution:
        events_distribution_plot(epoched.events)

    print("...data loading with MNE was finished. \n")

    return epoched
