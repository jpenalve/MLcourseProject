import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from datasets import ChannelsVoltageDataset
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, find_events
import os


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


==> We map the classes the following:
    C0  Rest
    C1  Left Fist   (real or imaged) (in runs 3, 4, 7, 8, 11, and 12) (T1)
    C2  Both Fists  (real or imaged) (in runs 5, 6, 9, 10, 13, and 14) (T1)
    C3  Right Fist  (real or imaged) (in runs 3, 4, 7, 8, 11, and 12) (T2)
    C4  Both Feet   (real or imaged) (in runs 5, 6, 9, 10, 13, and 14) (T2)

    find T2 and make it C3
    find T2 and make it C4

    find T1 and make it C2

"""


def get_dataloader_objects(my_cfg):
    # We want the classes as defined above
    classes_to_extract = ['C0', 'C1', 'C2', 'C3', 'C4']
    for cls in classes_to_extract:
        """LOAD RAW DATA"""
        epoched, offset_to_subtract = get_epoched_data(my_cfg, cls)

        """DATA PREPARATION"""
        # Convert data from volt to millivolt
        # Pytorch expects float32 for input and int64 for labels.
        event_current_class_column = 2 #  event_previous_class_column = 1   event_start_sample_column = 0
        data = (epoched.get_data() * 1e6).astype(np.float32)  # Get all epochs as a 3D array.
        # -offset_to_subtract -> Classes made matching to CX definition
        labels = (epoched.events[:, event_current_class_column] - offset_to_subtract).astype(np.int64)

        # Split data in train test and validation set
        train_data_temp, test_data, train_labels_temp, test_labels = train_test_split(data, labels, test_size=my_cfg.test_split,
                                                                                      shuffle=True)
        train_data, val_data, train_labels, val_labels = train_test_split(train_data_temp, train_labels_temp,
                                                                          test_size=my_cfg.validation_split, shuffle=True)
        myTransforms = Compose([ToTensor()])  # TODO: This has to be more sophisticated. Should also be list selectable like the optimizers

        # Define datasets
        train_ds = ChannelsVoltageDataset(train_data, train_labels, myTransforms) # TODO: Should also be list selectable like the optimizers
        val_ds = ChannelsVoltageDataset(val_data, val_labels, myTransforms)
        test_ds = ChannelsVoltageDataset(test_data, test_labels, myTransforms)
        print("train_ds.shape", train_ds.data.shape)

        # Define data loader
        train_dl = DataLoader(train_ds, my_cfg.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, my_cfg.batch_size, shuffle=False)
        test_dl = DataLoader(test_ds, my_cfg.batch_size, shuffle=False)
        input_dimension_ = train_ds.data.shape[1] * train_ds.data.shape[2]


def get_epoched_data(config, class_to_extract):
    # Experimental runs per subject (range from 1 to 14). Runs differ in tasks performed tasks!
    if class_to_extract == 'C0':
        offset_to_subtract = 1
        runs = range(1, 14)
        selected_classes = dict(resting=1)
    elif class_to_extract == 'C1':
        offset_to_subtract = 1
        runs = [3, 4, 7, 8, 11, 12]
        selected_classes = dict(both_hands_or_left_fist=2)
    elif class_to_extract == 'C2':
        offset_to_subtract = 0
        runs = [5, 6, 9, 10, 13, 14]
        selected_classes = dict(both_hands_or_left_fist=2)
    elif class_to_extract == 'C3':
        offset_to_subtract = 0
        runs = [3, 4, 7, 8, 11, 12]
        selected_classes = dict(both_feet_or_right_fist=3)
    elif class_to_extract == 'C4':
        offset_to_subtract = -1
        runs = [5, 6, 9, 10, 13, 14]
        selected_classes = dict(both_feet_or_right_fist=3)


    # Load the data
    subjects = config.selected_subjects
    raw_EDF_list = []
    raw_data_path = 'RawDataMNE'

    # Dirty hack: If we are on the cluster, the path to the RawDataMNE changes
    this_path = os.path.dirname(os.path.abspath(__file__))
    if not "MLCourseProject" in this_path:
        # We are (most likely ;)...) on the cluster
        raw_data_path = '../../var/tmp/RawDataMNE'

    for subj in subjects:
        fileNames = eegbci.load_data(subj, runs, path=raw_data_path)
        raw_EDF = [read_raw_edf(f, preload=True, stim_channel='auto', verbose='WARNING') for f in fileNames]
        raw_EDF_list.append(concatenate_raws(raw_EDF))

    raw = concatenate_raws(raw_EDF_list)

    # Pick the events and select the epochs from them
    events = find_events(raw, shortest_event=0)
    epoched = Epochs(raw, events, event_id=selected_classes, tmin=config.time_before_event_s,
                     tmax=config.time_after_event_s, baseline=(None, 0), picks=None,
                     preload=False, reject=None, flat=None, proj=True, decim=1, reject_tmin=None, reject_tmax=None,
                     detrend=None, on_missing='error', reject_by_annotation=True, metadata=None, verbose=None)
    return epoched, offset_to_subtract
