from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, find_events
from visualisations import eeg_sample_plot, events_distribution_plot


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

def get_epoched_data(config):
    # Load the data
    subjects = config.selected_subjects
    runs = config.selected_runs
    raw_EDF_list = []
    for subj in subjects:
        fileNames = eegbci.load_data(subj, runs, path='RawDataMNE')
        raw_EDF = [read_raw_edf(f, preload=True, stim_channel='auto', verbose='WARNING') for f in fileNames]
        raw_EDF_list.append(concatenate_raws(raw_EDF))

    raw = concatenate_raws(raw_EDF_list)

    # Pick the events and select the epochs from them
    events = find_events(raw, shortest_event=0)
    epoched = Epochs(raw, events, event_id=config.selected_classes, tmin=config.time_before_event_s,
                     tmax=config.time_after_event_s, baseline=(None, 0), picks=None,
                     preload=False, reject=None, flat=None, proj=True, decim=1, reject_tmin=None, reject_tmax=None,
                     detrend=None, on_missing='error', reject_by_annotation=True, metadata=None, verbose='WARNING')
    
    """SHOW DATA"""
    # Show some sample EEG data if desired
    if config.show_eeg_sample_plot:
        eeg_sample_plot(config.subjectIdx_to_plot, config.seconds_to_plot, config.channels_to_plot, raw_EDF_list)
    if config.show_events_distribution:
        events_distribution_plot(epoched.events)
        
        
    return epoched
