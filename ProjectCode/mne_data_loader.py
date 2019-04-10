from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, find_events


def load_the_edf_data(config):
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
                     detrend=None, on_missing='error', reject_by_annotation=True, metadata=None, verbose=None)
    return epoched
