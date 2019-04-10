import matplotlib.pyplot as plt
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

"""PRESETTING"""
# Number of subjects to investigate (range from 1 to 109).
selected_subjects = [1]
# Select the experimental runs per subject (range from 1 to 14).
selected_runs = range(1, 14)
# Number of channels to investigate (range from 1 to 64)
selected_channels = range(1, 10)
# Show sample plot of 1 subject
show_sample_plot = True
seconds_to_plot = 3
subjectIdx_to_plot = 0 # idx corresponding to selected_subjects list
channels_to_plot = 5

"""CODE START"""
# Load the data
subjects = selected_subjects
runs = selected_runs
raw_EDF_list = []
for subj in subjects:
    fileNames = eegbci.load_data(subj, runs, path = 'RawDataMNE')
    raw_EDF = [read_raw_edf(f, preload=True, stim_channel='auto', verbose='WARNING') for f in fileNames]
    raw_EDF_list.append(concatenate_raws(raw_EDF))

raw_EDFs_merged = concatenate_raws(raw_EDF_list)
raw_EDFs_np = raw_EDFs_merged.get_data()

if show_sample_plot:
    raw_for_plotshow = raw_EDF_list[subjectIdx_to_plot]
    sfreq = raw_for_plotshow.info['sfreq']  # sample frequency
    timelength_s = int(seconds_to_plot * sfreq)
    data, times = raw_for_plotshow[:channels_to_plot, :timelength_s]
    plt.plot(times, data.T)
    plt.title('Sample channels')
    raw_for_plotshow.plot(n_channels=channels_to_plot, scalings='auto', title='Auto-scaled Data from arrays',
             show=True, block=True)
    raw_numpy = raw_for_plotshow.get_data()
    print("Type:", type(raw_numpy), " Shape:", raw_numpy.shape)

print("raw_EDFs_np.shape", raw_EDFs_np.shape)

