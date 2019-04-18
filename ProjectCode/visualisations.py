# Define the functions for plotting or visualizing here
import matplotlib.pyplot as plt
import mne
import numpy as np


def eeg_sample_plot(subject, seconds_to_plot, channels_to_plot, raw):
    raw_for_plotshow = raw[subject]
    sfreq = raw_for_plotshow.info['sfreq']  # sample frequency
    timelength_s = int(seconds_to_plot * sfreq)
    
    ncols = 3
    nrows = int(np.ceil(channels_to_plot/ncols))
    plt.figure(figsize=(20, 4*nrows))
    
    # Plot n image
    for i in range(channels_to_plot):
        plt.subplot(nrows,ncols,i+1)
        plt.title("Channel #"+ str(i+1))
        data, times = raw_for_plotshow[i, :timelength_s]
        plt.plot(times, data.T)
        
    plt.figure()
    raw_for_plotshow.plot(n_channels=channels_to_plot, scalings='auto', title='Auto-scaled Data from arrays', show=True, block=False)


def events_distribution_plot(events):
    # Events
    mne.viz.plot_events(events, show=False)
    plt.title('Show event distribution over the merged dataset')
    plt.tight_layout()
    """ Each annotation includes one of three codes (T0, T1, or T2):
    
        T0 corresponds to rest
        T1 corresponds to onset of motion (real or imagined) of
            the left fist (in runs 3, 4, 7, 8, 11, and 12)
            both fists (in runs 5, 6, 9, 10, 13, and 14)
        T2 corresponds to onset of motion (real or imagined) of
            the right fist (in runs 3, 4, 7, 8, 11, and 12)
            both feet (in runs 5, 6, 9, 10, 13, and 14)
    """

