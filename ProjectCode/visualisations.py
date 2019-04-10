import matplotlib.pyplot as plt
import mne

def eeg_sample_plot(subject, seconds_to_plot, channels_to_plot, raw):
    raw_for_plotshow = raw[subject]
    sfreq = raw_for_plotshow.info['sfreq']  # sample frequency
    timelength_s = int(seconds_to_plot * sfreq)
    data, times = raw_for_plotshow[:channels_to_plot, :timelength_s]
    plt.plot(times, data.T)
    plt.title('Sample channels')
    raw_for_plotshow.plot(n_channels=channels_to_plot, scalings='auto', title='Auto-scaled Data from arrays', show=True, block=False)


def events_distribution_plot(events):
    # Events
    mne.viz.plot_events(events)
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
