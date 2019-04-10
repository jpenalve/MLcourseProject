import matplotlib.pyplot as plt
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

# # Download and Plot Single Subject Data

# Load and preprocess data
subject = 1
runs = range(1, 14) # There are 14 runs per subject. We want to look at all runs.

fnames = eegbci.load_data(subject, runs, path='RawDataMNE')
raws = [read_raw_edf(f, preload=True, stim_channel='auto') for f in fnames]
raw = concatenate_raws(raws)

# Extract data from the first 5 channels, from 1 s to 3 s.
sfreq = raw.info['sfreq']
data, times = raw[:5, :]
plt.plot(times, data.T)
plt.title('Sample channels')

raw.plot(n_channels=5, scalings='auto', title='Auto-scaled Data from arrays',
         show=True, block=True)

raw_numpy = raw.get_data()

print("Type:", type(raw_numpy), " Shape:", raw_numpy.shape)

# # Download Multiple Subject Data

# In[81]:

"""
Subject = [1,2,3,4,5,6,7,8,9,10]  
Runs = [1,2,3,4,5,6]  
Raw_List = []

for subj in Subject:
    fnames = eegbci.load_data(subj, Runs, path = 'RawDataMNE')
    raws = [read_raw_edf(f, preload=True, stim_channel='auto',verbose='WARNING') for f in fnames]
    Raw_List.append(concatenate_raws(raws))

Raw_List_All = concatenate_raws(Raw_List)
Raw_Numpy_All = Raw_List_All.get_data()


# In[105]:


Raw_Numpy_All.shape
plt.plot(Raw_Numpy_All[-1,:])
plt.figure()
plt.plot(Raw_Numpy_All[-1,50000:51000])
plt.figure()
plt.plot(Raw_Numpy_All[1,50000:51000]*1e6)
"""
