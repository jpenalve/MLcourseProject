import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import ChannelsVoltageDataset
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, find_events, concatenate_epochs
import os
from visualisations import eeg_sample_plot, events_distribution_plot
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
0 Baseline, eyes open           1                   T0(=1)                  1
1 Baseline, eyes closed         2                   T0(=1)                  0
2 Motor Ex: Left Hand           3,7,11              T1(=2)                  0
3 Motor Ex: Right Hand          3,7,11              T2(=3)                  0
4 Motor Im: Left Hand           4,8,12              T1(=2)                  -2
5 Motor Im: Right Hand          4,8,12              T2(=3)                  -2
6 Motor Ex: Both Hands          5,9,13              T1(=2)                  -4
7 Motor Ex: Both Feet           5,9,13              T2(=3)                  -4
8 Motor Im: Both Hands          6,10,14             T1(=2)                  -6
9 Motor Im: Both Feet           6,10,14             T2(=3)                  -6

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
    
    if my_cfg.removeLastData:
        remaining = data.shape[2] % 10
        data = data[:, :, :-remaining]
        
    input_dimension_ = data.shape[1] * data.shape[2]
    output_dimension_ = my_cfg.nClasses 
    
    if my_cfg.Elec2D:
        data = dataset_1Dto2D(data)
        input_dimension_ = data.shape[1] * data.shape[2] * data.shape[3]
        
    # Normalize data
    if my_cfg.normalize:
        data = normalize(data)
        
    print(str(data.shape[2])," time samples and ",str(data.shape[1])," EEG channels for one epoch are taken. ",\
          "Total epoch number is ",str(data.shape[0])," and there are ",str(len(my_cfg.selected_subjects))," subjects included.\n",\
          "There are in total ", str(my_cfg.nClasses)," classes for classification.\n",flush=True)

    # -offset_to_subtract -> Classes made matching to CX definition
    labels = epoched.events[:, event_current_class_column]

    # Split data in train test and validation set. Stratify makes sure the label distribution is the same
    temp_data, test_data, temp_labels, test_labels = train_test_split(data, labels, test_size=my_cfg.test_split,
                                                                      shuffle=True, stratify=labels)

    train_data, val_data, train_labels, val_labels = train_test_split(temp_data, temp_labels,
                                                                      test_size=my_cfg.validation_split, shuffle=True,
                                                                      stratify=temp_labels)
    
    
    # Do data augmentation of training data
    if my_cfg.augment_with_gauss_noise:
        train_data, train_labels = augment_with_gaussian_noise(train_data, train_labels, my_cfg.augment_std_gauss,
                                                               my_cfg.augmentation_factor)
    # Drop tiles randomly
    if my_cfg.dropOut:
        train_data = dropout_tiles(train_data,my_cfg)        
    
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
                                      my_cfg.normalize)  # TODO: Should also be list selectable like the optimizers
    val_ds = ChannelsVoltageDataset(val_data, val_labels, my_cfg.normalize)
    test_ds = ChannelsVoltageDataset(test_data, test_labels, my_cfg.normalize)

    # Define data loader
    train_dl = DataLoader(train_ds, my_cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, my_cfg.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, my_cfg.batch_size, shuffle=False)
    

    return train_dl, val_dl, test_dl, input_dimension_, output_dimension_


def normalize(data):
    print("Normalizing data...",flush=True)
    
    #mean = np.mean(data,axis=(1,2)).reshape(-1,1,1)
    #std = np.std(data,axis=(1,2)).reshape(-1,1,1)
    #data = np.divide( data - np.tile(mean,(data.shape[1],data.shape[2])) , np.tile(std,(data.shape[1],data.shape[2])) )
    
    for i in range(data.shape[0]):
        data[i,:,:] = ( data[i,:,:] - np.mean(data[i,:,:]) ) / np.std( data[i,:,:] )
        data[i,:,:] = (data[i,:,:]+1)*0.5
        
    print("...data was normalized.\n",flush=True)
    return data
    
def dropout_tiles(data,config):
    
    perc = config.dropOutTilePerc
    
    sTimeTile = config.dropOutTimeTile
    sChannelTile = config.dropOutChannelTile
    
    nEpochs = data.shape[0]
    nChannel = data.shape[1]
    nTime = data.shape[2]
    
    if config.dropOutChOnly == True:
        sTimeTile = nTime
        
    if config.dropOutTimeOnly == True:
        sChannelTile = nChannel
    
    nTimeTile = int(np.ceil(nTime/sTimeTile))
    nChannelTile = int(np.ceil(nChannel/sChannelTile))
    
    print("Data is being dropped with ", str(perc), "prob and ", str(sTimeTile), " sized time tiles(",\
          str(nTimeTile),") and ", str(sChannelTile)," sized channel tiles(",str(nChannelTile),").", flush=True)
    
    drop = np.random.choice([0, 1], (nEpochs,nChannelTile,nTimeTile), p=[perc,(1-perc)])
    
    drop_mean = np.mean(np.mean(drop,axis=2),axis=1)
    zero_index = np.where(drop_mean == 0)

    for idx in zero_index[0]:
        if nTimeTile*nChannelTile > 1:
            fill_i = np.random.randint(0, nTimeTile*nChannelTile-1)
        else:
            fill_i = int(0)
        fill_ch = int(fill_i / nTimeTile)
        fill_time = int(fill_i % nTimeTile)
        drop[idx,fill_ch,fill_time] = 1
    
    drop =  np.repeat(drop, sTimeTile, axis=2)
    drop =  np.repeat(drop, sChannelTile, axis=1)
    
    drop = drop[:,:data.shape[1],:data.shape[2]]
    data = np.multiply(data,drop)
    
    print("...data was being dropped.", flush=True)
    return data
    

def augment_with_gaussian_noise(data, labels, std, multiplier):
    
    print("Data is being augmented with gaussian noise...",flush=True)
    mean = 0
    augmented_data = []
    augmented_labels = []
    if std > 1:
        raise ValueError(' We expect in the range 0 to 1')

    for idx, tmp_data in tqdm(enumerate(data),total=len(data)):
        if idx % 100 == 0:
            pass
            #print('Augmented ', idx, 'of', len(data))
        for j in range(multiplier):
            tmp_label = labels[idx]
            if j == 0:  # Take the real data for once
                augmented_data.append(tmp_data)
                augmented_labels.append(tmp_label)
            else:
                tmp_std_data = np.std(tmp_data)
                tmp_std = tmp_std_data*std

                noise = np.random.normal(loc=mean, scale=tmp_std, size=np.shape(tmp_data))
                tmp_data_noisy = np.add(tmp_data, noise)
                augmented_data.append(tmp_data_noisy)
                augmented_labels.append(tmp_label)
    augmented_data = np.asarray(augmented_data, dtype=np.float64)
    augmented_labels = np.asarray(augmented_labels, dtype=np.int32)
    print("...augmentation with gaussian noise is finished. \n",flush=True)

    return augmented_data, augmented_labels

def get_epoched_data(my_cfg):
    # Experimental runs per subject (range from 1 to 14). Runs differ in tasks performed tasks!
    # -> We want to split up the dataset in all classes there are

    #arr_runs = np.array([1, 2, [3, 7, 11], [4, 8, 12], [5, 9, 13], [6, 10, 14]])
    #arr_selected_classes = np.array([1, 1, [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]])
    #arr_labels_offsets = np.array([1, 0, 0, -2, -4, -6])
    
    #***** WITHOUT BASELINE***************
    #arr_runs = np.array([[3, 7, 11], [4, 8, 12], [5, 9, 13], [6, 10, 14]])
    #arr_selected_classes = np.array([[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]])
    #arr_labels_offsets = np.array([-2, 0, 2, -4]) 
    
    #***** 4 CLASSES ***************
    #arr_runs = np.array([ [4, 8, 12], [6, 10, 14]])
    #arr_runs = np.array([ [3, 7, 11], [5, 9, 13] ])
    #arr_selected_classes = np.array([[2, 3], [2, 3]])
    #arr_labels_offsets = np.array([0, 2]) 
    
    #***** 2 CLASSES ***************
    arr_runs = np.array([ [3, 7, 11]])
    arr_selected_classes = np.array([[2, 3]])
    arr_labels_offsets = np.array([2]) 

    # Load the data
    subjects = my_cfg.selected_subjects
    current_path = os.path.abspath(__file__)
    # print(current_path)
    if 'studi7/home/ProjectCode/' in current_path:
        data_path = '../../var/tmp/RawDataMNE'
        print('We are on the cluster...\n',flush=True)
        data_path = '../../var/tmp/RawDataMNE'
    else:
        print('We are not on the cluster...\n',flush=True)
        data_path = 'RawDataMNE'

        
    print("Data is being loaded using MNE...",flush=True)

    list_epochs = []
    for idx, runs in tqdm(enumerate(arr_runs),total=len(arr_runs)):
        tmp_classes = arr_selected_classes[idx]
        tmp_offset = arr_labels_offsets[idx]
        raw_EDF_list = []

        for subj in subjects:
            fileNames = eegbci.load_data(subj, runs, path=data_path)
            raw_EDF = [read_raw_edf(f, preload=True, stim_channel='auto', verbose='WARNING') for f in fileNames]
            raw_EDF_list.append(concatenate_raws(raw_EDF))

        raw = concatenate_raws(raw_EDF_list)

        # Pick the events
        events = find_events(raw, shortest_event=0, verbose=my_cfg.verbose)
        # Subtract the offset to make the label match
        events[:, 2] = events[:, 2] - tmp_offset
        tmp_classes = (tmp_classes - tmp_offset).tolist()
        # Extract the epochs
        tmp_epoched = Epochs(raw, events, event_id=tmp_classes, tmin=my_cfg.time_before_event_s,
                          tmax=my_cfg.time_after_event_s, baseline=None, picks=None,
                          preload=False, reject=None, flat=None, proj=True, decim=my_cfg.downSample, reject_tmin=None, reject_tmax=None,
                          detrend=None, on_missing='error', reject_by_annotation=True, metadata=None,
                          verbose=my_cfg.verbose)

        # Store epoch for later use
        list_epochs.append(tmp_epoched)
        'DEBUG'
        """SHOW DATA"""
        # Show some sample EEG data if desired
        if my_cfg.show_eeg_sample_plot:
            eeg_sample_plot(my_cfg.subjectIdx_to_plot, my_cfg.seconds_to_plot, my_cfg.channels_to_plot, raw_EDF_list)
        if my_cfg.show_events_distribution:
            events_distribution_plot(tmp_epoched.events)

    epoched = concatenate_epochs(list_epochs)


    """SHOW DATA"""
    # Show some sample EEG data if desired
    if my_cfg.show_eeg_sample_plot:
        eeg_sample_plot(my_cfg.subjectIdx_to_plot, my_cfg.seconds_to_plot, my_cfg.channels_to_plot, raw_EDF_list)
    if my_cfg.show_events_distribution:
        events_distribution_plot(epoched.events)

    print("...data loading with MNE was finished. \n")

    return epoched




'''
# https://github.com/dalinzhang/Cascade-Parallel/blob/master/data_preprocess/pre_process.py

	data_2D[0] = [ 	   	 0, 	   0,  	   	 0, 	   0, data[21], data[22], data[23], 	   0,  	     0, 	   0, 	 	 0]
	data_2D[1] = [	  	 0, 	   0,  	   	 0, data[24], data[25], data[26], data[27], data[28], 	   	 0,   	   0, 	 	 0]
	data_2D[2] = [	  	 0, data[29], data[30], data[31], data[32], data[33], data[34], data[35], data[36], data[37], 	 	 0]
	data_2D[3] = [	  	 0, data[38],  data[0],  data[1],  data[2],  data[3],  data[4],  data[5],  data[6], data[39], 		 0]
	data_2D[4] = [data[42], data[40],  data[7],  data[8],  data[9], data[10], data[11], data[12], data[13], data[41], data[43]]
	data_2D[5] = [	  	 0, data[44], data[14], data[15], data[16], data[17], data[18], data[19], data[20], data[45], 		 0]
	data_2D[6] = [	  	 0, data[46], data[47], data[48], data[49], data[50], data[51], data[52], data[53], data[54], 		 0]
	data_2D[7] = [	  	 0, 	   0, 	 	 0, data[55], data[56], data[57], data[58], data[59], 	   	 0, 	   0, 		 0]
	data_2D[8] = [	  	 0, 	   0, 	 	 0, 	   0, data[60], data[61], data[62], 	   0, 	   	 0, 	   0, 		 0]
	data_2D[9] = [	  	 0, 	   0, 	 	 0, 	   0, 	     0, data[63], 		 0, 	   0, 	   	 0, 	   0, 		 0]

'''
def data_1Dto2D(data, Y=10, X=11):
    
    data_2D = np.zeros([1, Y, X, data.shape[1] ])

    data_2D[0,0,4:7,:] = data[21:24,:]

    data_2D[0,1,3:8,:] = data[24:29,:]
    
    data_2D[0,2,1:10,:] = data[29:38,:]

    data_2D[0,3,1,:] = data[39,:]
    data_2D[0,3,2:9,:] = data[0:7,:]
    data_2D[0,3,9,:] = data[39,:]
    
    data_2D[0,4,0,:] = data[42,:]
    data_2D[0,4,1,:] = data[40,:]
    data_2D[0,4,2:9,:] = data[7:14,:]
    data_2D[0,4,9,:] = data[41,:]
    data_2D[0,4,10,:] = data[43,:]
    
    data_2D[0,5,1,:] = data[44,:]
    data_2D[0,5,2:9,:] = data[14:21,:]
    data_2D[0,5,9,:] = data[45,:]
    
    data_2D[0,6,1:10,:] = data[46:55,:]

    data_2D[0,7,3:8,:] = data[55:60,:]
    
    data_2D[0,8,4:7,:] = data[60:63,:]
    
    data_2D[0,8,5,:] = data[63,:]
    
    return data_2D


def dataset_1Dto2D(dataset_1D):
    print('Dataset is being made 2D...',flush=True)
    dataset_2D = np.zeros([dataset_1D.shape[0],10,11,dataset_1D.shape[2]])
    for i in range(dataset_1D.shape[0]):
        dataset_2D[i,:,:,:] = data_1Dto2D(dataset_1D[i,:,:])
    print("Data is now 2D.",flush=True)
    return dataset_2D