# MLcourseProject - Deep Learning Your Brain
## Classification of Movement Execution and Imagination using EEG Signals
This repository is for the Advanced Topics in ML project Spring Semester 2019.

# Group members: 
Tim Fischer, Özhan Özen, Joaquin Penalver-Andres

# Project Description:
The project will focus on classification of movement imagination and movement tasks using EEG signals. 

It has been proven that the mental planning or execution of different movements, produces different neuronal footprints. These footprints can be detected by means of an EEG recording device. The goal of this project is to classify different movements that subjects may think of or execute. 

# Report
Please note that the report.html (which is basically the export of notebook) is under the folder ProjectCode. This readme file has almost the same content.

# DataSet
We will base our project on an existing dataset (link: https://www.physionet.org/physiobank/database/eegmmidb/#experimental-protocol ). 

The physionet dataset contains the following classes, with 109 subjects and 64 EGG Channels (Subjects 88, 92 and 100 have overlapping events. Please exclude these subjects).

LABELS	|	DESCRIPTION
------------------------------
0 	|	Baseline, eyes open           
1 	|	Baseline, eyes closed         
2 	|	Motor execution: Left Hand           
3 	|	Motor execution: Right Hand          
4 	|	Motor imagery: Left Hand           
5 	|	Motor imagery: Right Hand          
6 	|	Motor execution: Both Hands          
7 	|	Motor execution: Both Feet           
8 	|	Motor Im: Both Hands          
9 	|	Motor imagery: Both Feet           

We have chosen to include 8 classes (baselines are excluded) for classification. 

# Tools Used

- Pytorch (For NN training)
- MNE python (For downloading/loading eeg data and creating epochs)
- Tensorflow (for tensorboard visualization)
- Many other basic python modules for data processing
- The final test were done on a Ubuntu with two GPU


### Tensorboard
------------
To run the TensorBoard, open a new terminal, go to the ProjectCode folder and run the command `$ tensorboard --logdir=./logs --port=6006`. Then, open http://localhost:6006/ on your web browser. If you have logs in your log dir, you will see nice graphs ;)

(logdir should point to the log directory of your created logs)

Logs can be created by calling write_logs_for_tensorboard() -> Feel free to modify, extend this function or add more tensorboard functions for further analysis of network performance.


# Examples, literature or code that we were inspired:

- https://martinos.org/mne/stable/auto_examples/decoding/plot_decoding_csp_eeg.html#sphx-glr-auto-examples-decoding-plot-decoding-csp-eeg-py
- https://robintibor.github.io/braindecode/notebooks/Cropped_Decoding.html

# The preprocessing and Networks Trained:

### Pre-processing

- The epochs are taken to cover 2s on movement (trigger) offset with 160 samples per second. 
- Each epoch data are normalized to have zero mean and standard deviation of 1.
- Augmentation of data with gaussian noise and inpainting-like data removal for regularization are tried (both in time and channel axis), however, we could not detect significant difference.
- 20 subjects were included, with 8 classes (no baseline).
- Cropping the time axis in small windows (of 10 samples) was suggested in literature (Zhang,2018), we have followed the suggestion. Each network is trained with and without this technique.
- For 3D CNN, the 64 channels are mapped to their locations in the head as a 2D grid (11x10). In order to make this location information a rectangle, zeros are added where there is no electrode on the grid.
- Eventually the data shape were (nEpochs,nChannels,nSamples), (nEpochs,nChannelsX,nChannelsY,nSamples), (nEpochs*nWindows,nChannels,nSamples) or (nEpochs*nWindows,nChannelsX,nChannelsY,nSamples) depending on whether the time axis is cropped or not and whether it was 2D or 3D CNN.

### Networks Trained

1) 3D CNN
- Layer 1 -- (32x1x11x10x10--cropped) or (32x1x11x10x320--non_cropped) + ExpoRU (CELU) + BatchNorm
- Layer 2 -- (64x1x11x10x10--cropped) or (64x1x11x10x320--non_cropped) + ExpoRU (CELU) + BatchNorm
- Layer 3 -- (128x1x11x10x10--cropped) or (128x1x11x10x320--non_cropped) + ExpoRU (CELU) + BatchNorm
- Layer 4 -- (Flatten fully connected Linear, Droput=0.5) 

#### Note: All the kernel sizes were (3,3,3) with stride 1, and padding such that there were no reduction on the size.

2) 2D CNN 

- Layer 1 -- (32x1x64x10--cropped) or (32x1x64x320--non_cropped) + ExpoRU (CELU) + BatchNorm
- Layer 2 -- (64x1x64x10--cropped) or (64x1x64x320--non_cropped) + ExpoRU (CELU) + BatchNorm
- Layer 3 -- (128x1x64x10--cropped) or (128x1x64x320--non_cropped) + ExpoRU (CELU) + BatchNorm
- Layer 4 -- (Flatten fully connected Linear, Droput=0.5) 

#### Note: All the kernel sizes were (7,3) with stride 1, and padding such that there were no reduction on the size.

### For all the trainings, ADAM optimizer (lr:1e-3, weight decay:1e-4, scheduler with 20 steps and gamma 0.5), and Cross Entropy Loss (with sigmoid activation in the output) are used. Batch size was either 128 or 256.

# Results


The accuracy on the test sets for both CNNs with both cropped and non-cropped (of time axis) epochs are below. The in-class accuracy on the right is given for 3D-CNN cropped.
![](ProjectCode/Results/Figures/Tables.png)

As you see, cropping the time axis in time windows (of 10 samples) makes a huge difference
An explanation of Schirrmeister et al (2018) in arXiv:1703.05051v5 states that "cropping has the aim to force the ConvNet into using features that are presentin all crops of the trial, since the ConvNet can no longer use the differences between crops and the global  temporal structure." 

The training progress for accuracy and loss for cropped epochs are below.
![](ProjectCode/Results/Figures/cropped.png)

The training progress for accuracy and loss for non-cropped epochs are below. You could see the huge overfitting.
![](ProjectCode/Results/Figures/notcropped.png)

# Code: How-to
In order to run the project, you need to open the 'Main' jupyter notebook and run it from top to bottom.

All parameters necessary for adapting the classification can be modified inside the config/<myconfig.py> files.

To store specific settings, just add a class to myconfig.py, inheriting from DefaultConfig.

Put your configs which shall be evaluated inside the list_of_configs in the myconfig.py module.
	
Inside the main.py: Select your config via myList = myconfig.list_of_configs

In case of supplementary optimizers or nn, please add them to the optimizers.py or neural_nets package. Adapt the optimizer_list or nn_list (+nn_models_getter.py) the  in the defaultconifg.py respectively.

## Example config file to run any NN implemented/tested


#### Config class name to include in the list at the bottom.
class Config3DCNN_NOTCropped(DefaultConfig):
    verbose = 'CRITICAL'
    
    config_name = '3D CNN'
    config_remark = '3D CNN'
   
   #### Number of subjects.
    nSubj = 20 
    selected_subjects = selected_subjects[:nSubj]
    
   #### Selection of the network and optimizer, these names have to be in .py files.
    nn_list = ['ConvNet3D']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    optimizer_list = ['Adam']  # Extend if you want more. Add them in the optimizers.py module
    optimizer_selection_idx = 0  # Idx corresponds to entry optimizer_list (find below)
    learning_rate = 1e-3
    weight_decay = 1e-4
    
   #### Setting up a scheduler for learning rate.
    scheduler = True  
    schStepSize = 20
    schGamma = 0.5
    
   #### Number of epochs and early stopping settings.
    num_of_epochs = 50
    batch_size = 128
    use_early_stopping = True
    es_patience = num_of_epochs
    
   #### Normalization/augmentation settings
    normalize = True
    augment_with_gauss_noise = False
    augmentation_factor = 2
    augment_std_gauss = 0.2
    dropOut = False
    dropOutChOnly = False
    dropOutTimeOnly = False
    dropOutTilePerc = 0.5
    dropOutTimeTile = 40
    dropOutChannelTile = 8
    
   #### Epoch Settings
    time_before_event_s = 0.0  # Epochsize parameter: Start time before event.
    time_after_event_s = 2.0  # Epochsize parameter: Time after event.
    downSample = 1
    
   #### To make number of data points dividable with 10
    show_events_distribution = False
    removeLastData = True
    
   #### To make channel dimension 2d, or cropping the time axis.
    Elec2D = True
    wSize = 10
    wCropped = False
    
    
#### All the classes in this list will be trained.
list_of_configs = [Config3DCNN_Cropped]
