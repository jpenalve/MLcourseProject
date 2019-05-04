# MLcourseProject
This repository is for the Advanced Topics in ML project Spring Semester 2019

# Group members: 
Tim Fischer, Özhan Özen, Joaquin Penalver-Andres

# Project Description:
The project will focus on classification of movement imagination tasks using EEG signals. 

It is proven that thinking about different movement that one plans to execute, generates a different neural fooprint. This footprint can be detected by means of an EEG recording device. The goal of this project is to classify different movements that subjects may think of. We will base our project on an existing dataset (link: https://www.physionet.org/physiobank/database/eegmmidb/#experimental-protocol ). The methods and scope of the project are to be defined in the next weeks. 

==> Subjects 88, 92 and 100 have overlapping events. Please exclude these subjects.

We will classify the following:

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


Note: In case enough data cannot be found, we will focus on EEG artifact removal, instead.

# Steps
--------

1) Load the data.
2) Visualize the data (and understand the event info).
3) Normalize the data, and save it.
4) Write a data selection tool to allow selection of EEG-datasets by size(random) or specific indices.
5) Write Train / Validation / Test (TVT) split code.
6) Write DataLoaders and a preprocessing ("transforms") module. 
7) Create some simple CNNs. Make a module for each type of NN. (Make sure in- and output data is of correct shape)
8) Create a TVT module.
9) Visualize TVT results.
10) Organize the code (and folder structure) in a away, that TVT results with NN model and all hyperparameters
   are stored in a clean, comprehensible and repeatable manner.
	7.1) Make sure the code is as modular as possible. (NN, optimizers, regularization...)
11) Write a script that allows you to set up and execute various combinations of settings (NN, optimizers, preprocessing steps, and regularization...).
   (Modular structure necessary!)
12) --> Copy the code to the cluster and run first tests. Finalize the framework.

==> RESULT: Modular Framework which enables easy conductance of DL experiments on the EEG-datasets(~23.04.19)

Code How-To:
-------------
All parameters necessary for adapting the classification can be modified inside the config/myconfig.py files.

To store specific settings, just add a class to myconfig.py, inheriting from DefaultConfig.
(Currently DefaultConfig class is not final! ... so change there as well if you like)

Put your configs which shall be evaluated inside the list_of_configs in the myconfig.py module.
	
Inside the main.py: Select your config via myList = myconfig.list_of_configs

In case of supplementary optimizers or nn, please add them to the optimizers.py or neural_nets package. Adapt the optimizer_list or nn_list (+nn_models_getter.py) the  in the defaultconifg.py respecitvely.

Tensorboard:
------------
To run the TensorBoard, open a new terminal and run the command `$ tensorboard --logdir=./logs --port=6006`. Then, open http://localhost:6006/ on your web browser. If you have logs in your log dir, you will see nice graphs ;)

(logdir should point to the log directory of your created logs)

Logs can be created by calling write_logs_for_tensorboard() -> Feel free to modify, extend this function or add more tensorboard functions for further analysis of network performance.
# TO DOs:
---------
- Implement baseline reference as https://github.com/ChaiGuangJie/EEGFeedbackSystem/blob/master/classifier.py
- See the TODOs in the .py files (framework specific)

# Used Packages:

- MNE - Python

# Useful examples:

- https://martinos.org/mne/stable/auto_examples/decoding/plot_decoding_csp_eeg.html#sphx-glr-auto-examples-decoding-plot-decoding-csp-eeg-py
