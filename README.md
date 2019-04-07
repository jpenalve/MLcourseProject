# MLcourseProject
This repository is for the Advanced Topics in ML project Spring Semester 2019

# Group members: 
Tim Fischer, Özhan Özen, Joaquin Penalver-Andres

# Project Description:
The project will focus on classification of movement imagination tasks using EEG signals. 

It is proven that thinking about different movement that one plans to execute, generates a different neural fooprint. This footprint can be detected by means of an EEG recording device. The goal of this project is to classify different movements that subjects may think of. We will base our project on an existing dataset (link: https://www.physionet.org/physiobank/database/eegmmidb/#experimental-protocol ). The methods and scope of the project are to be defined in the next weeks. 

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


# Used Packages:

- pyEDFlib
- MNE - Python

# Useful examples:

- https://martinos.org/mne/stable/auto_examples/decoding/plot_decoding_csp_eeg.html#sphx-glr-auto-examples-decoding-plot-decoding-csp-eeg-py
