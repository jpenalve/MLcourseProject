from configs.defaultconfig import DefaultConfig
# ==> Subjects 88, 89, 92 and 100 have overlapping events. Please exclude these subjects.
# ==> Make sure to pick enough subjects! Otherwise baseline has too few labels!


# Own configs follow here
class ConfigNo01(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 5
    # Give it a unique name and a brief description if you like
    config_name = 'simpleNN'
    selected_subjects = [1, 2, 3, 4, 5, 6, 7]
    config_remark = 'This is a simple NN test.. nothing serious'
    show_events_distribution = True
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0

class AS_EEGNet(DefaultConfig):
    num_of_epochs = 10
    config_name = 'EEGNet_AS_10_Epochs'
    config_remark = 'EEGNet_AS_10_Epochs'
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0

# Put them all in a list
list_of_configs = [ConfigNo01]


