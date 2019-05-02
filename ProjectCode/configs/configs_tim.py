from configs.defaultconfig import DefaultConfig
# ==> Subjects 88, 89, 92 and 100 have overlapping events. Please exclude these subjects.
# ==> Make sure to pick enough subjects! Otherwise baseline has too few labels!


# Own configs follow here
class ConfigNo01(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 10
    # Give it a unique name and a brief description if you like
    config_name = 'simpleNN'
    selected_subjects = [1, 2, 3, 4, 5, 6, 7]
    config_remark = 'This is a simple NN test.. nothing serious'
    show_events_distribution = True


# Put them all in a list
list_of_configs = [ConfigNo01]

