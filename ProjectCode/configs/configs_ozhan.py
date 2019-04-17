from configs.defaultconfig import DefaultConfig


# Own configs follow here
class ConfigNo01(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 25

    # Give it a unique name and a brief description if you like
    config_name = 'simpleNN'
    config_remark = 'This is a simple NN test.. nothing serious'
    selected_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

class ConfigNo02(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 1

    # Give it a unique name and a brief description if you like
    config_name = 'example1'
    config_remark = 'Another example here'

class ConfigNo03(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 1

    # Give it a unique name and a brief description if you like
    config_name = 'example2'
    config_remark = 'Super crazy network tested with normal settings'


# Put them all in a list
list_of_configs = [ConfigNo01]