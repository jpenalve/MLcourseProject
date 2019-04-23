from configs.defaultconfig import DefaultConfig
# ==> Subjects 88, 92 and 100 have overlapping events. Please exclude these subjects.


# Own configs follow here
class ConfigNo01(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 1

    # Give it a unique name and a brief description if you like
    config_name = 'simpleNN'
    config_remark = 'This is a simple NN test.. nothing serious'

class ConfigNo02(DefaultConfig):
    # Overwriting base class attributes
    num_of_epochs = 5
    """selected_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                        50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                        60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                        70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                        80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                        90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
                        100, 101, 102, 103, 104, 105, 106, 107, 108, 109]""" #88 is errorous??? 92 100
    selected_subjects = [1]
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
list_of_configs = [ConfigNo02]

