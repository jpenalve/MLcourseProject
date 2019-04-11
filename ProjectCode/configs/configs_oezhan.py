from configs.defaultconfig import DefaultConfig


# Own configs follow here
class ConfigNo01(DefaultConfig):
    num_of_epochs = 1  # Dummy


class ConfigNo02(DefaultConfig):
    num_of_epochs = 2  # Dummy


class ConfigNo03(DefaultConfig):
    num_of_epochs = 3  # Dummy


# Put them all in a list
list_of_configs = [ConfigNo01, ConfigNo02, ConfigNo03]