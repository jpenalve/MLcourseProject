from configs.defaultconfig import DefaultConfig
import torch.nn as nn
# ==> Subjects 88, 89, 92 and 100 have overlapping events. Please exclude these subjects.
# ==> Make sure to pick enough subjects! Otherwise baseline has too few labels!

""" 
Parameters to play with:
DEFAULT IS:
normalize = True  # Epoch normalization to mean=0.5, std=0.5

augment_with_gauss_noise = True
augment_std_gauss = 0.2  # (See EEG Review Roy et. al. 2019)
augmentation_factor = 10

learning_rate = 0.001
weight_decay = 0.000075

nn_selection_idx = 1
nn_list = ['SimpleFC', 'DeepFC', 'EEGNet',
           'ConvNet01']  # Extend if you want more. Add them in the nn_models_getter.py module
dropout_perc = 0.25
scheduler = None  # torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)
"""
class DummyConfig(DefaultConfig):
    config_name = 'aaaaaDummyConfig'
    config_remark = 'DummyConfig'
    num_of_epochs = 1
    selected_subjects = [1, 2, 3, 4, 5, 6, 7]
    show_events_distribution = True
    augment_with_gauss_noise = False
    normalize = False  # Epoch normalization to mean=0.5, std=0.5
    nn_list = ['SimpleFC01']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    #time_before_event_s = -0.1  # Epochsize parameter: Start time before event.
    #time_after_event_s = 2.0  # Epochsize parameter: Time after event.

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Fully Connected START


class SimpleFC01(DefaultConfig):
    config_name = 'SimpleFC01'
    config_remark = 'SimpleFC: Nor mormalization'
    normalize = False  # Epoch normalization to mean=0.5, std=0.5
    nn_list = ['SimpleFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class SimpleFC02(DefaultConfig):
    config_name = 'SimpleFC02'
    config_remark = 'SimpleFC: No mormalization and no gauss'
    normalize = False  # Epoch normalization to mean=0.5, std=0.5
    augment_with_gauss_noise = False
    nn_list = ['SimpleFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class SimpleFC03(DefaultConfig):
    config_name = 'SimpleFC03'
    config_remark = 'SimpleFC: No  no gauss'
    normalize = True  # Epoch normalization to mean=0.5, std=0.5
    augment_with_gauss_noise = False
    nn_list = ['SimpleFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class SimpleFC04(DefaultConfig):
    config_name = 'SimpleFC04'
    config_remark = 'SimpleFC: Higher Learning rate with scheduler'
    scheduler = True
    learning_rate = 0.01
    nn_list = ['SimpleFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class SimpleFC05(DefaultConfig):
    config_name = 'SimpleFC05'
    config_remark = 'SimpleFC: Higher weight decay: weight_decay = 0.0075 (default 0.000075)'
    weight_decay = 0.0075
    nn_list = ['SimpleFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class SimpleFC06(DefaultConfig):
    config_name = 'SimpleFC06'
    config_remark = 'SimpleFC: No weight decay (default 0.000075)'
    weight_decay = 0
    nn_list = ['SimpleFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class DeepFC01(DefaultConfig):
    config_name = 'DeepFC01'
    config_remark = 'DeepFC: Nor mormalization'
    normalize = False  # Epoch normalization to mean=0.5, std=0.5
    nn_list = ['DeepFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class DeepFC02(DefaultConfig):
    config_name = 'DeepFC02'
    config_remark = 'DeepFC: No mormalization and no gauss'
    normalize = False  # Epoch normalization to mean=0.5, std=0.5
    augment_with_gauss_noise = False
    nn_list = ['DeepFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class DeepFC03(DefaultConfig):
    config_name = 'DeepFC03'
    config_remark = 'DeepFC: No  no gauss'
    normalize = True  # Epoch normalization to mean=0.5, std=0.5
    augment_with_gauss_noise = False
    nn_list = ['DeepFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class DeepFC04(DefaultConfig):
    config_name = 'DeepFC04'
    config_remark = 'DeepFC: Higher Learning rate with scheduler'
    scheduler = True
    learning_rate = 0.01
    nn_list = ['DeepFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class DeepFC05(DefaultConfig):
    config_name = 'DeepFC05'
    config_remark = 'DeepFC: Higher weight decay: weight_decay = 0.0075 (default 0.000075)'
    weight_decay = 0.0075
    nn_list = ['DeepFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class DeepFC06(DefaultConfig):
    config_name = 'DeepFC06'
    config_remark = 'DeepFC: No weight decay (default 0.000075)'
    weight_decay = 0
    nn_list = ['DeepFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0

# 06.05
class DeepFC07(DefaultConfig):
    config_name = 'DeepFC07'
    config_remark = 'DeepFC: Higher weight decay: weight_decay = 0.0075 (default 0.000075)'
    weight_decay = 0.00075
    nn_list = ['DeepFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0

class DeepFC08(DefaultConfig):
    config_name = 'DeepFC08'
    config_remark = 'DeepFC: Higher weight decay: weight_decay = 0.0075 (default 0.000075)'
    dropout_perc = 0.2
    nn_list = ['DeepFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0

class DeepFC09(DefaultConfig):
    config_name = 'DeepFC09'
    config_remark = ''
    weight_decay = 0.0
    dropout_perc = 0.2
    nn_list = ['DeepFC']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0

# Fully Connected END
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# EEGNet Tests START


class EEGNet01(DefaultConfig):
    config_name = 'EEGNet01'
    config_remark = 'EEGNET: Nor mormalization'
    normalize = False  # Epoch normalization to mean=0.5, std=0.5
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class EEGNet02(DefaultConfig):
    config_name = 'EEGNet02'
    config_remark = 'EEGNET: No mormalization and no gauss'
    normalize = False  # Epoch normalization to mean=0.5, std=0.5
    augment_with_gauss_noise = False
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class EEGNet03(DefaultConfig):
    config_name = 'EEGNet03'
    config_remark = 'EEGNET: No  no gauss'
    normalize = True  # Epoch normalization to mean=0.5, std=0.5
    augment_with_gauss_noise = False
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class EEGNet04(DefaultConfig):
    config_name = 'EEGNet04'
    config_remark = 'EEGNET: Higher Learning rate with scheduler'
    scheduler = True
    learning_rate = 0.01
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class EEGNet05(DefaultConfig):
    config_name = 'EEGNet05'
    config_remark = 'EEGNET: Higher weight decay: weight_decay = 0.0075 (default 0.000075)'
    weight_decay = 0.0075
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class EEGNet06(DefaultConfig):
    config_name = 'EEGNet06'
    config_remark = 'EEGNET: No weight decay (default 0.000075)'
    weight_decay = 0
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


# Same with higher droput (default of EEG NET= 0.25)


class EEGNet07(DefaultConfig):
    config_name = 'EEGNet07'
    config_remark = 'EEGNET: Nor mormalization'
    dropout_perc = 0.5
    normalize = False  # Epoch normalization to mean=0.5, std=0.5
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class EEGNet08(DefaultConfig):
    config_name = 'EEGNet08'
    config_remark = 'EEGNET: dropout = 0.5; No mormalization and no gauss'
    dropout_perc = 0.5
    normalize = False  # Epoch normalization to mean=0.5, std=0.5
    augment_with_gauss_noise = False
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class EEGNet09(DefaultConfig):
    config_name = 'EEGNet09'
    config_remark = 'EEGNET: dropout = 0.5; No  no gauss'
    dropout_perc = 0.5
    normalize = True  # Epoch normalization to mean=0.5, std=0.5
    augment_with_gauss_noise = False
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class EEGNet10(DefaultConfig):
    config_name = 'EEGNet10'
    config_remark = 'EEGNET: dropout = 0.5; Higher Learning rate with scheduler'
    dropout_perc = 0.5
    scheduler = True
    learning_rate = 0.01
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class EEGNet11(DefaultConfig):
    config_name = 'EEGNet11'
    config_remark = 'EEGNET: dropout = 0.5; Higher weight decay: weight_decay = 0.0075 (default 0.000075)'
    dropout_perc = 0.5
    weight_decay = 0.0075
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0


class EEGNet12(DefaultConfig):
    config_name = 'EEGNet12'
    config_remark = 'EEGNET: dropout = 0.5; No weight decay (default 0.000075)'
    dropout_perc = 0.5
    weight_decay = 0
    nn_list = ['EEGNet']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0

# DEEPER EEG NET
class EEGNetDeeper01(DefaultConfig):
    config_name = 'EEGNetDeeper01'
    config_remark = 'EEGNetDeeper: Like EEGNet11 but with more layers'
    dropout_perc = 0.1
    weight_decay = 0.00015
    scheduler = True
    nn_list = ['EEGNetDeeper']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    augment_with_gauss_noise = True  # DEBUG TEMP! eigentlich true


class EEGNetDeeper02(DefaultConfig):
    config_name = 'EEGNetDeeper02'
    config_remark = 'EEGNetDeeper: Like EEGNet11 but with more layers'
    dropout_perc = 0.25
    #weight_decay = 0.00015
    scheduler = True
    nn_list = ['EEGNetDeeper']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    augment_with_gauss_noise = False  # DEBUG TEMP! eigentlich true


class EEGNetDeeper03(DefaultConfig):
    config_name = 'EEGNetDeeper03'
    config_remark = 'EEGNetDeeper: Like EEGNet11 but with more layers'
    dropout_perc = 0.2
    weight_decay = 0.00015
    scheduler = True
    nn_list = ['EEGNetDeeper']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    augment_with_gauss_noise = False  # DEBUG TEMP! eigentlich true


class EEGNetDeeper04(DefaultConfig):
    config_name = 'EEGNetDeeper04'
    config_remark = 'EEGNetDeeper: Like EEGNet11 but with more layers'
    dropout_perc = 0.5
    #weight_decay = 0.00015
    scheduler = True
    nn_list = ['EEGNetDeeper']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    augment_with_gauss_noise = True  # DEBUG TEMP! eigentlich true


class EEGNetDeeper05(DefaultConfig):
    config_name = 'EEGNetDeeper05'
    config_remark = 'EEGNetDeeper: Like EEGNet11 but with more layers'
    dropout_perc = 0.5
    weight_decay = 0.00015
    scheduler = False
    nn_list = ['EEGNetDeeper']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    augment_with_gauss_noise = True  # DEBUG TEMP! eigentlich true


class EEGNetDeeper06(DefaultConfig):
    config_name = 'EEGNetDeeper06'
    config_remark = 'EEGNetDeeper: Like EEGNet11 but with more layers'
    dropout_perc = 0.5
    weight_decay = 0.0
    scheduler = False
    nn_list = ['EEGNetDeeper']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    augment_with_gauss_noise = False  # DEBUG TEMP! eigentlich true


class EEGNetDeeper07(DefaultConfig):
    config_name = 'EEGNetDeeper06'
    config_remark = 'EEGNetDeeper: Like EEGNet11 but with more layers'
    dropout_perc = 0.1
    weight_decay = 0.0
    scheduler = False
    nn_list = ['EEGNetDeeper']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    augment_with_gauss_noise = False  # DEBUG TEMP! eigentlich true

# EEGNet Tests END
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Make the EEGNet deeper (looks like we need more capacity) 3 more layers
class EEGNetDeeper(DefaultConfig):
    config_name = 'EEGNetDeeper'
    config_remark = 'EEGNetDeeper: Like EEGNet11 but with more layers'
    dropout_perc = 0.1
    weight_decay = 0.00015
    scheduler = True
    nn_list = ['EEGNetDeeper']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    augment_with_gauss_noise = False  # DEBUG TEMP! eigentlich true


class EEGNet11Dbg(DefaultConfig):
    config_name = 'EEGNet11'
    config_remark = 'EEGNET: dropout = 0.5; Higher weight decay: weight_decay = 0.0075 (default 0.000075)'

    dropout_perc = 0.1
    nn_list = ['EEGNetDeeper']  # Extend if you want more. Add them in the nn_models_getter.py module
    nn_selection_idx = 0
    weight_decay = 0.00015
    scheduler = False
    augment_with_gauss_noise = False  # DEBUG TEMP! eigentlich true
    # loss_fn = nn.NLLLoss()
    # num_of_epochs = 1
    #selected_subjects = [1, 2, 3, 4, 5, 6, 7]# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Put them all in a list
#list_of_configs = [DummyConfig]

#list_of_configs = [EEGNet06, EEGNet07, EEGNet08, EEGNet09, EEGNet10, EEGNet11, EEGNet12, EEGNetDeeper01,
                   #EEGNetDeeper02, EEGNetDeeper03, EEGNetDeeper04, EEGNetDeeper05, EEGNetDeeper06, EEGNetDeeper07,
                  # DeepFC07, DeepFC08, DeepFC09]

#CONFIGS to create for eval plots
list_of_configs = [EEGNet06, EEGNet10, EEGNet11, EEGNet12,
                   EEGNetDeeper01, EEGNetDeeper03, EEGNetDeeper07,
                   SimpleFC01, SimpleFC02, SimpleFC05, SimpleFC06,
                   DeepFC01 , DeepFC05, DeepFC06]

"""
ALL CONFIGS
list_of_configs = [EEGNet01, EEGNet02, EEGNet03, EEGNet04, EEGNet05, EEGNet06,
                   EEGNet07, EEGNet08, EEGNet09, EEGNet10, EEGNet11, EEGNet12,
                   EEGNetDeeper01,EEGNetDeeper02, EEGNetDeeper03, EEGNetDeeper04, EEGNetDeeper05, 
                   EEGNetDeeper06, EEGNetDeeper07,
                   SimpleFC01, SimpleFC02, SimpleFC03, SimpleFC04, SimpleFC05, SimpleFC06,
                   DeepFC01, DeepFC02, DeepFC03, DeepFC04, DeepFC05, DeepFC06]
"""
