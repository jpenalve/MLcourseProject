
"""
Experiments Results: The best ist the ConvNet ShallowFBCSPNet
-----------------------------------------

Test it with different strategies: (20 Epochs enough for tendency)

learning_rate=0.0625 * 0.01, weight_decay_factor=0

- Without cropped training DONE
- With gaussian augmentation (No Crop) DONE
- Without gaussian augmentation
- With cropped training

Take the best model (With gauss and cropped training) and go on with
- Cosine Annealing learning_rate

If better: Take cosine annealing.

...Go on with:

L2 regularization
weight_decay_factor = 0.0001
weight_decay_factor = 0.00001
weight_decay_factor = 0.000001

=> Take the best value.
Apply early stopping.
"""
from configs import configs_tim, defaultconfig
from data_loader_creation import get_dataloader_objects
from optimizers import get_optimizer
import torch
import torch.nn.functional as F
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
""" USER: SELECT THE CONFIGURATION YOU NEED """
myList = configs_tim.list_of_configs
# myList = configs_joaquin.list_of_configs
# myList = configs_ozhan.list_of_configs
import logging
import importlib
import sys
import numpy as np
from braindecode.torch_ext.util import set_random_seeds
from classification_results import results_storer
from utils_train import write_class_accuracies_to_txt
if torch.cuda.is_available():
    cuda = True
else:
    cuda = False
n_classes = 10
#list_of_models = ['EEGNetv4', 'ShallowFBCSPNet', 'Deep4Net'] # We know due to experiments that Shallow FBCSPNet is best
list_of_models = ['ShallowFBCSPNet']
my_cfg = defaultconfig.DefaultConfig
my_cfg.augment_with_gauss_noise = False
start_idx = 0
cropped = False
""" PREPARE DATALOADERS """
# TODO: Write a method that checks if we have already stored the DL objects for this specific my_cfg -> LOAD THEM
# TODO: If not -> STORE THEM (...We need a unique identifier for each DL object.. for example MD5 value)


train_dl, val_dl, test_dl, input_dimension_, output_dimension_ = get_dataloader_objects(my_cfg)


while start_idx < len(list_of_models):
    print('++++ CONFIGURATION %2d, of %2d' % (start_idx+1, len(list_of_models)))
    print('croppend is ', cropped)
    tmp_model_id = list_of_models[start_idx]

    """CLASSIFICATION"""
    importlib.reload(logging)  # see https://stackoverflow.com/a/21475297/1469195
    log = logging.getLogger()
    log.setLevel('INFO')
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.INFO, stream=sys.stdout)

    # Set if you want to use GPU.
    set_random_seeds(seed=20170629, cuda=cuda)
    in_chans = train_dl.dataset.data.shape[1]
    # Enable Logging
    importlib.reload(logging)  # see https://stackoverflow.com/a/21475297/1469195
    log = logging.getLogger()
    log.setLevel('INFO')
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.INFO, stream=sys.stdout)
    from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
    from braindecode.models.deep4 import Deep4Net
    from braindecode.models.eegnet import Conv2dWithConstraint, EEGNet, EEGNetv4
    from braindecode.models.hybrid import HybridNetModule

    """ BUILD THE MODEL"""

    if cropped:
        """For cropped decoding, we now transform the model into a model that outputs a dense time series of 
        predictions. For this, we manually set the length of the final convolution layer to some length that makes the 
        receptive field of the ConvNet smaller than the number of samples in a trial (see final_conv_length=12 in the
        model definition)."""
        final_conv_length_param = 12
        input_time_length_param = None
        str_addon = '_cropped'
    else:
        input_time_length_param= train_dl.dataset.data.shape[2]
        str_addon = '_Not_cropped'
        final_conv_length_param = 'auto'
    if tmp_model_id == 'ShallowFBCSPNet':
        my_cfg.config_name = 'ShallowFBCSPNet'+str_addon
        my_cfg.config_remark = str_addon
        model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                                input_time_length=input_time_length_param,
                                final_conv_length=final_conv_length_param)
        super_crop_size = 450
        if cuda: model.cuda()
    elif tmp_model_id == 'Deep4Net':
        my_cfg.config_name = 'Deep4Net'+str_addon
        my_cfg.config_remark = str_addon
        model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                         input_time_length=input_time_length_param,
                         final_conv_length=final_conv_length_param)
        #super_crop_size = 450
        if cuda: model.cuda()
    elif tmp_model_id == 'EEGNetv4':
        my_cfg.config_name = 'EEGNetv4'+str_addon
        my_cfg.config_remark = str_addon
        model = EEGNetv4(in_chans=in_chans, n_classes=n_classes,
                             input_time_length=input_time_length_param,
                             final_conv_length=final_conv_length_param)
        #super_crop_size = 450
        if cuda: model.cuda()


    optimizer = get_optimizer('AdamW', learning_rate=0.0625 * 0.01, model_parameters=model.parameters(), sgd_momentum=0,
                              weight_decay_factor=0)

    model.compile(loss=F.cross_entropy, optimizer=optimizer, iterator_seed=1, cropped=cropped)
    if cropped:
        model.fit(train_dl.dataset.data.numpy(), train_dl.dataset.target.numpy(), epochs=my_cfg.num_of_epochs,
                  batch_size=my_cfg.batch_size, scheduler='cosine', input_time_length=super_crop_size,
                  validation_data=(val_dl.dataset.data.numpy(), val_dl.dataset.target.numpy()))
    else:
        model.fit(train_dl.dataset.data.numpy(), train_dl.dataset.target.numpy(), epochs=my_cfg.num_of_epochs,
                  batch_size=my_cfg.batch_size, scheduler='cosine',
                  validation_data=(val_dl.dataset.data.numpy(), val_dl.dataset.target.numpy()))

    time_spent_for_training_s = np.round(np.sum(model.epochs_df.runtime.tolist()))
    train_losses = model.epochs_df.train_loss.tolist()
    train_accuracies = model.epochs_df.train_misclass.tolist()

    val_losses = model.epochs_df.valid_loss.tolist()
    val_accuracies = model.epochs_df.valid_misclass.tolist()
    result_dict = model.evaluate(test_dl.dataset.data.numpy(), test_dl.dataset.target.numpy())
    test_loss = result_dict["loss"]
    test_accuracy = 1 - result_dict["misclass"]
    if cuda:  # Works only with cuda... strange...
        # Class specific
        class_correct = list(0. for i in range(n_classes))
        class_appearances = list(0. for i in range(n_classes))
        class_predictions= (model.predict_classes(test_dl.dataset.data.numpy())).tolist()
        for i in range(len(class_predictions)):
            tmp_label = test_dl.dataset.target.numpy()[i]
            if tmp_label == class_predictions[i]:
                class_correct[tmp_label] += 1
            class_appearances[tmp_label] += 1

    # Store the results
    txt_file_path = results_storer.store_results_for_plot(my_cfg, test_loss, test_accuracy, train_losses,
                                          train_accuracies, time_spent_for_training_s, val_losses, val_accuracies)
    if cuda:
        write_class_accuracies_to_txt(txt_file_path, n_classes, class_appearances, class_correct, print_enabled=True)

    start_idx += 1  # Iterator

    if not start_idx < len(list_of_models):
        print('ONE LIST RUN IS FINNISHED')
        if not cropped: # We are done
            print('Done with main braindecode')
        else:
            cropped = False
            start_idx = 0 # we do only one setting
