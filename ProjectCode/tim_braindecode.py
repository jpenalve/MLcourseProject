
from configs import configs_tim
from data_loader_creation import get_dataloader_objects
from optimizers import get_optimizer
import torch
import torch.nn.functional as F
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

if torch.cuda.is_available():
    cuda = True
else:
    cuda = False
for idx, my_cfg in enumerate(myList):
    print('++++ CONFIGURATION %2d, of %2d' % (idx, len(myList)))

    """ PREPARE DATALOADERS """
    # TODO: Write a method that checks if we have already stored the DL objects for this specific my_cfg -> LOAD THEM
    # TODO: If not -> STORE THEM (...We need a unique identifier for each DL object.. for example MD5 value)
    train_dl, val_dl, test_dl, input_dimension_, output_dimension_ = get_dataloader_objects(my_cfg)

    """CLASSIFICATION"""
    importlib.reload(logging)  # see https://stackoverflow.com/a/21475297/1469195
    log = logging.getLogger()
    log.setLevel('INFO')
    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.INFO, stream=sys.stdout)


    # Set if you want to use GPU.
    set_random_seeds(seed=20170629, cuda=cuda)
    n_classes = 10
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
    cropped = False
    n_epochs = 1

    if cropped:
        """For cropped decoding, we now transform the model into a model that outputs a dense time series of 
        predictions. For this, we manually set the length of the final convolution layer to some length that makes the 
        receptive field of the ConvNet smaller than the number of samples in a trial (see final_conv_length=12 in the
        model definition)."""
        final_conv_length_param = 12
        model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                                input_time_length=None,
                                final_conv_length=final_conv_length_param)
    else:
        model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                                input_time_length=train_dl.dataset.data.shape[2],
                                final_conv_length='auto')

        model = Deep4Net(in_chans=in_chans, n_classes=n_classes,
                         input_time_length=train_dl.dataset.data.shape[2],
                         final_conv_length='auto')
        """
        model = Conv2dWithConstraint(in_channels=in_chans,kernel_size=5, out_channels=output_dimension_)
        model = EEGNet(in_chans=in_chans, n_classes=n_classes,
                         input_time_length=train_dl.dataset.data.shape[2],
                         final_conv_length='auto')
        model = EEGNetv4(in_chans=in_chans, n_classes=n_classes,
                         input_time_length=train_dl.dataset.data.shape[2],
                         final_conv_length='auto')
        model = HybridNetModule(in_chans=in_chans, n_classes=n_classes,
                         input_time_length=train_dl.dataset.data.shape[2])
"""




    if cuda:
        model.cuda()

    optimizer = get_optimizer('AdamW', learning_rate=0.0625 * 0.01, model_parameters=model.parameters(), sgd_momentum=0,
                              weight_decay_factor=0)
    model.compile(loss=F.cross_entropy, optimizer=optimizer, iterator_seed=1, cropped=cropped)
    if cropped:
        super_crop_size = np.round(train_dl.dataset.data.shape[2]/5)
        model.fit(train_dl.dataset.data.numpy(), train_dl.dataset.target.numpy(), epochs=n_epochs, batch_size=64,
                  scheduler='cosine', input_time_length=450, validation_data=(val_dl.dataset.data.numpy().squeeze(),
                  val_dl.dataset.target.numpy().squeeze()))
    else:

        model.fit(train_dl.dataset.data.numpy(), train_dl.dataset.target.numpy(), epochs=n_epochs, batch_size=64,
                  scheduler='cosine', validation_data=(val_dl.dataset.data.numpy().squeeze(),
                                                       val_dl.dataset.target.numpy().squeeze()))

    time_spent_for_training_s = np.round(np.max(model.epochs_df.runtime.tolist()) / 60)
    train_losses = model.epochs_df.train_loss.tolist()
    train_accuracies = model.epochs_df.train_misclass.tolist()

    val_losses = model.epochs_df.valid_loss.tolist()
    val_accuracies = model.epochs_df.valid_misclass.tolist()

    result_dict = model.evaluate(test_dl.dataset.data.numpy(), test_dl.dataset.target.numpy())
    test_loss = result_dict["loss"]
    test_accuracy = 1 - result_dict["misclass"]

    # Store the results

    results_storer.store_results_for_plot(my_cfg,test_loss, test_accuracy, train_losses,
                                 train_accuracies, time_spent_for_training_s, val_losses, val_accuracies)