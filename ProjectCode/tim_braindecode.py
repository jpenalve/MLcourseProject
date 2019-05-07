from neural_nets.nn_models_getter import get_nn_model
from optimizers import get_optimizer
from utils_train import fit, test
from configs import configs_tim
from data_loader_creation import get_dataloader_objects
from classification_results import results_storer
from visualisations import plot_metrics_from_pkl

""" USER: SELECT THE CONFIGURATION YOU NEED """
myList = configs_tim.list_of_configs
# myList = configs_joaquin.list_of_configs
# myList = configs_ozhan.list_of_configs

for idx, my_cfg in enumerate(myList):
    print('++++ CONFIGURATION %2d, of %2d' % (idx, len(myList)))

    """ PREPARE DATALOADERS """
    # TODO: Write a method that checks if we have already stored the DL objects for this specific my_cfg -> LOAD THEM
    # TODO: If not -> STORE THEM (...We need a unique identifier for each DL object.. for example MD5 value)
    train_dl, val_dl, test_dl, input_dimension_, output_dimension_ = get_dataloader_objects(my_cfg)

    """CLASSIFICATION"""
    # allow logging
    import logging
    import importlib

    importlib.reload(logging)  # see https://stackoverflow.com/a/21475297/1469195
    log = logging.getLogger()
    log.setLevel('INFO')
    import sys

    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.INFO, stream=sys.stdout)


    from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
    from torch import nn
    from braindecode.torch_ext.util import set_random_seeds




    # Set if you want to use GPU
    # You can also use torch.cuda.is_available() to determine if cuda is available on your machine.
    cuda = False
    set_random_seeds(seed=20170629, cuda=cuda)
    n_classes = 10
    in_chans = train_dl.dataset.data.shape[1]
    # final_conv_length = auto ensures we only get a single output in the time dimension
    model = ShallowFBCSPNet(in_chans=in_chans, n_classes=n_classes,
                            input_time_length=train_dl.dataset.data.shape[2],
                            final_conv_length='auto')
    if cuda:
        model.cuda()

    from braindecode.torch_ext.optimizers import AdamW
    import torch.nn.functional as F
    #optimizer = AdamW(model.parameters(), lr=1*0.01, weight_decay=0.5*0.001) # these are good values for the deep model
    optimizer = AdamW(model.parameters(), lr=0.0625 * 0.01, weight_decay=0)
    model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1,)

    model.fit(train_dl.dataset.data.numpy(), train_dl.dataset.target.numpy(), epochs=1, batch_size=64, scheduler='cosine',
              validation_data=(val_dl.dataset.data.numpy().squeeze(), val_dl.dataset.target.numpy().squeeze()),)



    result_dict = model.evaluate(test_dl.dataset.data.numpy(), test_dl.dataset.target.numpy())
