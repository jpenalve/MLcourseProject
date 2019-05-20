# Define the funcs for plotting or visualizing here
import matplotlib.pyplot as plt
import mne
import numpy as np
import matplotlib.pyplot as plt
import pickle


def eeg_sample_plot(subject, seconds_to_plot, channels_to_plot, raw):
    raw_for_plotshow = raw[subject]
    sfreq = raw_for_plotshow.info['sfreq']  # sample frequency
    timelength_s = int(seconds_to_plot * sfreq)
    
    ncols = 3
    nrows = int(np.ceil(channels_to_plot/ncols))
    plt.figure(figsize=(20, 4*nrows))
    
    # Plot n image
    for i in range(channels_to_plot):
        plt.subplot(nrows,ncols,i+1)
        plt.title("Channel #"+ str(i+1))
        data, times = raw_for_plotshow[i, :timelength_s]
        plt.plot(times, data.T)
        
    plt.figure()
    raw_for_plotshow.plot(n_channels=channels_to_plot, scalings='auto', title='Auto-scaled Data from arrays', show=True, block=False)


def events_distribution_plot(events):
    # Events
    mne.viz.plot_events(events, show=False)
    plt.title('Show event distribution over the merged dataset')
    plt.tight_layout()
    plt.show(block=True)
    """ Each annotation includes one of three codes (T0, T1, or T2):
    
        T0 corresponds to rest
        T1 corresponds to onset of motion (real or imagined) of
            the left fist (in runs 3, 4, 7, 8, 11, and 12)
            both fists (in runs 5, 6, 9, 10, 13, and 14)
        T2 corresponds to onset of motion (real or imagined) of
            the right fist (in runs 3, 4, 7, 8, 11, and 12)
            both feet (in runs 5, 6, 9, 10, 13, and 14)
    """

def write_accuracies_for_tensorboard(accuracy_list, step, model, logger):
    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #

    # 1. Log scalar values (scalar summary)
    for i in range(len(accuracy_list)):
        info = {'ClassAccuracy/Class '+str(i): accuracy_list[i]}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step + 1)
    
    
def write_logs_for_tensorboard(loss, accuracy, step, model, logger):
    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #

    # 1. Log scalar values (scalar summary)
    info = {'.Overall/Loss': loss, '.Overall/Accuracy': accuracy}

    for tag, value in info.items():
        logger.scalar_summary(tag, value, step + 1)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
        logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)

    # 3. Log training data (image summary)
    # info = {'data': data.view(-1, 28, 28)[:10].cpu().numpy()}

    # for tag, data in info.items():
    #    logger.image_summary(tag, data, iteration + 1)

def plot_metrics_from_pkl(pkl_file_path):
    #  Get all pkl files

    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    """ ORDER:
        [my_cfg, test_loss, test_accuracy, train_losses, train_accuracies, time_spent_for_training_s,
        val_losses, val_accuracies]
    """
    my_cfg = data[0]
    test_loss = data[1]
    test_accuracy = data[2]
    train_losses = data[3]
    train_accuracies = data[4]
    val_losses = data[6]
    val_accuracies = data[7]

    #  Plot them
    plot_performance_metrics(my_cfg, train_losses, val_losses, train_accuracies, val_accuracies,
                             test_accuracy, test_loss)


def plot_performance_metrics(my_cfg, train_losses, val_losses, train_accuracies, val_accuracies, test_acc, test_loss):

    plt.close('all')
    nn_name = my_cfg.config_name
    plt.figure()
    plt.plot(np.arange(len(train_losses)), train_losses)
    plt.plot(np.arange(len(val_losses)), val_losses)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss value')
    plt.title('Train/val loss of ' + nn_name + '(Tst loss: ' + str(test_loss)[0:5] + ')')
    plt.show(block=True)

    plt.figure()
    plt.plot(np.arange(len(train_accuracies)), train_accuracies)
    plt.plot(np.arange(len(val_accuracies)), val_accuracies)
    plt.legend(['train_acc', 'val_acc'])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Train/val accuracy of ' + nn_name + '(Tst acc: ' + str(test_acc)[0:5] + ')')
    plt.show(block=True)
    
    
    
def curve_name_gen(config):
    
    config.curve_name = config.config_name
    print("\n\n\n\n",config.curve_name,"\n-------------------------\n", flush=True)
