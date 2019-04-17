import pickle
from datetime import datetime
import os


def store_results(my_cfg, model_trained, optimizer, test_loss, test_accuracy,
                  train_losses, train_accuracies, time_spent_for_training_s, val_losses, val_accuracies):

    # Create folder of todays date
    todays_date = datetime.today().strftime('%Y_%m_%d')
    date_path = 'classification_results/tmp_local/'
    if not os.path.exists(date_path):
        os.mkdir(date_path) 

    date_path = date_path + todays_date
    if not os.path.exists(date_path):
        os.mkdir(date_path) 

    file_name = my_cfg.config_name
    path_name = date_path + '/' + file_name + '.pkl'

    # If it exists, rename it
    while os.path.isfile(path_name):
        file_name += '_'
        path_name = date_path + '/' + file_name + '.pkl'

    with open(path_name, 'wb') as classification_results:
        pickle.dump([my_cfg, model_trained, optimizer, test_loss, test_accuracy,
                     train_losses, train_accuracies, time_spent_for_training_s,
                     val_losses, val_accuracies], classification_results)
