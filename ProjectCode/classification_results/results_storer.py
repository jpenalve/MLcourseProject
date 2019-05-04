import pickle
from datetime import datetime
import os
from utils_train import test

def store_results(my_cfg, model_trained, optimizer, test_loss, test_accuracy,
                  train_losses, train_accuracies, time_spent_for_training_s, val_losses, val_accuracies, test_dl):


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
    path_name_txt = date_path + '/' + file_name + '.txt'
    txt_file = open(path_name_txt, "a")

    # If it exists, rename it
    while os.path.isfile(path_name):
        file_name += '_'
        path_name = date_path + '/' + file_name + '.pkl'

    with open(path_name, 'wb') as classification_results:
        pickle.dump([my_cfg, model_trained, optimizer, test_loss, test_accuracy,
                     train_losses, train_accuracies, time_spent_for_training_s,
                     val_losses, val_accuracies], classification_results)

    # Write in text file some data (easy, fast access for evaluations)
    txt_file.write(my_cfg.config_remark + '\n')
    txt_file.write('->val_loss: {:.4f}, val_accuracy: {:.4f}% \n'.format(val_losses[-1], val_accuracies[-1]))
    txt_file.write('->train_loss: {:.4f}, train_accuracy: {:.4f}% \n'.format(train_losses[-1], train_accuracies[-1]))

    # Write detailled test loss metrics
    test_loss, test_accuracy = test(model_trained, test_dl, my_cfg.loss_fn, print_loss=True, write_class_txt=True,
                                    txt_file_handle=txt_file)
    txt_file.write('->test_loss: {:.4f}, test_accuracy: {:.4f}% \n'.format(test_loss, test_accuracy))


    txt_file.close()