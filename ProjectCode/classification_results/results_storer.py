import pickle


def store_results(my_cfg, model_trained, optimizer, test_loss, test_accuracy,
                  train_losses, train_accuracies, val_losses, val_accuracies):
    file_name = my_cfg.config_name
    path_name = 'classification_results/' + file_name
    with open(path_name, 'wb') as classification_results:
        pickle.dump([my_cfg, model_trained, optimizer, test_loss, test_accuracy,
                     train_losses, train_accuracies, val_losses, val_accuracies],
                    classification_results)
