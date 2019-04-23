from visualisations import eeg_sample_plot, events_distribution_plot


from neural_nets.nn_models_getter import get_nn_model
from optimizers import get_optimizer
from utils_train import fit, test
from configs import configs_tim

from classification_results import results_storer

""" USER: SELECT THE CONFIGURATION YOU NEED """
myList = configs_tim.list_of_configs
#myList = configs_joaquin.list_of_configs
#myList = configs_oezhan.list_of_configs
 
for my_cfg in myList:

    """SHOW DATA"""
    # Show some sample EEG data if desired
    if my_cfg.show_eeg_sample_plot:
        eeg_sample_plot(my_cfg.subjectIdx_to_plot, my_cfg.seconds_to_plot, my_cfg.channels_to_plot, my_cfg.raw_EDF_list)
    if my_cfg.show_events_distribution:
        events_distribution_plot(epoched.events)

    """CLASSIFICATION"""
    # Get the model
    model_untrained = get_nn_model(my_cfg.nn_list[my_cfg.nn_selection_idx], input_dimension=input_dimension_,
                                   output_dimension=len(my_cfg.selected_classes))

    # Get the optimizer
    optimizer = get_optimizer(my_cfg.optimizer_list[my_cfg.optimizer_selection_idx], my_cfg.learning_rate,
                              model_untrained.parameters(), my_cfg.momentum, my_cfg.weight_decay)

    # Train and show validation loss
    train_losses, train_accuracies, val_losses, val_accuracies, model_trained, time_spent_for_training_s =\
        fit(train_dl, val_dl, model_untrained, optimizer, my_cfg.loss_fn, my_cfg.num_of_epochs)

    # Test the net
    test_loss, test_accuracy = test(model_trained, test_dl, my_cfg.loss_fn, print_loss=True)

    # Store the results
    results_storer.store_results(my_cfg, model_trained, optimizer, test_loss, test_accuracy, train_losses,
                                 train_accuracies, time_spent_for_training_s, val_losses, val_accuracies)



