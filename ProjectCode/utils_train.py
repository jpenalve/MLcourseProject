import numpy as np
import torch
import time
from datetime import timedelta
from copy import deepcopy
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import matplotlib.pyplot as plt
from logger import Logger
from visualisations import write_logs_for_tensorboard, write_accuracies_for_tensorboard

def train(model, train_loader, optimizer, config):
    '''
    Trains the model for one epoch
    '''
    loss_fn = config.loss_fn
    
    model.train()
    losses = []
    n_correct = 0

    for iteration, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        outputs = model(data)
        optimizer.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        predicted_labels = outputs.argmax(1)
        n_correct += torch.sum(predicted_labels == labels).item()
        #if iteration % print_every == 0:
        #   print('Training iteration {}: loss {:.4f}'.format(iteration, loss.item()))
    accuracy = 100.0 * n_correct / len(train_loader.dataset)
    return np.mean(np.array(losses)), accuracy


def test(model, test_loader, config, print_loss=False, path_name_txt=None, epoch = None, logger = None):
    '''
    Tests the model on data from test_loader
    '''
    loss_fn = config.loss_fn
    model.eval()
    test_loss = 0
    n_correct = 0
    number_of_classes = config.nClasses
    
    class_correct = list(0. for i in range(number_of_classes))
    class_appearances = list(0. for i in range(number_of_classes))
    class_accuracy = list(0. for i in range(number_of_classes))
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_fn(output, labels)
            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            n_correct += torch.sum(predicted == labels).item()
            correct_idcs = (predicted == labels).squeeze()
            tmp_batch_size = images.shape[0]
            for i in range(tmp_batch_size):
                label = labels[i]
                class_correct[label] += correct_idcs[i].item()
                class_appearances[label] += 1

    if (epoch is not None) and (logger is not None):
        for i in range(number_of_classes):
            if not(class_appearances[i] == 0):
                class_accuracy[i] = 100 * class_correct[i] / class_appearances[i]
        write_accuracies_for_tensorboard(class_accuracy, epoch, model, logger)
        
    average_loss = test_loss / len(test_loader)
    accuracy = 100.0 * n_correct / len(test_loader.dataset)
    
    if print_loss:
        if path_name_txt is not None:
            write_class_accuracies_to_txt(path_name_txt, config, class_appearances, class_correct)

    return average_loss, accuracy

def fit(train_dataloader, val_dataloader, model, optimizer, config):
    
    time_start = time.time()
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    logger = Logger('./logs',config.curve_name)
    logger_train = Logger('./logs',config.curve_name,training=True)
    
    if config.scheduler:
        tmp_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config.schStepSize, gamma=config.schGamma)

        
    if config.use_early_stopping:
        best_val_loss = np.inf
        best_model = None
        patience = config.es_patience  # if no improvement after estop_patience epochs, stop training
        counter = 0
        
    # Track learning rate
    previous_learning_rate =  optimizer.param_groups[0]['lr'];
    print('Learning Rate: ', optimizer.param_groups[0]['lr'],'\n')
    
    for epoch in range(config.num_of_epochs):
        if previous_learning_rate != optimizer.param_groups[0]['lr']:
            previous_learning_rate = optimizer.param_groups[0]['lr']
            print('\nNew Learning Rate: ', optimizer.param_groups[0]['lr'])
            
            
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, config)
        val_loss, val_accuracy = test(model, val_dataloader, config, epoch=epoch, logger=logger)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # We'll monitor learning rate -- just to show that it's decreasing
        if config.scheduler:
            tmp_scheduler.step()  # argument only needed for ReduceLROnPlateau
            
        print('-> Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}%, val_loss: {:.4f}, val_accuracy: {:.4f}%'.format(
            epoch + 1, config.num_of_epochs,
            train_losses[-1],
            train_accuracies[-1],
            val_losses[-1],
            val_accuracies[-1]))
        
        if config.use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = deepcopy(model)
                counter = 0
            else:
                counter += 1
            if counter == patience:
                print('No improvement for {} epochs; training stopped.'.format(patience))
                model = best_model
                break
        
        write_logs_for_tensorboard(val_loss, val_accuracy, epoch, model, logger)
        write_logs_for_tensorboard(train_loss, train_accuracy, epoch, model, logger_train)


    time_spent_for_training_s = str(timedelta(seconds=time.time()-time_start))
    print("Time spend for training: ", time_spent_for_training_s, " hh:mm:ss.ms \n")
    return train_losses, train_accuracies, val_losses, val_accuracies, model, time_spent_for_training_s



def write_class_accuracies_to_txt(path_name_txt, config, class_appearances, class_correct, print_enabled=True):

    num_of_classes = config.nClasses
    txt_file_handle = open(path_name_txt, "a")
    
    # Look at each class
    for i in range(num_of_classes):
        # Class not present
        if class_appearances[i] == 0:
            if print_enabled:
                print('Accuracy of class %1d : %19s' % (i, 'No label available'))

            txt_file_handle.write('Accuracy of class %1d : %19s' % (i, 'No label available \n'))
        else:
            if print_enabled:
                print('Accuracy of class %1d : %2d %% of %1d labels'
                      % (i, 100 * class_correct[i] / class_appearances[i], class_appearances[i]))

            txt_file_handle.write('Accuracy of class %1d : %2d %% of %1d labels \n'
                                  % (i, 100 * class_correct[i] / class_appearances[i], class_appearances[i]))
    txt_file_handle.close()
    

def final_test_acc(model_trained, test_dl, config):
    # Test the net
    print('\nPerformance on the test set:')
    test_loss, test_accuracy = test(model_trained, test_dl, config, print_loss=True)
    print('->test_loss: {:.4f}, test_accuracy: {:.4f}%'.format(test_loss, test_accuracy))
    return test_loss, test_accuracy
    
def plot_all_metrics(training_curves):
    
    plt.figure(figsize=(20, 6))
    plt.subplot(121)
    keys = []
    for k, v in sorted(training_curves.items()):
        plt.plot(np.arange(len(v[1])), v[1])
        keys.append("tra_"+k)
    for k, v in sorted(training_curves.items()):
        plt.plot(np.arange(len(v[3])), v[3])
        keys.append("val_"+k)
    plt.title('Accuracy for different optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(keys)
    plt.grid(True)
    
    plt.subplot(122)
    keys = []
    for k, v in sorted(training_curves.items()):
        plt.plot(np.arange(len(v[0])), v[0])
        keys.append("tra_"+k)
    for k, v in sorted(training_curves.items()):
        plt.plot(np.arange(len(v[2])), v[2])
        keys.append("val_"+k)
    plt.title('Loss for different optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(keys)
    plt.grid(True)
    
    
    
def plot_val_metrics(training_curves):
    
    plt.figure(figsize=(20, 6))
    plt.subplot(121)
    keys = []
    for k, v in sorted(training_curves.items()):
        plt.plot(np.arange(len(v[3])), v[3])
        keys.append("val_"+k)
    plt.title('Validation Accuracy for different optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(keys)
    plt.grid(True)
    
    plt.subplot(122)
    keys = []
    for k, v in sorted(training_curves.items()):
        plt.plot(np.arange(len(v[2])), v[2])
        keys.append("val_"+k)
    plt.title('Validation Loss for different optimizers')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(keys)
    plt.grid(True)

