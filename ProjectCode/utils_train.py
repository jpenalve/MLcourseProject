import numpy as np
import torch
import time
from datetime import timedelta
from copy import deepcopy
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import matplotlib.pyplot as plt
from logger import Logger
from visualisations import write_logs_for_tensorboard

def train(model, train_loader, optimizer, loss_fn, print_every=100):
    '''
    Trains the model for one epoch
    '''
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


def test(model, test_loader, loss_fn, print_loss=False):
    '''
    Tests the model on data from test_loader
    '''
    model.eval()
    test_loss = 0
    n_correct = 0
    number_of_classes = 10
    class_correct = list(0. for i in range(number_of_classes))
    class_total = list(0. for i in range(number_of_classes))
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
                class_total[label] += 1



    average_loss = test_loss / len(test_loader)
    accuracy = 100.0 * n_correct / len(test_loader.dataset)
    if print_loss:
        #print('--> Test average loss: {:.4f}, accuracy: {:.3f}'.format(average_loss, accuracy))
        #print("\n\n")

        # Look at each class
        for i in range(number_of_classes):
            # Class not present
            if class_total[i] == 0:
                print('Accuracy of class %1d : %19s' % (i, 'No label available'))
            else:
                print('Accuracy of class %1d : %2d %% of %1d labels'
                      % (i, 100 * class_correct[i] / class_total[i], class_total[i]))


    return average_loss, accuracy


def fit(train_dataloader, val_dataloader, model, optimizer, loss_fn, n_epochs, scheduler=None, apply_early_stopping=False, estop_patience=5):
    time_start = time.time()
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    logger = Logger('./logs')
    if apply_early_stopping:
        best_val_loss = np.inf
        best_model = None
        patience = estop_patience  # if no improvement after estop_patience epochs, stop training
        counter = 0
    # Track learning rate
    previous_learning_rate =  optimizer.param_groups[0]['lr'];
    print('Learning Rate: ', optimizer.param_groups[0]['lr'])
    for epoch in range(n_epochs):
        if previous_learning_rate != optimizer.param_groups[0]['lr']:
            print('New Learning Rate: ', optimizer.param_groups[0]['lr'])
        train_loss, train_accuracy = train(model, train_dataloader, optimizer, loss_fn)
        print_val_loss = True
        val_loss, val_accuracy = test(model, val_dataloader, loss_fn, print_val_loss)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        # We'll monitor learning rate -- just to show that it's decreasing
        if scheduler:
            scheduler.step()  # argument only needed for ReduceLROnPlateau
        print('-> Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}%, val_loss: {:.4f}, val_accuracy: {:.4f}%'.format(
            epoch + 1, n_epochs,
            train_losses[-1],
            train_accuracies[-1],
            val_losses[-1],
            val_accuracies[-1]))
        print('--------------------------------------------------------------------- \n')
        if apply_early_stopping:
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


    time_spent_for_training_s = str(timedelta(seconds=time.time()-time_start))
    print("Time spend for training: ", time_spent_for_training_s, " hh:mm:ss.ms \n")
    return train_losses, train_accuracies, val_losses, val_accuracies, model, time_spent_for_training_s





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

