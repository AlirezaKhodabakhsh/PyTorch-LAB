import numpy as np
import torch
import os

#########################################################################################################
#########################################################################################################
def train(model, train_loader, device, optimizer, criterion, epoch):
    # !!!
    print(60 * "#")
    print(6 * "#" + " Epoch " + str(epoch) + " " + 45 * "#")
    print(60 * "#")

    # loss/acc all training data
    loss_train, acc_train = 0, 0

    # Set mode on "train mode"
    model.train()

    for iter_train, (image, label) in enumerate(train_loader, 1):
        # label = label.to(torch.long)

        # Transfer training data on desired device
        image, label = image.to(device), label.to(device)

        # "require_grad" have accumulating property
        optimizer.zero_grad()

        # Feed training data to model
        y_hat = model(image)

        # average of mini-batch loss & mini-batch accuracy
        loss = criterion(y_hat, label)
        acc = np.count_nonzero(label.to('cpu').numpy() == y_hat.argmax(dim=1).to('cpu').numpy())

        # summation of all mini-batches loss (loss of all data)
        # & summation of all mini-batches accuracy (accuracy of all data)
        loss_train += loss.item() * image.size(0)
        acc_train += acc

        # Gradient
        loss.backward()

        # Update all parameters that embedded on "optimizer".
        # Update all parameters that "require_grad" is true.
        optimizer.step()

    # average loss/accuracy of all training data
    loss_train /= len(train_loader.sampler)
    acc_train /= len(train_loader.sampler)

    # Display average loss/accuracy of all training data
    print("Loss_Train: {:.2f}\tAcc_Train : {:.2f}".format(loss_train, acc_train))

    # output of function in order to use in loss/accuracy plot
    return loss_train, acc_train
#########################################################################################################
#########################################################################################################
def valid(model, valid_loader, device, criterion):
    # loss/acc all validation data
    loss_valid, acc_valid = 0, 0

    # Set mode on "valid mode"
    model.eval()

    for iter_valid, (image, label) in enumerate(valid_loader, 1):
        # in evaluation phase, we don't need to use gradient because of lack of updating.
        with torch.no_grad():
            #label = label.to(torch.long)

            # Transfer training data on desired device
            image, label = image.to(device), label.to(device)

            # Feed training data to model
            y_hat = model(image)

            # average of mini-batch loss & mini-batch accuracy
            loss = criterion(y_hat, label)
            acc = np.count_nonzero(label.to('cpu').numpy() == y_hat.argmax(dim=1).to('cpu').numpy())

            # summation of all mini-batches loss (loss of all data)
            # & summation of all mini-batches accuracy (accuracy of all data)
            loss_valid += loss.item() * image.size(0)
            acc_valid += acc

    # average loss/accuracy of all training data
    loss_valid /= len(valid_loader.sampler)
    acc_valid /= len(valid_loader.sampler)

    # Display average loss/accuracy of all training data
    print("Loss_Valid: {:.2f}\tAcc_Valid : {:.2f}".format(loss_valid, acc_valid))

    return loss_valid, acc_valid
#########################################################################################################
#########################################################################################################
def save_model(model, optimizer, epoch, root, loss_valid_min, loss_valid):
    if loss_valid <= loss_valid_min:

        filename = root + '\model_epoch_{}_loss_valid_{:.2f}.pt'.format(epoch, loss_valid)
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   filename)

        loss_valid_min = loss_valid
        print("\nSave Model: {}".format("YES"))
    else:
        print("\nSave Model: {}".format("NO"))
    return loss_valid_min
#########################################################################################################
#########################################################################################################
def test(untrained_model, test_loader, device, criterion, root):
    # Define untrained model
    model = untrained_model

    # Load best model from desired directory
    best_model = torch.load(root)

    # Put parameters of best model on untrained model
    model.load_state_dict(best_model['state_dict'])

    # loss/acc all test data
    loss_test, acc_test = 0, 0

    # Set mode on "Test mode"
    model.eval()

    for iter_test, (image, label) in enumerate(test_loader, 1):
        # label = label.to(torch.long)

        # Transfer training data on desired device
        image, label = image.to(device), label.to(device)

        # Feed training data to model
        y_hat = model(image)

        # average of mini-batch loss & mini-batch accuracy
        loss = criterion(y_hat, label)
        acc = np.count_nonzero(label.to('cpu').numpy() == y_hat.argmax(dim=1).to('cpu').numpy())

        # summation of all mini-batches loss (loss of all data)
        # & summation of all mini-batches accuracy (accuracy of all data)
        loss_test += loss.item() * image.size(0)
        acc_test += acc

    # average loss/accuracy of all training data
    loss_test /= len(test_loader.sampler)
    acc_test /= len(test_loader.sampler)

    #
    print(60 * "#")
    print(6 * "#" + " TEST INFO " + 43 * "#")
    print(60 * "#")

    #
    print("Best Epoch : {}\tLoss_Test : {:.2f}\tAcc_Test : {:.2f}".format(best_model['epoch'], loss_test, acc_test))

    #
    print(60 * "#")
    print(6 * "#" + " TEST INFO " + 43 * "#")
    print(60 * "#")
