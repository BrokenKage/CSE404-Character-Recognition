import torch
from torch import nn
from torch.nn import Sequential, Linear, MaxPool2d, ReLU, Conv2d
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

import matplotlib.pyplot as plt
from IPython import display

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        self.conv_stack = Sequential(
            Conv2d(1, 6, kernel_size=3),
            ReLU(),
            MaxPool2d(2,2),
            Conv2d(6, 32, kernel_size=4)
        )
        self.linear_relu_stack = Sequential(
            Linear(32*10*10, 512),
            ReLU(),
            Linear(512,512),
            ReLU(),
            Linear(512,47)
        )
    
    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    losses = []
    for batch, (X, y) in enumerate(dataloader):
        X , y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        losses.append(loss)
    return np.array(losses).mean()

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax() == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size

    return correct

def main():
    training_data = datasets.EMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
        split="balanced"
    )
    test_data = datasets.EMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
        split="balanced"
    )

    print(len(training_data))
    print(len(test_data))

    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    # Remove this
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    class_names, indexes = np.unique(training_data.targets, return_index=True)
    perm = class_names.argsort()
    indexes = indexes[perm]
    class_names = [training_data.classes[i] for i in class_names]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    #print(model)

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-2

    optimizer = torch.optim.SGD(model.parameters(), learning_rate)

    epochs = 10
    history = {"losses": [], "accuracies": []}
    for t in range(epochs):
        print(f"Epoch: {t+1}")
        history['losses'].append(
            train(train_dataloader, model, loss_fn, optimizer, device)
        )

        history['accuracies'].append(
            test(test_dataloader, model, loss_fn, device)
        )
        #plt.clf()
        #fig1 = plt.figure()
        #plt.plot(history["losses"], 'r-', lw=2, label='loss')
        #plt.legend()
        #display.clear_output(wait=True)
        #display.display(plt.gcf())

        #plt.clf()
        #fig2 = plt.figure()
        #plt.plot(history["accuracies"], 'b-', lw=1, label='accuracy')
        #plt.legend()

        #display.display(plt.gcf())

    n_rows = 6
    n_cols = 8
    plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col

            if index > 46:
                break

            plt.subplot(n_rows, n_cols, index + 1)
            X, y = test_data.__getitem__(index)
            y_pred = model(X.to(device)[None,...])
            y_pred = y_pred.argmax(1)
            plt.imshow(X[0], cmap="binary", interpolation="nearest")
            plt.axis('off')

            if y == y_pred:
                plt.title(class_names[y_pred], fontsize=12, color='g')
            else:
                plt.title(class_names[y_pred], fontsize=12, color='r')
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()

if __name__ == "__main__":
    main()