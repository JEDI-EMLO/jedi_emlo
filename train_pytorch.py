import torch
import numpy as np
import pandas as pd
import sys
import os

import torch.nn as nn

import torchvision.models as models
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

pathname = os.getcwd()
path = os.path.abspath(pathname)

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'model.h5'
train_data_dir = os.path.join('data', 'train')
validation_data_dir = os.path.join('data', 'validation')
cats_train_path = os.path.join(path, train_data_dir, 'cats')
nb_train_samples = 2 * len([name for name in os.listdir(cats_train_path)
                            if os.path.isfile(
        os.path.join(cats_train_path, name))])
nb_validation_samples = 800
epochs = 10
batch_size = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_bottlebeck_features():
    model = models.vgg16(pretrained=True)
    model_f = nn.Sequential(*list(model.children())[0])
    model_f = model_f.to(device)
    model_f.eval()

    data = torchvision.datasets.ImageFolder(root=train_data_dir, transform=transforms.Compose(
        [transforms.Resize((img_width, img_height)), transforms.ToTensor()]))
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    bottleneck_features_train = []
    with torch.no_grad():
        for img, label in dataloader:
            img = img.to(device)
            output = model_f(img).data.cpu().numpy()
            bottleneck_features_train.extend(output)
            np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

    data = torchvision.datasets.ImageFolder(root=validation_data_dir, transform=transforms.Compose(
        [transforms.Resize((img_width, img_height)), transforms.ToTensor()]))
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    bottleneck_features_validation = []
    for img, label in dataloader:
        img = img.to(device)
        output = model_f(img).data.cpu().numpy()
        bottleneck_features_validation.extend(output)
        np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

    return


loss_fn = nn.BCELoss()


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    # pbar = tqdm(train_loader)
    train_loss = 0
    train_correct = 0
    # scheduler.step()
    for batch_idx, (data, target) in enumerate(train_loader):
        # target = target.unsqueeze(1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target.float())
        # print(loss)
        loss.backward()
        optimizer.step()
        # pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
        # train_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        train_loss += loss_fn(output, target.float()).sum().item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        train_correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    acc = 100. * train_correct / len(train_loader.dataset)
    print('Epoch: {:.0f},LR: {}.\nTrain set: train Average loss: {:.4f}, train_Accuracy: {}/{} ({:.4f}%)\n'.format(
        epoch, optimizer.param_groups[0]['lr'], train_loss, train_correct, len(train_loader.dataset),
        100. * train_correct / len(train_loader.dataset)))
    return train_loss, acc


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            test_loss += loss_fn(output, target.float()).sum().item()
            pred_test = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred_test.eq(target.view_as(pred_test)).sum().item()

    test_loss /= len(test_loader.dataset)
    val_acc = 100. * correct / len(test_loader.dataset)

    print('Test set: test Average loss: {:.4f}, test Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, val_acc


def train_top_model():
    x_data = torch.from_numpy(np.load(open('bottleneck_features_train.npy', 'rb')))
    x_labels = torch.from_numpy(np.array([0] * (int(nb_train_samples / 2)) + [1] * (int(nb_train_samples / 2))))
    train_data = []
    for i in range(len(x_data)):
        train_data.append([x_data[i], x_labels[i]])
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=False, batch_size=batch_size)

    y_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    y_labels = np.array(
        [0.0] * (int(nb_validation_samples / 2)) +
        [1.0] * (int(nb_validation_samples / 2)))
    val_data = []
    for i in range(len(y_data)):
        val_data.append([y_data[i], y_labels[i]])
    valloader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=batch_size)
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(8192, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()).to(device)
    optimizer = optim.RMSprop(net.parameters())

    metrics = []
    for epoch in range(epochs):
        train_loss, acc = train(net, device, trainloader, optimizer, epoch)
        test_loss, val_acc = test(net, device, valloader)
        metrics.append([epoch, train_loss, acc, test_loss, val_acc])

    metrics_df = pd.DataFrame(metrics, columns=['epoch', 'train_loss', 'acc', 'test_loss', 'val_acc'])
    metrics_df.to_csv('metrics.csv', index=False)

    torch.save(net, 'model.pth')


save_bottlebeck_features()
train_top_model()
