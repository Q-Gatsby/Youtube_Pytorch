import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

train_dataset = torchvision.datasets.MNIST(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

load_train = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
load_test = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=28 * 28, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        return self.model(x)


model = MNIST()
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
learning_rate = 1e-3
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

epoch = 10
for i in range(epoch):
    for imgs, targets in load_train:
        # train
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        imgs = torch.reshape(imgs, shape=[-1, 28*28]).to(device)
        targets = targets.to(device)
        output = model(imgs)
        loss = loss_fn(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print(f'---第{i+1}次训练的Loss为{loss.item()}!---')

# test
total_test_loss = 0
with torch.no_grad():
    acc = 0
    for imgs, targets in load_test:
        imgs = torch.reshape(imgs, shape=[-1, 28*28]).to(device)
        targets = targets.to(device)
        output = model(imgs)
        loss = loss_fn(output, targets)
        total_test_loss += loss

        # acc
        # max函数返回两个值 第一个值是values，第二个是对应的indices
        # max returns (value ,index)
        _, prediction = torch.max(output, dim=1)
        acc += (prediction == targets).sum().item()

    accuracy = acc / len(test_dataset)
    print(f'Accuracy: {accuracy}')
