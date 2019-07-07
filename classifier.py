import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from visdom import Visdom
import numpy as np
from data_reader import ClassDataset

from networks import *

vis = Visdom(env='model_1')
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Hyper parameters
num_epochs = 30
num_classes = 3
batch_size = 3
learning_rate = 0.001
# dataset

train_file = 'classifier_summary.txt'
test_file = 'classifier_test.txt'
train_data = ClassDataset(train_file,
                       transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
test_data = ClassDataset(test_file,
                      transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=1,
                                          shuffle=True)


# model = ConvNet(num_classes).to(device)
# model = resnet18(pretrained=False).to(device)
# model = alexnet().to(device)
model = ResNet18().to(device)
# model = VGG('VGG16').to(device)
# if os.path.exists('model.pkl'):
#     # model.load_state_dict(torch.load('model.pkl'))
#     model = torch.load('model.pkl')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 50 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([loss.item()]), win='total_loss', update='append' if i > 0 else None)

# Test the model
    model.eval()  # eval mode
    acc = np.zeros(6)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # out_arr = outputs.to(torch.device('cpu')).numpy()
            # index = np.argmax(out_arr)
            if (predicted == labels).sum().item() == 1:
                acc[labels.to(torch.device('cpu')).numpy() - 1] += 1
            # print('right: ' + str((predicted == labels).sum().item()) + ' ' + 'predicted: ' + str(index))
        vis.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([100 * correct / total]), win='acc',
                 update='append' if i > 0 else None)
        print('Test Accuracy of the model on the test data: {} %'.format(100 * correct / total))
    for i in range(num_classes + 1):
        print(acc[i])
# Save the model checkpoint
# torch.save(model.state_dict(), 'model.pkl')
torch.save(model, 'model.pkl')