import torch
import torchvision
import time
import torchvision.transforms as transforms
from rkgb.main import make_inputs

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

from asuta import Asuta

device = torch.device("cuda")
net = Net().to(device)
sample = [torch.rand(1, 3, 32, 32).to(device)]
new_net = Asuta(net, sample)

stream = torch.cuda.current_stream(device)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# datas = []
# for i, data in enumerate(trainloader, 0):
#     inputs, labels = data
#     start_time = time.time()
#     model_inputs = make_inputs(net, inputs, {})
#     end_time = time.time()
#     training_time = end_time - start_time
#     print(f'training_time (sec): {training_time}')
    # datas.append([inputs, labels])

# print(datas)
    
for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    start_event.record(stream)
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # outputs = net(inputs)
        outputs = new_net(inputs)

        # loss = criterion(outputs, labels)
        # loss = torch.mean(outputs)
        # loss.backward()

        # new_net.backward()
        # optimizer.step()

        # print statistics
        # running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #     running_loss = 0.0

    end_event.record(stream)
    torch.cuda.synchronize(device)
    training_time = start_event.elapsed_time(end_event)
    print(f'training_time (sec): {training_time/1000}')

print('Finished Training')