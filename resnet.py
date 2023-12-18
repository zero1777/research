import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import rkgb.src as rkgb
import time

from asuta import Asuta

device = torch.device("cuda")

net = models.resnet50().to(device)
sample = [torch.rand(10, 3, 224, 224).to(device)]

new_net  = Asuta(net, sample)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

repeat = 3
running_loss = 0.0

for _ in range(repeat):
    optimizer.zero_grad()

    start_time = time.time()
    # outputs = net(sample[0])
    outputs = new_net(*sample)

    loss = criterion(outputs, torch.tensor([1, 2, 3, 4, 5, 5, 4, 3, 2, 1]).to(device))
    loss.backward()
    new_net.backward() 
    optimizer.step()

    # running_loss += loss.item()
    # print(f'loss: {running_loss}')
    # running_loss = 0.0
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f'training_time (sec): {train_time}')