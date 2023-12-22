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
s = [torch.rand(5, 3, 224, 224).to(device)]
batch_size = 128 
sample = [torch.rand(batch_size, 3, 224, 224).to(device)]

new_net  = Asuta(net, s)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
y = torch.randint(1, 91, (batch_size, ))

repeat = 1
running_loss = 0.0

torch.cuda.reset_peak_memory_stats()
max_before = torch.cuda.max_memory_allocated()
print(f"Before: {max_before}, {torch.cuda.memory_reserved()}")

for _ in range(repeat):
    optimizer.zero_grad()

    start_time = time.time()
    # outputs = net(sample[0])
    outputs = new_net(*sample)
    peak_mem = torch.cuda.max_memory_allocated() - max_before
    print(f'peak_mem: {peak_mem}')

    # loss = criterion(outputs, y.to(device))
    # loss.backward()
    # new_net.backward() 
    # optimizer.step()

    # running_loss += loss.item()
    # print(f'loss: {running_loss}')
    # running_loss = 0.0
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f'training_time (sec): {train_time}')

# print(f'peak_mem (GB): {peak_mem/1024/1024/1024}')