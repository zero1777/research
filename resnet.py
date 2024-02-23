import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import rkgb.src as rkgb
import time

from asuta import Asuta

device = torch.device("cuda")
batch_size = 100

net = models.vgg16().to(device)
s = [torch.rand(50, 3, 128, 128).to(device)]
sample = [torch.rand(batch_size, 3, 128, 128).to(device)]

net = models.resnet50().to(device)
s = [torch.rand(128, 3, 224, 224).to(device)]
sample = [torch.rand(batch_size, 3, 224, 224).to(device)]
# y = torch.randint(1, 91, (batch_size, ))

# p = [torch.rand(100, 3, 224, 224).to(device)]
# net = models.squeezenet1_1().to(device)
# s = [torch.rand(5, 3, 224, 224).to(device)]
# batch_size = 512 
# sample = [torch.rand(batch_size, 3, 224, 224).to(device)]
# y = torch.rand(batch_size, 1000, 1, 1) 

new_net  = Asuta(net, s)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# ss = time.time()
# new_net.forward(*p)
# ee = time.time()
# print(f'init_time (sec): {ee-ss}')

repeat = 1
running_loss = 0.0

torch.cuda.reset_peak_memory_stats()
max_before = torch.cuda.max_memory_allocated()/1000/1000/1000
print(f"Before: {max_before}, {torch.cuda.memory_reserved()/1000/1000/1000}")

for _ in range(repeat):
    optimizer.zero_grad()

    start_time = time.time()
    # outputs = net(sample[0])
    outputs = new_net(*sample)

    # loss = criterion(outputs, y.to(device))
    loss = outputs.mean()
    loss.backward()
    new_net.backward() 
    optimizer.step()

    # running_loss += loss.item()
    # print(f'loss: {running_loss}')
    # running_loss = 0.0
    
    end_time = time.time()
    train_time = end_time - start_time
    print(f'training_time (sec): {train_time}')

    peak_mem = torch.cuda.max_memory_allocated() - max_before
    print(f'peak_mem (GB): {peak_mem/1000/1000/1000}')

# print(f'peak_mem (GB): {peak_mem/1024/1024/1024}')
