import torch
import torchvision
import torchvision.models as models
import torch.optim as optim
from gpt import get_GPT

import time

with_optimizer = False

device = torch.device("cuda")
batch_size = 500

# model
# net = models.vgg16().to(device)
# net = models.resnet50().to(device)
# net = models.resnet152().to(device)
# s = torch.rand(batch_size, 3, 128, 128).to(device)

# GPT model
batch_size_gpt = 15
net = get_GPT(model="GPT2-small").to(device)
s = torch.randint(0,600, [batch_size_gpt, 512]).to(device)


optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
torch.cuda.reset_peak_memory_stats()
repeat = 3

for _ in range(repeat):
    if with_optimizer:
        optimizer.zero_grad()

    outputs = net(s)
    loss = outputs.mean()
    loss.backward()
    
    if with_optimizer:
        optimizer.step()

print(f"Peak memory: {torch.cuda.max_memory_allocated()/1000/1000/1000} GB")
