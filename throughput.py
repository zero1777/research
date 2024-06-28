import torch
import torchvision
import torchvision.models as models
import torch.optim as optim
import math
from gpt import get_GPT

from modeler import Modeler

import time


# data = torch.rand(10000, 3, 128, 128)
# batch_size = 512
# num_batches = math.ceil(data.size()[0] / batch_size)

# data_list = [data[batch_size * x: batch_size * (x+1),:,:,:] for x in range(num_batches)]

with_optimizer = False

device = torch.device("cuda")

# model
# net = models.vgg16().to(device)
# net = models.resnet50().to(device)
# net = models.resnet152().to(device)
# s = torch.rand(batch_size, 3, 128, 128).to(device)
# test = [torch.rand(256, 3, 128, 128).to(device)]

# GPT
net = get_GPT(model="GPT2-small").to(device)
test = [torch.randint(0, 600, [18, 512]).to(device)]
data = torch.randint(0, 600, [6000, 512]).to(device)
batch_size = 20
num_batches = math.ceil(data.size()[0] / batch_size)
print(f'num_batches: {num_batches}')
data_list = [data[batch_size * x: batch_size * (x+1),:] for x in range(num_batches)]

md = Modeler(net)
new_model = md.build(test, 14.8, "default")

del md
torch.cuda.empty_cache()


torch.cuda.reset_peak_memory_stats()
# repeat = 3

start = time.time()

# Pytorch baseline
# for d in data_list:
#     d = d.to(device)
#     # print(s.size())
#     outputs = net(d)
#     loss = outputs.mean()
#     loss.backward()

# modeler
for d in data_list:
    d = d.to(device)
    # print(d.size())
    outputs = new_model(d)
    loss = outputs.mean()
    loss.backward()
    new_model.backward()

end = time.time()

print(f'Training time (sec): {end-start}')    
print(f"Peak memory: {torch.cuda.max_memory_allocated()/1000/1000/1000} GB")
