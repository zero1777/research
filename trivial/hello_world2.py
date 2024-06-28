# Original training for ResNet50
import torch
import torchvision.models as models

device = torch.device("cuda")
batch_size = 500
net = models.resnet50().to(device)
sample = torch.rand(batch_size, 3, 128, 128).to(device)

outputs = net(sample)
loss = outputs.mean()
loss.backward()


# Modified version
import torch
import torchvision.models as models

from asuta import Asuta

device = torch.device("cuda")
batch_size = 500
net = models.resnet50().to(device)
sample = torch.rand(batch_size, 3, 128, 128).to(device)

new_net = Asuta(net, sample)

outputs = new_net(sample)
loss = outputs.mean()
loss.backward()
new_net.backward()

# sample = [torch.rand(batch_size, 3, 128, 128).to(device)]

# new_net  = Asuta(net, sample)

# repeat = 3
# running_loss = 0.0

# torch.cuda.reset_peak_memory_stats()

# for _ in range(repeat):
#     outputs = new_net(*sample)

#     loss = outputs.mean()
#     loss.backward()
#     new_net.backward() 


#     peak_mem = torch.cuda.max_memory_allocated()
#     print(f'peak_mem (GB): {peak_mem/1000/1000/1000}')


# from asuta import Asuta