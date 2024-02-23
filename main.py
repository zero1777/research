import rkgb.src as rkgb
import rkgb.src.utils as rkgb_utils
from rkgb.main import make_inputs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from copy import deepcopy
import torch.onnx

from graph import Graph, C_op, D_op, OpSchedule
from utils import *
# from node import C_op, D_op, OpSchedule
from compiler import Compiler, RngState, Storage
from asuta import Asuta


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda")

# model = SimpleCNN().to(device)
# sample = [torch.rand(1, 3, 32, 32).to(device)]

# model = models.resnet50().to(device)
# sample = [torch.rand(5, 3, 224, 224).to(device)]

# model = models.resnet152().to(device)
# sample = [torch.rand(50, 3, 224, 224).to(device)]

model = models.vgg16().to(device)
sample = [torch.rand(5, 3, 224, 224).to(device)]

# model = models.squeezenet1_0().to(device)
# sample = [torch.rand(5, 3, 224, 224).to(device)]

print("---  Doing rematerialization with Asuta ----")

optimizer = torch.optim.Adam(model.parameters())
for_test = Asuta(model, sample)
# compare(for_test, model, sample)
# train_test(for_test, sample, optimizer)

# torch.cuda.empty_cache()
# print(torch.cuda.memory_allocated())
# print(torch.cuda.memory_reserved())

# normal_model_train_test(model, sample)
# y = for_test(*sample)

print('---  Done rematerialization with Asuta ----')