import rkgb.src as rkgb
from rkgb.utils.ast_add_on import ast_to_str
import torch
import torch.nn as nn
import torch.nn.functional as F


class Asuta(torch.nn.Module):
    def __init__(
        self,
        original_model,
        model_inputs,

    ):
        super().__init__()
        self.original_model = original_model
        self.model_inputs = model_inputs
        self.rkgb_results = rkgb.make_all_graphs(
            original_model, model_inputs, verbose=False, bool_kg=True
        )
        self.kgraph_list = self.rkgb_results.K_graph_list
        self.dict_constants = self.rkgb_results.K_graph_list[0].dict_constants
        self.eq_classes = self.rkgb_results.equivalent_classes
        self.init_code = ast_to_str(self.kgraph_list[0].init_code)
        self.output = self.kgraph_list[-1].output_kdn_data


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

model = SimpleCNN()
sample = [torch.randn(1, 3, 32, 32)]
print("---  Doing rematerialization with Asuta ----")
for_test = Asuta(model, sample)
print('---  Done rematerialization with Asuta ----')