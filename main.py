import rkgb.src as rkgb
import rkgb.src.utils as rkgb_utils
from rkgb.main import make_inputs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from copy import deepcopy

from graph import Graph
from utils import *
from node import C_node, D_node, OpSchedule
from compiler import Compiler, RngState, Storage


class Asuta(torch.nn.Module):
    def __init__(
        self,
        original_model,
        model_inputs,
    ):
        super().__init__()
        self.graph = Graph(original_model, model_inputs)
        self.device = get_device()
        self.recompute_list = ["__7_input data"]
        self.storage = Storage(self.device, self.graph.model, self.graph.dict_constants)
        self.construct_op_list()
        self.add_evict_regenerate()
        # self.compile_function()

        
    def construct_op_list(self):
        self.tmp_fwd_op_list = []
        self.tmp_bwd_op_list = []
        self.fwd_op_list = []
        self.bwd_op_list = []
        self.kdn_users_counter = {} # dict: kdn.name -> num of users
        self.kdn_dict = {} # dict: kdn.name -> kdn

        for kg in self.graph.graph_list:
            users = {} # dict: name -> num of users
            op_list = [] # list of C_node and D_node
            alive_datas = set() # current alive datas (Notice that the range of alive datas is only in one k_graph, we will reconstruct it later)
            recompute_tensors = {} # dict: kdn.name -> kcn

            # initialze
            for kdn in kg.list_kdn:
                self.kdn_dict[kdn.name] = kdn
                self.kdn_users_counter[kdn.name] = len(kdn.users_global)
                users[kdn.name] = len(kdn.users_real)
                # print(f'kdn: {kdn.name}, {[c.name for c in kdn.users_global]}')
            
            for kcn in kg.list_kcn:
                op_list.append(C_node(kcn, alive_datas=alive_datas))
                # "loss" op needs to be executed in both forward and backward, so we need to add it twice
                if "loss" in kcn.name:
                    op_list.append(C_node(kcn, alive_datas=alive_datas))
                for kdn in kcn.users:
                    alive_datas.add(kdn.name)

                ''' # debug
                print(f'kcn: {kcn.name}, alive_datas: {alive_datas}')
                ''' # debug

                # update the counter of used kdn (decrement) since kcn is executed
                for deps in kcn.deps_global:
                    if deps.name not in users:
                        continue
                    if deps not in kcn.deps_fake:
                        users[deps.name] -= 1
                        # if the counter is 0, then the kdn is no longer needed
                        if users[deps.name] == 0:
                            alive_datas.remove(deps.name)
                            op_list.append(D_node(deps))

            
            ''' # debug
            for op in op_list:
                print(f'op: {op}')
            ''' # debug
        
            loss_idx = 0
            for i, op in enumerate(op_list):
                if "loss" in op.name:
                    loss_idx = i
                    break

            self.tmp_fwd_op_list.append(op_list[:loss_idx+1])
            self.tmp_bwd_op_list.append(op_list[loss_idx+1:])

        self.fwd_op_list = [op for fwlist in self.tmp_fwd_op_list for op in fwlist]
        reverse_bwd_op_list = self.tmp_bwd_op_list[::-1]
        self.bwd_op_list = [op for bwlist in reverse_bwd_op_list for op in bwlist]

        ''' # debug
        for op in self.fwd_op_list:
            print(f'fwd_op: {op}')

        for op in self.bwd_op_list:
            print(f'bwd_op: {op}')
        ''' # debug

        list_kdn = []
        for kg in self.graph.graph_list:
            list_kdn += kg.list_kdn
        
        self.op_sched = OpSchedule(
            self.fwd_op_list + self.bwd_op_list,
            None,
            self.graph.graph_list[0].input_kdn_data,
            self.graph.graph_list[0].input_kdn_grad,
            self.graph.output,
            list_kdn,
        )

        self.data_memory = {}
        for kdn in list_kdn:
            self.data_memory[kdn.name] = kdn.mem
        
        ''' # debug
        print(f'data_memory: {self.data_memory}')
        ''' # debug

    def add_evict_regenerate(self):
        users = self.kdn_users_counter.copy()
        alive_datas = set() # current alive datas
        evict_list = []

        self.fwd_op_list_v2 = []

        # forward list
        for op in self.fwd_op_list:
            self.fwd_op_list_v2.append(op)
            if isinstance(op, C_node):
                # print(f'add_evict_regenerate: {op.name}, {op.users_global}')
                op.alive_datas = alive_datas.copy()
                for deps_name in op.deps_global:
                    if deps_name not in users:
                        assert deps_name == "sources data"
                        continue
                    if deps_name in evict_list:
                        print(f'op: {op.name}, deps_name: {deps_name}')

                    users[deps_name] -= 1

                    if deps_name in self.recompute_list:
                        assert "grad" not in deps_name
                        if users[deps_name] == 1:
                            evict_list.append(deps_name)
                            self.fwd_op_list_v2.append(D_node(self.kdn_dict[deps_name]))
                for kdn_name in op.users_global:
                    alive_datas.add(kdn_name)

            elif isinstance(op, D_node):
                alive_datas.remove(op.name)
        
        # for a in self.fwd_op_list_v2:
            # print(f'{a}')
            
        # def reconstruct_kdn(kdn_name):
        #     parent_kcn = recompute_tensors[kdn_name]
        #     for deps in parent_kcn.deps_global:
        #         if deps.name in alive_datas:
        #             continue
        #         if deps.name not in users:
        #             continue
        #         assert deps.name in recompute_tensors
        #         reconstruct_kdn(deps.name)

        #     op_list.append(C_node(parent_kcn, alive_datas=alive_datas))
        #     for kdn in parent_kcn.users:
        #         alive_datas.add(kdn.name)
            
        # if kdn is not used anymore in forward and in swap list, then we can delete it first
        # for kdn in kg.list_kdn:
        #     if kdn.name not in self.recompute_list:
        #         continue
        #     if users[kdn.name] == 1:
        #         assert len(kdn.deps_global) == 1
        #         recompute_tensors[kdn.name] = list(kdn.deps_global)[0]
        #         alive_datas.remove(kdn.name)
        #         op_list.append(D_node(kdn))
        #         print(f'R_test: {kdn.name}, {[k.name for k in kdn.deps_global]}')
    
                # if "bwd" in kcn.name:
        #     for deps in kcn.deps_global:
        #         if deps.name in alive_datas:
        #             continue
        #         print(f'deps name: {deps.name}')
        #         assert deps.name in recompute_tensors
        #         reconstruct_kdn(deps.name)
        #         del(recompute_tensors[deps.name])

    def compile_function(self):
        self.compiler = Compiler(self.storage)
        self.fct_list = self.compiler.compile(self.op_sched) # compile op_sched -> list of functions
        loss_idx = len(self.fwd_op_list)
        self.fwd_fct_list = self.fct_list[:loss_idx]
        self.bwd_fct_list = self.fct_list[loss_idx:]

        ''' # debug
        for l in self.fwd_fct_list:
            print(f'fwd_fct: {l}')
        print('\n')
        for l in self.bwd_fct_list:
            print(f'bwd_fct: {l}')
        ''' # debug

    def _exec(self, fct_list):
        for fct in fct_list:
            fct()
    
    def forward(self, *args, **kwargs):
        if not self.training:
            self.graph.model.eval()
            return self.graph.model(*args, **kwargs)
        
        # set input data
        model_inputs = make_inputs(self.graph.model, args, kwargs)
        for k, v in model_inputs.items():
            print(f'k: {k}, v: {v}')
            self.storage.add_val(k, v)
        
        # execute init code
        exec(self.graph.init_code, self.storage.gd, self.storage.ld)

        for kg in self.graph.graph_list:
            for kdn in kg.list_kdn:
                tensor_val = torch.empty(
                    0, device=self.device,
                    requires_grad=kdn.info.requires_grad
                )
                self.storage.ld[kdn.main_target] = tensor_val
        
        # execute the generated function list (forward)
        for l in self.fwd_fct_list:
            self._exec(l)

        return self.storage.get_val(self.graph.output.main_target)
    
    def backward(self):
        # execute the generated function list (backward)
        for l in self.bwd_fct_list:
            self._exec(l)

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

model = SimpleCNN().to(device)
sample = [torch.rand(1, 3, 32, 32).to(device)]

# model = models.resnet50().to(device)
# sample = [torch.rand(5, 3, 224, 224).to(device)]

print("---  Doing rematerialization with Asuta ----")

# compare(Asuta(model, sample), model, sample)
# optimizer = torch.optim.Adam(model.parameters())
for_test = Asuta(model, sample)
# train_test(for_test, sample, optimizer)
# y = for_test(sample)

print('---  Done rematerialization with Asuta ----')