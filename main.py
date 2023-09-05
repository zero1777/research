import rkgb.src as rkgb
import rkgb.src.utils as rkgb_utils
from rkgb.main import make_inputs

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from geometric import Geometric
from utils import *
from def_op import RunOp, DelOp, OpSchedule
from compiler import Compiler, RngState, RK_Storage


class Asuta(torch.nn.Module):
    def __init__(
        self,
        original_model,
        model_inputs,
        mode = "training",
    ):
        super().__init__()
        self.mode = mode # or "inference"
        self.graph = Geometric(original_model, model_inputs)
        self.device = get_device()
        self.storage = RK_Storage(self.device, self.graph.model, self.graph.dict_constants)
        self.construct_op_list()
        self.compile_function()
        
    def construct_op_list(self):
        self.tmp_fwd_op_list = []
        self.tmp_bwd_op_list = []
        self.fwd_op_list = []
        self.bwd_op_list = []

        for kg in self.graph.kgraph_list:
            users = {}
            op_list = []
            for kdn in kg.list_kdn:
                users[kdn.name] = len(kdn.users_real)

            for kcn in kg.list_kcn:
                if "loss" in kcn.name:
                    op_list.append(RunOp(kcn))
                op_list.append(RunOp(kcn))

                for deps in kcn.deps_global:
                    if deps.name not in users:
                        continue
                    if deps not in kcn.deps_fake:
                        users[deps.name] -= 1
                        if users[deps.name] == 0:
                            op_list.append(DelOp(deps))
            
            # debug
            # for op in op_list:
            #     print(f'op: {op}')
        
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

        # debug
        for op in self.fwd_op_list:
            print(f'fwd_op: {op}')

        for op in self.bwd_op_list:
            print(f'bwd_op: {op}')

        list_kdn = []
        for kg in self.graph.kgraph_list:
            list_kdn += kg.list_kdn
        
        self.op_sched = OpSchedule(
            self.fwd_op_list + self.bwd_op_list,
            None,
            self.graph.kgraph_list[0].input_kdn_data,
            self.graph.kgraph_list[0].input_kdn_grad,
            self.graph.output,
            list_kdn,
        )

    def compile_function(self):
        self.compiler = Compiler(self.storage)
        self.fct_list = self.compiler.compile(self.op_sched)
        loss_idx = len(self.fwd_op_list)
        self.fwd_fct_list = self.fct_list[:loss_idx]
        self.bwd_fct_list = self.fct_list[loss_idx:]

    def _exec(self, fct_list):
        for fct in fct_list:
            fct()
    
    def forward(self, *args, **kwargs):
        if self.mode == "inference":
            self.graph.model.eval()
            return self.graph.model(*args, **kwargs)
        
        model_inputs = make_inputs(self.graph.model, args, kwargs)
        for k, v in model_inputs.items():
            print(f'k: {k}, v: {v}')
            self.storage.add_val(k, v)
        
        exec(self.graph.init_code, self.storage.gd, self.storage.ld)

        for kg in self.graph.kgraph_list:
            for kdn in kg.list_kdn:
                tensor_val = torch.empty(
                    0, device=self.device,
                    requires_grad=kdn.info.requires_grad
                )
                self.storage.ld[kdn.main_target] = tensor_val
        
        for l in self.fwd_fct_list:
            self._exec(l)

        return self.storage.get_val(self.graph.output.main_target)

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

def test(model, sample):
    for_rkgb = rkgb.make_all_graphs(model, sample, verbose=False, bool_kg=True)
    kgraph_list = for_rkgb.K_graph_list

    list_sols = []
    fwd_op_list = []
    for kg in kgraph_list:
        op_list = []
        users = {}
        for kdn in kg.list_kdn:
            users[kdn.name] = len(kdn.users_real)

        for kcn in kg.list_kcn:
            if "loss" in kcn.name:
                op_list.append(RunOp(kcn))
            op_list.append(RunOp(kcn))

            for deps in kcn.deps_global:
                if deps.name not in users:
                    continue
                if deps not in kcn.deps_fake:
                    users[deps.name] -= 1
                    if users[deps.name] == 0:
                        op_list.append(DelOp(deps))

        for op in op_list:
            print(f'op: {op}')
        


        # loss_idx = 0
        # for i, op in enumerate(op_list):
        #     if "loss" in op.name:
        #         loss_idx = i
        #         break

        # fwd_sched = OpSchedule(
        #     op_list[:loss_idx+1],
        #     None,
        #     kg.input_kdn_data,
        #     kg.input_kdn_grad,
        #     kg.output_kdn_data,
        #     kg.list_kdn,
        # )

        # list_sols.append(fwd_sched)

    list_kdn = []
    for kg in kgraph_list:
        list_kdn += kg.list_kdn

    # for op in fwd_op_list:
    #     print(f'fwd_op: {op.name}, {op.main_target}')
    #     print(f'main code: {op.main_code}')
    #     print(f'body code: {op.body_code}')
    #     print(f'deps_global: {op.deps_global}')
    #     print(f'deps fake: {op.deps_fake}')
    #     print(f'user_global: {op.users_global}')

    # fwd_op_sched = OpSchedule(
    #     fwd_op_list,
    #     None,
    #     kgraph_list[0].input_kdn_data,
    #     kgraph_list[0].input_kdn_grad,
    #     kgraph_list[-1].output_kdn_data,
    #     list_kdn,
    # )

    # op_sched = OpSchedule(
    #     fwd_op_list,
    #     None,
    #     kgraph_list[0].input_kdn_data,
    #     kgraph_list[0].input_kdn_grad,
    #     kgraph_list[-1].output_kdn_data,
    #     list_kdn,
    # )

    # device = get_device()
    # original_mod = model
    # storage = RK_Storage(device, original_mod, dict_constants)

    # dict_constants = for_rkgb.K_graph_list[0].dict_constants
    # # for k, v in model_inputs.items():
    # #     storage.add_val(k, v)

    # compiler = Compiler(op_sched)

model = SimpleCNN()
sample = [torch.randn(1, 3, 32, 32)]

model = models.resnet18()
sample = torch.randn(1, 3, 224, 224)

print("---  Doing rematerialization with Asuta ----")
# test(model, sample)
for_test = Asuta(model, sample)
# r1 = for_test(*sample)
# r2 = model(*sample)
# if torch.equal(r1, r2):
#     print('---  Rematerialization with Asuta is correct ----')
print('---  Done rematerialization with Asuta ----')