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
        self.graph = Graph(original_model, model_inputs)
        self.device = get_device()
        self.storage = RK_Storage(self.device, self.graph.model, self.graph.dict_constants)
        self.construct_op_list()
        self.compile_function()
        
    def construct_op_list(self):
        self.tmp_fwd_op_list = []
        self.tmp_bwd_op_list = []
        self.fwd_op_list = []
        self.bwd_op_list = []

        for kg in self.graph.graph_list:
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

        for kg in self.graph.graph_list:
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

def verify(model1, model2, inputs, dict_kwargs=None):
    '''
        model1 : original model
        model2 : asuta model 
    '''
    device = torch.device("cpu")
    module = model1.graph.model
    # module = model1
    dict_inputs = rkgb.make_inputs(model2, inputs.to(device), dict_kwargs)
    _dict_inputs = dict()
    for k, v in dict_inputs.items():
        if isinstance(v, torch.Tensor):
            _dict_inputs[k] = v.clone()
        else:
            _dict_inputs[k] = deepcopy(v)

    model1.train()
    model2.train()
    torch.random.manual_seed(0)
    y1 = model1(**_dict_inputs)
    torch.random.manual_seed(0)
    y2 = model2(**dict_inputs)
    same_train = torch.allclose(y1, y2)

    model1.eval()
    model2.eval()
    torch.random.manual_seed(0)
    y1 = model1(**_dict_inputs)
    torch.random.manual_seed(0)
    y2 = model2(**dict_inputs)
    same_eval = torch.allclose(y1, y2)
    if not same_eval:
        print(torch.mean(y1 - y2)/y1)

    same_grad = True
    for n, _ in model2.named_parameters():
        if not torch.allclose(model2.get_parameter(n), module.get_parameter(n)):
            print("Unequal weight found in:", n)
            same_grad = False

        if (
            model2.get_parameter(n).grad != None
            and module.get_parameter(n).grad != None
        ):
            grad1 = module.get_parameter(n).grad
            grad2 = model2.get_parameter(n).grad
            if not torch.allclose(grad1, grad2):
                print("Unequal grad found in:", n)
                print(torch.mean((grad1 - grad2) / grad1))
                same_grad = False

    return same_train, same_eval, same_grad

model = SimpleCNN()
sample = [torch.rand(1, 3, 32, 32)]

model = models.resnet101()
sample = torch.rand(5, 3, 244, 244)
# y = model(sample)

same_train, same_eval, same_grad = verify(Asuta(model, sample), model, sample)
# same_train, same_eval, same_grad = verify(model, model, sample)
if same_train:
    print(f'---  Same training result ----')
if same_eval:
    print(f'---  Same evaluation result ----')
if same_grad:
    print(f'---  Same gradient ----')

print("---  Doing rematerialization with Asuta ----")
# test(model, sample)
# for_test = Asuta(model, sample)
# y = for_test(sample)
# loss = y.mean()
# loss.backward()
# r2 = model(*sample)
# if torch.equal(r1, r2):
#     print('---  Rematerialization with Asuta is correct ----')
print('---  Done rematerialization with Asuta ----')