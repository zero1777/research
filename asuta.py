import rkgb.src as rkgb
import rkgb.src.utils as rkgb_utils
from rkgb.main import make_inputs

import torch
import torch.nn as nn
from copy import deepcopy
import time

from graph import Graph
from utils import *
from node import C_op, D_op, OpSchedule
from compiler import Compiler, RngState, Storage
from logger import Logger

class Asuta(torch.nn.Module):
    def __init__(
        self,
        original_model,
        model_inputs,
    ):
        super().__init__()
        self.graph = Graph(original_model, model_inputs)
        self.device = get_device()
        self.eviction_list = []
        self.eviction_list = ["__7_input data", "__16_input0 data"]
        self.storage = Storage(self.device, self.graph.model, self.graph.dict_constants)
        self.logger = Logger("asuta.log", print_log=True)
        self.pcie_bw = 16 * 1024 * 1024 * 1024 # 16 GB/s
        self.num_evict = 3
        self.construct_op_list()
        self.construct_op_list_v2()
        self.compile_function()

    def construct_op_list(self):
        self.tmp_fwd_op_list = []
        self.tmp_bwd_op_list = []
        self.fwd_op_list = []
        self.bwd_op_list = []
        self.kdn_users_counter = {} # dict: kdn.name -> num of users
        self.kdn_dict = {} # dict: kdn.name -> kdn
        self.kcn_dict = {} # dict: kcn.name -> kcn

        for kg in self.graph.graph_list:
            users = {} # dict: name -> num of users
            op_list = [] # list of C_op and D_op
            alive_datas = set() # current alive datas (Notice that the range of alive datas is only in one k_graph, we will reconstruct it later)

            # initialze
            for kdn in kg.list_kdn:
                self.kdn_dict[kdn.name] = kdn
                self.kdn_users_counter[kdn.name] = len(kdn.users_global)
                users[kdn.name] = len(kdn.users_real)

                self.logger.debug(f'users: {kdn.name}')
                self.logger.debug(f'kdn: {kdn.name}, {[n.name for n in kdn.users_real]}')
                self.logger.debug(f'kdn: {kdn.name}, {[c.name for c in kdn.users_global]}')
            
            for kcn in kg.list_kcn:
                self.kcn_dict[kcn.name] = kcn
                op_list.append(C_op(kcn, alive_datas=alive_datas))
                # "loss" op needs to be executed in both forward and backward, so we need to add it twice
                if "loss" in kcn.name:
                    op_list.append(C_op(kcn, alive_datas=alive_datas))
                for kdn in kcn.users:
                    alive_datas.add(kdn.name)

                self.logger.debug(f'kcn: {kcn.name}, alive_datas: {alive_datas}')

                # update the counter of used kdn (decrement) since kcn is executed
                for deps in kcn.deps_global:
                    if deps.name not in users:
                        continue
                    if deps not in kcn.deps_fake:
                        users[deps.name] -= 1
                        # if the counter is 0, then the kdn is no longer needed
                        if users[deps.name] == 0:
                            alive_datas.remove(deps.name)
                            op_list.append(D_op(deps))
        
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

        self.logger.debug(f'fwd_op: {[op.name for op in self.fwd_op_list]}')
        self.logger.debug(f'bwd_op: {[op.name for op in self.bwd_op_list]}')

        list_kdn = []
        for kg in self.graph.graph_list:
            list_kdn += kg.list_kdn
        
        list_kcn = []
        for kg in self.graph.graph_list:
            list_kcn += kg.list_kcn
        
        self.op_sched = OpSchedule(
            self.fwd_op_list + self.bwd_op_list,
            None,
            self.graph.graph_list[0].input_kdn_data,
            self.graph.graph_list[0].input_kdn_grad,
            self.graph.output,
            list_kdn,
        )

        self.total_memory = 0
        self.data_memory = {}
        for kdn in list_kdn:
            self.data_memory[kdn.name] = kdn.mem
            if "data" in kdn.name:
                self.total_memory += kdn.mem

        self.total_overhead = 0
        self.data_overhead = {}
        self.compute_overhead = {}
        for kcn in list_kcn:
            if "bwd" not in kcn.name and "loss" not in kcn.name:
                kdn_name = kcn.name.replace("fwd_", "")
                kdn_name += " data"
                self.data_overhead[kdn_name] = kcn.time
            self.compute_overhead[kcn.name] = kcn.time
            self.total_overhead += kcn.time
        
        self.logger.info(f'data_memory: {self.data_memory}')
        self.logger.info(f'total_memory: {self.total_memory}')
        self.logger.info(f'data_overhead: {self.data_overhead}')
        self.logger.info(f'compute_overhead: {self.compute_overhead}')
        self.logger.info(f'total_overhead: {self.total_overhead}')

        # self.select_eviction_list()

    def select_eviction_list(self):
        list_kdn = []
        for kg in self.graph.graph_list:
            list_kdn += kg.list_kdn

        self.recompute_cost = {}
        self.swap_cost = {}
        for kdn in list_kdn:
            if "grad" in kdn.name or "phantoms" in kdn.name:
                continue
            self.recompute_cost[kdn.name] = self.data_memory[kdn.name] / self.data_overhead[kdn.name]
            self.swap_cost[kdn.name] = self.data_memory[kdn.name] / self.pcie_bw

        self.sorted_rcost = dict(sorted(self.recompute_cost.items(), key=lambda item: item[1], reverse=True))

        self.logger.info(f'recompute_cost: {self.recompute_cost}')
        self.logger.info(f'swap_cost: {self.swap_cost}')
        self.logger.info(f'sorted_rcost: {self.sorted_rcost}')

        self.eviction_list = list(self.sorted_rcost.keys())[:self.num_evict]
        self.logger.info(f'eviction_list: {self.eviction_list}')
        
    def construct_op_list_v2(self):
        alive_datas = set() # current alive datas
        evict_list = {} # dict: kdn.name -> kcn
        users = {} # dict: name -> num of users

        self.fwd_op_list_v2 = []
        self.bwd_op_list_v2 = []
        
        # build kdn user counts (only for forward)
        for kg in self.graph.graph_list:
            for kdn in kg.list_kdn:
                cnt = 0
                for i in kdn.users_global:
                    if "fwd" in i.name: 
                        cnt += 1 
                users[kdn.name] = cnt


        # forward list
        for op in self.fwd_op_list:
            self.fwd_op_list_v2.append(op)
            if isinstance(op, C_op):
                op.alive_datas = alive_datas.copy()
                for deps_name in op.deps_global:
                    if deps_name not in users:
                        assert deps_name == "sources data"
                        continue

                    users[deps_name] -= 1

                    if deps_name in self.eviction_list:
                        assert "grad" not in deps_name
                        if users[deps_name] == 0:
                            assert len(self.kdn_dict[deps_name].deps) == 1
                            parent_op = [n for n in self.kdn_dict[deps_name].deps]
                            evict_list[deps_name] = parent_op[0]
                            dnode = D_op(self.kdn_dict[deps_name])
                            dnode.is_swap = False
                            self.fwd_op_list_v2.append(dnode)

                for kdn_name in op.users_global:
                    alive_datas.add(kdn_name)

            elif isinstance(op, D_op):
                alive_datas.remove(op.name)
        

        def regen_tensor(kdn_name):
            parent_op = evict_list[kdn_name]
            for deps in parent_op.deps_global:
                if deps.name in evict_list:
                    regen_tensor(deps.name)
            cnode = C_op(parent_op, alive_datas=alive_datas.copy())
            cnode.is_swap = False
            self.bwd_op_list_v2.append(cnode)
            del evict_list[kdn_name]

        # backward list
        for op in self.bwd_op_list:
            if isinstance(op, C_op):
                op.alive_datas = alive_datas.copy()
                if "loss" in op.name:
                    self.bwd_op_list_v2.append(op) 
                    continue

                for user_name in op.users_global:
                    assert "grad" in user_name
                    data_name = user_name.replace("grad", "data")
                    if data_name in evict_list:
                        print(f'need grad: {op.name}, {data_name}')
                        regen_tensor(data_name)
            
                for deps_name in op.deps_global:
                    if deps_name not in op.deps_fake and deps_name in evict_list:
                        print(f'need op: {op.name}, deps_name: {deps_name}, parent: {evict_list[deps_name].name}')
                        regen_tensor(deps_name)

                for kdn_name in op.users_global:
                    alive_datas.add(kdn_name)
                
            elif isinstance(op, D_op):
                alive_datas.remove(op.name)
                if op.name in evict_list:
                    print(f'kdn already in evict {op.name}')
                    continue

            self.bwd_op_list_v2.append(op)

        self.logger.debug(f'fwd_op_list_evict: {[op.name for op in self.fwd_op_list_v2]}')
        self.logger.debug(f'bwd_op_list_evict: {[op.name for op in self.bwd_op_list_v2]}')
            
        list_kdn = []
        for kg in self.graph.graph_list:
            list_kdn += kg.list_kdn
        
        self.op_sched_v2 = OpSchedule(
            self.fwd_op_list_v2 + self.bwd_op_list_v2,
            None,
            self.graph.graph_list[0].input_kdn_data,
            self.graph.graph_list[0].input_kdn_grad,
            self.graph.output,
            list_kdn,
        )

    def compile_function(self):
        self.compiler = Compiler(self.storage)
        # self.fct_list = self.compiler.compile(self.op_sched) # compile op_sched -> list of functions
        # loss_idx = len(self.fwd_op_list)
        self.fct_list = self.compiler.compile(self.op_sched_v2) # compile op_sched -> list of functions
        loss_idx = len(self.fwd_op_list_v2)
        self.fwd_fct_list = self.fct_list[:loss_idx]
        self.bwd_fct_list = self.fct_list[loss_idx:]

        self.logger.debug(f'fwd_fct: {[fct for fct in self.fwd_fct_list]}')
        self.logger.debug(f'bwd_fct: {[fct for fct in self.bwd_fct_list]}')

    def _exec(self, fct_list):
        for fct in fct_list:
            fct()
    
    def forward(self, *args, **kwargs):
        if not self.training:
            self.graph.model.eval()
            return self.graph.model(*args, **kwargs)
        
        # set input data
        # stream = torch.cuda.current_stream(self.device)
        # se = torch.cuda.Event(enable_timing=True)
        model_inputs = make_inputs(self.graph.model, args, kwargs)
        # ee = torch.cuda.Event(enable_timing=True)
        # se.record(stream)
        # ee.record(stream)
        # print(f'forward: {se.elapsed_time(ee)/1000}')
        for k, v in model_inputs.items():
            self.storage.add_val(k, v)
        
        # execute init code
        exec(self.graph.init_code, self.storage.gd, self.storage.ld)
        # start_time = time.time()
        # end_time = time.time()
        # training_time = end_time - start_time
        # print(f'training_time (sec): {training_time}')

        for kg in self.graph.graph_list:
            for kdn in kg.list_kdn:
                tensor_val = torch.empty(
                    0, device=self.device,
                    requires_grad=kdn.info.requires_grad
                )
                self.storage.ld[kdn.main_target] = tensor_val
            

        # execute the generated function list (forward)
        # stream = torch.cuda.current_stream(self.device)
        # for i, l in enumerate(self.fwd_fct_list):
        #     se = torch.cuda.Event(enable_timing=True)
        #     ee = torch.cuda.Event(enable_timing=True)
        #     se.record(stream)
        #     self._exec(l)
        #     ee.record(stream)
        #     torch.cuda.synchronize(self.device)
        #     print(f'forward: {i}, {se.elapsed_time(ee)/1000}, {torch.cuda.max_memory_allocated() - 17553408}, {torch.cuda.memory_allocated() - 17553408}')
            
        # stream = torch.cuda.current_stream(self.device)
        # se = torch.cuda.Event(enable_timing=True)
        # ee = torch.cuda.Event(enable_timing=True)
        # se.record(stream)
        for l in self.fwd_fct_list:  
            self._exec(l)
        # ee.record(stream)
        # torch.cuda.synchronize(self.device)
        # print(f'forward: {se.elapsed_time(ee)/1000}')
        

        return self.storage.get_val(self.graph.output.main_target)
    
    
    def backward(self):
        # execute the generated function list (backward)
        for i, l in enumerate(self.bwd_fct_list):
            self._exec(l)

        # for l in self.bwd_fct_list:
            # self._exec(l)