import rkgb.src as rkgb
import rkgb.src.utils as rkgb_utils
from rkgb.main import make_inputs
import ast

import torch
import torch.nn as nn
from copy import deepcopy
import time

from graph import Graph, C_op, D_op, OpSchedule
from utils import *
# from node import C_op, D_op, OpSchedule
from compiler import Compiler, RngState, Storage
from logger import Logger
    

class Asuta(torch.nn.Module):
    def __init__(
        self,
        original_model,
        model_inputs,
        evict_list = None,
        evict_tensor_mode = None,
    ):
        super().__init__()
        self.graph = Graph(original_model, model_inputs)
        self.device = get_device()
        self.eviction_list = []
        if evict_list is not None:
            self.eviction_list = evict_list
            
        # self.eviction_list = ["__13_input data", "__25_input data", "__28_input data", "__154_input data", "__47_input data", "__59_input data", "__63_input data", "__76_input data", "__80_input data", "__98_input data", "__242_input data", "__254_input data", "__211_input data", "__275_input data", "__293_input data", "__349_input data", "__406_input data", "__316_input data", "__332_input data", "__389_input data", "__121_input data", "__133_input data", "__137_input data", "__178_input data", "__194_input data"]
        # self.eviction_list = ["__540_out data"] # 389, 402
        # self.eviction_list = ["__13_input data",  "__25_input data",  "__28_input data",  "__47_input data",  "__59_input data",  "__63_input data",  "__76_input data",  "__80_input data",  "__98_input data",  "__121_input data",  "__133_input data",  "__137_input data",  "__150_input data",  "__154_input data",  "__166_out data",  "__178_input data",  "__190_input data",  "__194_input data",  "__207_input data",  "__211_input data",  "__223_out data",  "__242_input data",  "__254_input data",  "__258_input data",  "__271_input data",  "__275_input data",  "__293_input data",   "__316_input data",  "__328_input data",  "__332_input data",  "__345_input data",  "__349_input data",  "__373_input data",  "__385_input data",  "__389_input data",  "__402_input data",  "__406_input data",  "__418_out data",  "__430_input data",  "__442_input data",  "__446_input data",  "__459_input data",  "__463_input data" ]
        # self.eviction_list = ["__13_input data",  "__25_input data",  "__28_input data",  "__47_input data",  "__59_input data",  "__63_input data",  "__76_input data",  "__80_input data",  "__98_input data",  "__121_input data",  "__133_input data",  "__137_input data",  "__150_input data",  "__154_input data",    "__178_input data",  "__190_input data",  "__194_input data",  "__207_input data",  "__211_input data",   "__242_input data",  "__254_input data",  "__258_input data",  "__271_input data",  "__275_input data",  "__293_input data",   "__316_input data",  "__328_input data",  "__332_input data",  "__345_input data",  "__349_input data",  "__373_input data",  "__385_input data",  "__389_input data",  "__402_input data",  "__406_input data",    "__430_input data",  "__442_input data",  "__446_input data",  "__459_input data",  "__463_input data",  "__495_input data",  "__507_input data",  "__511_input data",  "__524_input data",  "__528_input data",  "__546_input data",  "__569_input data",  "__581_input data",  "__585_input data",  "__598_input data",  "__602_input data",  "__627_input data",  "__639_input data",  "__643_input data",  "__656_input data",  "__660_input data",   "__684_input data"] # 11.40, 9.14 4.54 4.38 ResNet50
        # self.eviction_list = ["__13_input data",  "__25_input data",  "__28_input data",  "__47_input data",  "__59_input data",  "__63_input data",  "__76_input data",  "__80_input data",  "__98_input data",  "__92_out data",  "__121_input data",  "__133_input data",  "__137_input data",  "__150_input data",  "__154_input data",     "__178_input data",  "__190_input data",  "__194_input data",  "__207_input data",  "__211_input data",  ]
        # self.eviction_list = ["__13_input data",  "__25_input data",  "__28_input data",  "__47_input data",  "__59_input data",  "__63_input data",  "__76_input data",  "__80_input data",  "__98_input data",      "__121_input data",  "__133_input data",  "__137_input data",  "__150_input data",  "__154_input data",   "__178_input data",  "__190_input data",  "__194_input data",  "__207_input data",  "__211_input data",   "__242_input data",  "__254_input data",  "__258_input data",  "__271_input data",  "__275_input data",  "__293_input data",     "__316_input data",  "__328_input data",  "__332_input data",  "__345_input data",  "__349_input data",  "__373_input data",  "__385_input data",  "__389_input data",  "__402_input data",  "__406_input data",    "__430_input data",  "__442_input data",  "__446_input data",  "__459_input data",  "__463_input data",   "__512_input data",  "__524_input data",  "__528_input data",  "__541_input data",  "__545_input data",  "__563_input data",   "__586_input data",  "__598_input data",  "__602_input data",  "__615_input data",  "__619_input data",  "__644_input data",  "__656_input data",  "__660_input data",  ]
        # self.eviction_list = ["__13_input data",  "__25_input data",  "__28_input data",  "__47_input data",  "__59_input data",  "__63_input data",  "__76_input data",  "__80_input data",  "__98_input data",  "__121_input data"] #10.58
        # self.eviction_list = [ "__39_input data", "__47_input data","__54_input data"] # 2.43 # 5.79 "10.44" 11.59 13.79 VGG16
        # self.eviction_list = ["__35_fv data",  "__39_input0 data",  "__172_fv data",]
        # self.eviction_list = ["__35_fv data",  "__39_input0 data",  "__48_x data",  "__62_x data",  "__104_scores data",  "__107_fv data",  "__108_scores data",  "__109_x5 data",  "__110_x6 data",  "__129_x0 data",  "__134_fv data",  "__140_x data",  "__153_x data",  "__145_x data",  "__166_x0 data",  "__171__0 data",  "__172_fv data",  "__182_x data",  "__196_x data",  "__238_scores data",  "__241_fv data",  "__242_scores data",  "__243_x5 data",  "__244_x6 data",  "__263_x0 data",  "__268_fv data",  "__274_x data",  "__287_x data",  "__279_x data",  "__300_x0 data",  "__305__0 data",  "__306_fv data",  "__315_x data",  "__329_x data",  "__371_scores data",  "__374_fv data",  "__375_scores data",  "__376_x5 data",  "__377_x6 data",  "__396_x0 data",  "__401_fv data",  "__407_x data",  "__420_x data",  "__412_x data",  "__433_x0 data",  "__438__0 data",  "__439_fv data",  "__448_x data",  "__462_x data",  "__504_scores data",  "__507_fv data",  "__508_scores data",  "__509_x5 data",  "__510_x6 data",  "__529_x0 data",  "__534_fv data",  "__540_x data",  "__553_x data",  "__545_x data",  "__566_x0 data",  "__571__0 data",  "__572_fv data",  "__582_x data",  "__596_x data",  "__638_scores data",  "__641_fv data",  "__642_scores data",  "__643_x5 data",  "__644_x6 data",  "__663_x0 data",  "__668_fv data",  "__674_x data",  "__687_x data",  "__679_x data",  "__700_x0 data",  "__705__0 data",  "__706_fv data",  "__715_x data",  "__729_x data",  "__771_scores data",  "__774_fv data",  "__775_scores data",  "__776_x5 data",  "__777_x6 data",  "__796_x0 data",  "__801_fv data",  "__807_x data",  "__820_x data",  "__812_x data",  "__833_x0 data",  "__838__0 data",  "__839_fv data",  "__848_x data",  "__862_x data",  "__904_scores data",  "__907_fv data",  "__908_scores data",  "__909_x5 data",  "__910_x6 data",  "__929_x0 data"] GPT2 14.388
        self.storage = Storage(self.device, self.graph.model, self.graph.dict_constants)
        self.logger = Logger("asuta.log", print_log=True)
        self.pcie_bw = 16 * 1024 * 1024 * 1024 # 16 GB/s
        self.num_evict = 100
        self.mode = "r" # s, r
        self.version = "f" # s, f
        self.do_evict = False
        self.peak_memory_usage = [0, 0, 'fw']
        self.evict_tensor_mode = {}

        if evict_tensor_mode is not None:
            self.evict_tensor_mode = evict_tensor_mode

        print(f'{self.mode}, {self.version}')

        # self.eviction_list = ['__8_fv data', '__24__0 data', '__27_scale data', '__38_fv data', '__41_fv data', '__44_fv data', '__45_attn_scores data', '__47_attn_scores0 data', '__50_input data', '__52_fv data', '__53_fv data', '__58__0 data', '__61_scale data', '__72_fv data', '__75_fv data', '__78_fv data', '__79_attn_scores data', '__81_attn_scores0 data', '__84_input data', '__86_fv data', '__87_fv data', '__93__0 data', '__96_scale data', '__107_fv data', '__110_fv data', '__113_fv data', '__114_attn_scores data', '__116_attn_scores0 data', '__119_input data', '__121_fv data', '__122_fv data', '__127__0 data', '__130_scale data', '__141_fv data', '__144_fv data', '__147_fv data', '__148_attn_scores data', '__150_attn_scores0 data', '__153_input data', '__155_fv data', '__156_fv data', '__161__0 data', '__164_scale data', '__175_fv data', '__178_fv data', '__181_fv data', '__182_attn_scores data', '__184_attn_scores0 data', '__187_input data', '__189_fv data', '__190_fv data', '__196__0 data', '__199_scale data', '__210_fv data', '__213_fv data', '__216_fv data', '__217_attn_scores data', '__219_attn_scores0 data', '__222_input data', '__224_fv data', '__225_fv data', '__230__0 data', '__233_scale data', '__244_fv data', '__247_fv data', '__250_fv data', '__251_attn_scores data', '__253_attn_scores0 data', '__256_input data', '__258_fv data', '__259_fv data', '__264__0 data', '__267_scale data', '__278_fv data', '__281_fv data', '__284_fv data', '__285_attn_scores data', '__287_attn_scores0 data', '__290_input data', '__292_fv data', '__293_fv data', '__295__0 data', '__298_scale data']
        # self.eviction_list = []
        # self.evict_tensor_mode = {'__8_fv data': 'swap', '__24__0 data': 'swap', '__27_scale data': 'swap', '__38_fv data': 'swap', '__41_fv data': 'swap', '__44_fv data': 'swap', '__45_attn_scores data': 'swap', '__47_attn_scores0 data': 'swap', '__50_input data': 'swap', '__52_fv data': 'swap', '__53_fv data': 'swap', '__58__0 data': 'swap', '__61_scale data': 'swap', '__72_fv data': 'swap', '__75_fv data': 'swap', '__78_fv data': 'swap', '__79_attn_scores data': 'swap', '__81_attn_scores0 data': 'swap', '__84_input data': 'swap', '__86_fv data': 'swap', '__87_fv data': 'swap', '__93__0 data': 'swap', '__96_scale data': 'swap', '__107_fv data': 'swap', '__110_fv data': 'swap', '__113_fv data': 'swap', '__114_attn_scores data': 'swap', '__116_attn_scores0 data': 'swap', '__119_input data': 'swap', '__121_fv data': 'swap', '__122_fv data': 'swap', '__127__0 data': 'swap', '__130_scale data': 'swap', '__141_fv data': 'swap', '__144_fv data': 'swap', '__147_fv data': 'swap', '__148_attn_scores data': 'swap', '__150_attn_scores0 data': 'swap', '__153_input data': 'swap', '__155_fv data': 'swap', '__156_fv data': 'swap', '__161__0 data': 'swap', '__164_scale data': 'swap', '__175_fv data': 'swap', '__178_fv data': 'swap', '__181_fv data': 'swap', '__182_attn_scores data': 'swap', '__184_attn_scores0 data': 'swap', '__187_input data': 'swap', '__189_fv data': 'swap', '__190_fv data': 'swap', '__196__0 data': 'swap', '__199_scale data': 'swap', '__210_fv data': 'swap', '__213_fv data': 'swap', '__216_fv data': 'swap', '__217_attn_scores data': 'swap', '__219_attn_scores0 data': 'swap', '__222_input data': 'swap', '__224_fv data': 'swap', '__225_fv data': 'swap', '__230__0 data': 'swap', '__233_scale data': 'swap', '__244_fv data': 'swap', '__247_fv data': 'swap', '__250_fv data': 'swap', '__251_attn_scores data': 'swap', '__253_attn_scores0 data': 'swap', '__256_input data': 'swap', '__258_fv data': 'swap', '__259_fv data': 'swap', '__264__0 data': 'swap', '__267_scale data': 'swap', '__278_fv data': 'swap', '__281_fv data': 'swap', '__284_fv data': 'swap', '__285_attn_scores data': 'swap', '__287_attn_scores0 data': 'swap', '__290_input data': 'swap', '__292_fv data': 'swap', '__293_fv data': 'swap', '__295__0 data': 'swap', '__298_scale data': 'swap'}
        
        
        self.gen_op_list()
        # self.gen_op_list_evict()
        self.compile_function(evict=False)

        # mem_cnt = 0
        # print(f'eviction_list: ', end="")
        # for op in self.eviction_list:
        #     # print(f'({op}, {self.data_memory[op]})', end=" ")
        #     print(f'\"{op}\", ', end=" ")
        #     mem_cnt += self.data_memory[op]
        # print(f'\nmem_cnt: {mem_cnt}')

    def build(self):
        self.gen_op_list_evict()
        self.compile_function(evict=True)

    def gen_op_list(self):
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

        # print(f'data memory: {self.data_memory}')
        # print(f'data overhead: {self.data_overhead}')
        # print(f'compute overhead: {self.compute_overhead}')
        # print(f'total overhead: {self.total_overhead}')

        if self.do_evict:
            self.select_eviction_list()
            
        # self.evict_tensor_mode = {}
        # for tensor in self.eviction_list:
        #     if self.mode == "s":
        #         self.evict_tensor_mode[tensor] = "swap"
        #     else :
        #         self.evict_tensor_mode[tensor] = "recompute"
        # change_mode = ["__524_input data", "__700_input data", "__271_input data","__76_input data"]
        # change_mode = []
        # for tensor in change_mode:
        #     self.evict_tensor_mode[tensor] = "swap"

    def select_eviction_list(self):
        list_kdn = []
        for kg in self.graph.graph_list:
            list_kdn += kg.list_kdn

        self.recompute_cost = {}
        self.swap_cost = {}
        cnt = 0
        for kdn in list_kdn:
            if "grad" in kdn.name or "phantoms" in kdn.name:
                continue
            cnt += 1
            self.recompute_cost[kdn.name] = self.data_memory[kdn.name] / self.data_overhead[kdn.name]
            self.swap_cost[kdn.name] = self.data_memory[kdn.name] / self.pcie_bw

        # print(f'cnt: {cnt}')

        self.sorted_rcost = dict(sorted(self.swap_cost.items(), key=lambda item: item[1], reverse=True))
        # print(f'sorted_rcost: {self.sorted_rcost}')

        self.logger.info(f'recompute_cost: {self.recompute_cost}')
        self.logger.info(f'swap_cost: {self.swap_cost}')
        self.logger.info(f'sorted_rcost: {self.sorted_rcost}')

        
        # self.eviction_list = list(self.sorted_rcost.keys())[:self.num_evict]
        self.eviction_list = list(self.swap_cost.keys())[:self.num_evict]
        self.logger.info(f'eviction_list: {self.eviction_list}')
        
    def gen_op_list_evict(self):
        alive_datas = set() # current alive datas
        evict_list = {} # dict: kdn.name -> kcn
        users = {} # dict: name -> num of users
        delete_after = {} # dict: c_op name -> list of d_op (to be deleted after)

        self.fwd_op_list_evict = []
        self.bwd_op_list_evict = []
        
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
            if isinstance(op, C_op):
                self.fwd_op_list_evict.append(op)
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
                            # dnode.is_swap = True if self.mode == "s" else False 
                            dnode.is_swap = True if self.evict_tensor_mode[deps_name] == "swap" else False
                            self.fwd_op_list_evict.append(dnode)

                for kdn_name in op.users_global:
                    alive_datas.add(kdn_name)
                    
            elif isinstance(op, D_op):
                self.fwd_op_list_evict.append(op)
                alive_datas.remove(op.name)
                
            # elif isinstance(op, D_op):
            #     in_evict_list = False
            #     for kcn in op.users_real:
            #         if in_evict_list:
            #             break
            #         for kdn_tmp in kcn.users:
            #             if kdn_tmp.name in self.eviction_list:
            #                 in_evict_list = True
            #                 break

            #     if not in_evict_list:
            #         self.fwd_op_list_evict.append(op)
            #         alive_datas.remove(op.name)
            #     else:
            #         c_op_name = "bwd_" + op.name.replace(" data", "")
            #         # print(f'c_op_name: {c_op_name}, {op.name}')
            #         delete_after[c_op_name] = op 

            # elif isinstance(op, D_op):
            #     in_evict_list = False
            #     kdn_name = None
            #     for kcn in op.users_real:
            #         kdn_tmp = list(kcn.users)[0]
            #         # print(f'kcn: {kcn.name}, {[n.name for n in kcn.users]}')
            #         if kdn_tmp.name in self.eviction_list:
            #             print(f'kcn: {kcn.name}, {kdn_tmp.name}')
            #             in_evict_list = True
            #             kdn_name = kdn_tmp.name
            #             break

            #     if not in_evict_list:
            #         self.fwd_op_list_evict.append(op)
            #         alive_datas.remove(op.name)
            #     else:
            #         c_op_name = "bwd_" + op.name.replace(" data", "")
            #         # print(f'c_op_name: {c_op_name}, {op.name}')
            #         delete_after[c_op_name] = op
        

        def regen_tensor(kdn_name):
            parent_op = evict_list[kdn_name]
            if self.evict_tensor_mode[kdn_name] == "recompute":
                # print(f'recompute: {kdn_name}')
                for deps in parent_op.deps_global:
                    if deps.name in evict_list:
                        regen_tensor(deps.name)
            cnode = C_op(parent_op, alive_datas=alive_datas.copy())
            # cnode.is_swap = True if self.mode == "s" else False
            cnode.is_swap = True if self.evict_tensor_mode[kdn_name] == "swap" else False
            self.bwd_op_list_evict.append(cnode)
            del evict_list[kdn_name]
            

        # backward list
        for op in self.bwd_op_list:
            if isinstance(op, C_op):
                op.alive_datas = alive_datas.copy()
                if "loss" in op.name:
                    self.bwd_op_list_evict.append(op) 
                    continue

                # print(f'op: {op.name}')
                for user_name in op.users_global:
                    # print(f'user_name: {user_name}')
                    assert "grad" in user_name
                    data_name = user_name.replace("grad", "data")
                    if data_name in evict_list:
                        print(f'need grad: {op.name}, {data_name}')
                        regen_tensor(data_name)
            
                for deps_name in op.deps_global:
                    # print(f'deps_name: {deps_name}')
                    if deps_name not in op.deps_fake and deps_name in evict_list:
                        print(f'need op: {op.name}, deps_name: {deps_name}, parent: {evict_list[deps_name].name}')
                        regen_tensor(deps_name)

                for kdn_name in op.users_global:
                    alive_datas.add(kdn_name)
                
            elif isinstance(op, D_op):
                alive_datas.remove(op.name)
                if op.name in evict_list:
                    # print(f'kdn already in evict {op.name}')
                    continue

            self.bwd_op_list_evict.append(op)
            if op.name in delete_after.keys():
                self.bwd_op_list_evict.append(delete_after[op.name])
            
        self.logger.debug(f'fwd_op_list_evict: {[op.name for op in self.fwd_op_list_evict]}')
        self.logger.debug(f'bwd_op_list_evict: {[op.name for op in self.bwd_op_list_evict]}')
        # print(f'fwd_op_list_evict: {[[i, op.name] for i, op in enumerate(self.fwd_op_list_evict)]}')
        # print(f'\n')
        # print(f'bwd_op_list_evict: {[[i, op.name] for i, op in enumerate(self.bwd_op_list_evict)]}')
        # print(f'bwd_op_list_evict: {[[i, op.name] for i, op in enumerate(self.bwd_op_list_evict)]}')
            
        list_kdn = []
        for kg in self.graph.graph_list:
            list_kdn += kg.list_kdn
        
        self.op_sched_evict = OpSchedule(
            self.fwd_op_list_evict + self.bwd_op_list_evict,
            None,
            self.graph.graph_list[0].input_kdn_data,
            self.graph.graph_list[0].input_kdn_grad,
            self.graph.output,
            list_kdn,
        )

    def gen_profile_list(self):
        alive_datas = set() # current alive datas
        users = {} # dict: name -> num of users

        self.fwd_profile_op_list = []
        self.bwd_profile_op_list = []
        self.idx_kdn_dict = {} # dict: idx -> kdn
        
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
            self.fwd_profile_op_list.append(op)
            if isinstance(op, C_op):
                # if "loss" not in op.name:
                    # print(f'op: {op.name}, {len(self.fwd_profile_op_list)-1}')
                    # self.idx_kdn_dict[len(self.fwd_profile_op_list)-1] = op
                self.idx_kdn_dict[len(self.fwd_profile_op_list)-1] = op
                    # print(f'op: {op.main_code}')
                # print(f'op: {op.name}')
                op.alive_datas = alive_datas.copy()
                for deps_name in op.deps_global:
                    if deps_name not in users:
                        assert deps_name == "sources data"
                        continue

                    users[deps_name] -= 1

                    if users[deps_name] == 0:
                        assert len(self.kdn_dict[deps_name].deps) == 1
                        dnode = D_op(self.kdn_dict[deps_name])
                        dnode.is_swap = False
                        self.fwd_profile_op_list.append(dnode)

                for kdn_name in op.users_global:
                    alive_datas.add(kdn_name)

            elif isinstance(op, D_op):
                alive_datas.remove(op.name) 
        
        for op in self.bwd_op_list:
            if isinstance(op, C_op):
                op.alive_datas = alive_datas.copy()
                self.bwd_profile_op_list.append(op) 
                if "loss" in op.name:
                    continue

                for kdn_name in op.users_global:
                    alive_datas.add(kdn_name)
                
            elif isinstance(op, D_op):
                alive_datas.remove(op.name)

        list_kdn = []
        for kg in self.graph.graph_list:
            list_kdn += kg.list_kdn
            
            
        self.op_sched_profile = OpSchedule(
            self.fwd_profile_op_list + self.bwd_profile_op_list,
            None,
            self.graph.graph_list[0].input_kdn_data,
            self.graph.graph_list[0].input_kdn_grad,
            self.graph.output,
            list_kdn,
        )

        p_storage = Storage(self.device, self.graph.model, self.graph.dict_constants)
        pcompiler = Compiler(p_storage)
        self.pstorage = Storage(self.device, self.graph.model, self.graph.dict_constants)

        _, exec_list, _, _ = pcompiler.compile(self.op_sched_profile)
        loss_idx = len(self.fwd_profile_op_list)
        fwd_exec_list = exec_list[:loss_idx+2]
        
        self.compile_list = []
        # for i, code in enumerate(fwd_exec_list):
        #     if i in self.idx_kdn_dict:
        #         print(f'code: {code}')
        #     else:
        #         print(f'code_222: {code}')
        for code in fwd_exec_list:
            # print(f'code: {code}')
            self.compile_list.append(
                compile(ast.parse("\n".join(code)), "", "exec")
            )
                
    def compile_function(self, evict):
        self.exec_list = []
        self.fwd_code = []
        self.fwd_compile_code = []
        self.bwd_code = []
        self.bwd_compile_code = []

        self.compiler = Compiler(self.storage)

        if not evict:
            self.fct_list, self.exec_list, self.fwd_code, self.bwd_code = self.compiler.compile(self.op_sched) # compile op_sched -> list of functions
            loss_idx = len(self.fwd_op_list)
        else:
            self.fct_list, self.exec_list, self.fwd_code, self.bwd_code = self.compiler.compile(self.op_sched_evict) # compile op_sched -> list of functions
            loss_idx = len(self.fwd_op_list_evict)

        self.fwd_fct_list = self.fct_list[:loss_idx]
        self.bwd_fct_list = self.fct_list[loss_idx:]

        self.fwd_exec_list = self.exec_list[:loss_idx+2]
        self.bwd_exec_list = self.exec_list[loss_idx+2:]

        # for code_list in self.fwd_code:
        #     # print(code_list)
        #     self.fwd_compile_code.append(
        #         compile(ast.parse("\n".join(code_list)), "", "exec")
        #     )

        # for code_list in self.bwd_code:
        #     # print(code_list)
        #     self.bwd_compile_code.append(
        #         compile(ast.parse("\n".join(code_list)), "", "exec")
        #     )

        tt = 1
        for code_list in self.fwd_exec_list:
            # print(f'code2: {code_list}')
            self.fwd_compile_code.append(
                compile(ast.parse("\n".join(code_list)), "", "exec")
            )
        
        # print("---bwd---")
        for code_list in self.bwd_exec_list:
            # print(code_list)
            self.bwd_compile_code.append(
                compile(ast.parse("\n".join(code_list)), "", "exec")
            )


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
        model_inputs = make_inputs(self.graph.model, args, kwargs)

        for k, v in model_inputs.items():
            self.storage.add_val(k, v)
        
        # execute init code
        exec(self.graph.init_code, self.storage.gd, self.storage.ld)


        for kg in self.graph.graph_list:
            for kdn in kg.list_kdn:
                tensor_val = torch.empty(
                    0, device=self.device,
                    requires_grad=kdn.info.requires_grad,
                )
                self.storage.ld[kdn.main_target] = tensor_val

                
        if self.version == "s":
            for l in self.fwd_fct_list:  
                self._exec(l)

        else :
            tt = 0
            for code in self.fwd_compile_code:
                exec(code, self.storage.gd, self.storage.ld)
                max_mem = torch.cuda.max_memory_allocated()/1000/1000/1000
                alloc_mem = torch.cuda.memory_allocated()/1000/1000/1000
                if self.peak_memory_usage[0] < max_mem:
                    self.peak_memory_usage = [max_mem, tt, 'fw']
                # print(f'forward: {tt}, {alloc_mem}, {max_mem}')
                tt += 1
        

        return self.storage.get_val(self.graph.output.main_target)
    
    
    def backward(self):
        # execute the generated function list (backward)

        if self.version == "s":
            for i, l in enumerate(self.bwd_fct_list):
                # print(f'backward: {i}')
                self._exec(l)
        else:
            tt = 0
            for code in self.bwd_compile_code:
                exec(code, self.storage.gd, self.storage.ld)
                max_mem = torch.cuda.max_memory_allocated()/1000/1000/1000
                alloc_mem = torch.cuda.memory_allocated()/1000/1000/1000
                if self.peak_memory_usage[0] < max_mem:
                    self.peak_memory_usage = [max_mem, tt, 'bw']
                # print(f'backward: {tt}, {alloc_mem}, {max_mem}')
                tt += 1

        # for l in self.bwd_fct_list:
        #     self._exec(l)
        
    
    
    
    