import rkgb.src as rkgb
from rkgb.utils.ast_add_on import ast_to_str
from rkgb.utils.small_fcts import check_attr
import numpy as np
import torch
import warnings

class Graph:
    def __init__(
        self,
        original_model,
        model_inputs,
        verbose=False,
    ):
        super().__init__()
        self.model = original_model
        self.inputs = model_inputs
        self.rkgb_results = rkgb.make_all_graphs(
            original_model, model_inputs, verbose=verbose, bool_kg=True
        )
        self.graph_list = self.rkgb_results.K_graph_list
        self.dict_constants = self.rkgb_results.K_graph_list[0].dict_constants
        self.eq_classes = self.rkgb_results.equivalent_classes
        self.init_code = ast_to_str(self.graph_list[0].init_code)
        self.output = self.graph_list[-1].output_kdn_data

class C_node: # computation node 
    def __init__(self, kcn, alive_datas=set()):
        self.name = kcn.name
        self.time = kcn.time
        self.overhead = kcn.overhead
        self.main_target = kcn.main_target
        self.tensor_targets = kcn.tensor_targets
        self.main_code = kcn.main_code
        self.body_code = kcn.body_code
        self.inplace_code = kcn.inplace_code
        self.deps_fake = [kdn.name for kdn in kcn.deps_fake]
        self.deps_global = [kdn.name for kdn in kcn.deps_global]
        self.users_global = [kdn.name for kdn in kcn.users_global]
        self.alive_datas = alive_datas
        self.type = "Run"
        self.is_rand = kcn.is_rand
        self.is_swap = False
        self.proxy = False
        for kdn in kcn.users:
            if kdn.kdn_type != "data":
                continue
            self.proxy = kdn.info.requires_grad

    def __eq__(self, op2):
        return check_attr(self, op2, ["name"])

    def __str__(self):
        return f"C_node: Run {self.name} {self.main_target}"


class D_node: # data node
    def __init__(self, kdn, proxy=True):
        self.name = kdn.name
        self.kdn_type = kdn.kdn_type
        self.time = 0
        self.save_mem = kdn.mem
        self.main_target = kdn.main_target
        self.tensor_targets = kdn.tensor_targets
        self.all_targets = kdn.all_targets
        self.container_targets = kdn.container_targets
        self.inplace_targets = kdn.inplace_targets
        self.info = kdn.info
        self.type = "Del"
        self.proxy = proxy
        self.includes_phantoms = kdn.includes_phantoms
        self.includes_base = kdn.includes_base
        self.is_swap = False

    def __eq__(self, op2):
        return check_attr(self, op2, ["name"])

    def __str__(self):
        return f"D_node: Del {self.name} {self.main_target}"


class NodeSchedule:
    def __init__(
        self,
        op_list,
        alive_list,
        input_kdn_data,
        input_kdn_grad,
        output_kdn_data,
        list_kdn,
        no_grad=False,
    ):
        self.op_list = op_list
        self.op_name_list = [op.name for op in self.op_list]
        self.alive_list = alive_list
        L = len(op_list)

        self.no_grad = no_grad

        self.input_size = (
            input_kdn_data.main_target,
            input_kdn_data.mem,
        )
        self.output_size = (
            output_kdn_data.main_target,
            output_kdn_data.mem,
        )
        self.kdn_dict = {kdn.name: kdn for kdn in list_kdn}

        # save the del_input op in case needed
        input_kdn = input_kdn_data
        self.del_input_op = D_node(input_kdn, proxy=False)
        self.del_input_idx = L

        list_kdn = list_kdn + [input_kdn_grad, input_kdn_data]
        self.mem_sizes = [kdn.mem for kdn in list_kdn]
        self.kdn_names = [kdn.name for kdn in list_kdn]
        self.kdn_info = {
            kdn.name: kdn.info for kdn in list_kdn
        }  # dict: name->info

        self.is_fwd = True
        # self.get_mem_time()
        # assert self.valid_sched()

    def get_mem_time(self):
        """
        everytime op_list/alive_list are changed, run this to update mem
        """
        L = len(self.op_list)
        self.save = np.zeros(L)
        self.tmp = np.zeros(L)
        input_grad = False
        output_grad = False
        for i, op in enumerate(self.op_list):
            if isinstance(op, C_node):
                self.tmp[i] = op.overhead
                if "bwd" in op.name:
                    self.is_fwd = False
                    # rotor assumes the space for input data but not input grad
                    for kdn_name in op.users_global:
                        if not input_grad and self.input_size[0] in kdn_name:
                            self.tmp[i:] += self.input_size[1]
                            input_grad = True

            self.save[i] += self.alive_list[i][:-2].dot(
                np.array(self.mem_sizes[:-2])
            )  # input kdn is not included
            # if (
            #     not output_grad
            #     and self.alive_list[i][
            #         self.kdn_names.index(self.output_size[0] + " grad")
            #     ]
            # ):
            #     self.save[i:] -= self.output_size[1]
            #     output_grad = True
        self.overhead = max(self.save + self.tmp) - self.save[-1]
        self.time = sum([op.time for op in self.op_list])

    def del_input(self):
        self.op_list.insert(self.del_input_idx, self.del_input_op)
        alive_status = self.alive_list[self.del_input_idx - 1].copy()
        self.alive_list.insert(self.del_input_idx, alive_status)
        for i in range(self.del_input_idx, len(self.op_list)):
            self.alive_list[i][-1] = False

        self.get_mem_time()

