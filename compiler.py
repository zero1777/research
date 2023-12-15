from rkgb.utils.ast_add_on import make_str_assign, make_str_list_assign
import torch
import numpy as np
from node import D_op 

# region Define Register Hooks

# TODO: modify fct_get_pack / fct_get_unpack functions to simply return x

def fct_get_pack(storage, no_save_list, sanity_check=False):
    # no_save_list contains a list of names
    def pack(x):
        for i, c in enumerate(no_save_list):
            if storage.ld[c].data_ptr() == x.data_ptr():
                if sanity_check:
                    assert torch.equal(
                        storage.ld[c].data.as_strided_(
                            x.shape, x.stride(), x.storage_offset()
                        ),
                        x,
                    )
                return (
                    c,
                    x.shape,
                    x.stride(),
                    x.storage_offset(),
                    # x.clone(),
                )
        return x

    return pack


def fct_get_unpack(storage):
    def unpack(x):
        if isinstance(x, tuple):
            return storage.ld[x[0]].data.as_strided_(*x[1:4])

        return x

    return unpack

    # endregion


# region Basic Functions


def fct_get_shapes(storage, tensor_name):
    def fct():
        storage.shapes[tensor_name] = storage.ld[tensor_name].shape
        storage.dtypes[tensor_name] = storage.ld[tensor_name].dtype

    return fct


def fct_get_rng_state(storage, op_name):
    def fct():
        storage.rng_state.get(op_name)

    return fct


def fct_restore_rng_state(storage, op_name):
    def fct():
        storage.rng_state.restore(op_name)

    return fct


def fct_run_forward_no_grad(storage, code):
    def fct():
        with torch.no_grad():
            exec(code, storage.gd, storage.ld)

    return fct


def fct_run_forward_with_grad(storage, code, no_save_list=[]):
    def fct():
        with torch.autograd.graph.saved_tensors_hooks(
            fct_get_pack(storage, no_save_list), fct_get_unpack(storage)
        ):
            exec(code, storage.gd, storage.ld)

    return fct


def fct_run_inplace(storage, inplace_code):
    def fct():
        exec(inplace_code, storage.gd, storage.ld)

    return fct


def fct_run_detach(storage, tensor_name):
    def fct():
        storage.ld[tensor_name].data = storage.ld[f"_{tensor_name}"].data

    return fct


def fct_assign_proxy(storage, tensor_name):
    def fct():
        storage.ld[f"_{tensor_name}"] = storage.ld[tensor_name]

    return fct


def fct_requires_grad(storage, tensor_name):
    def fct():
        storage.ld[tensor_name].requires_grad_()

    return fct


def fct_run_backward(storage, tensor_name, retain_graph):
    def fct():
        storage.ld[f"_{tensor_name}"].backward(
            storage.ld[tensor_name].grad, retain_graph=retain_graph
        )

    return fct


def fct_run_backward_with_inputs(
    storage, tensor_name, retain_graph, input_names
):
    inputs = [storage.ld[name] for name in input_names]

    def fct():
        storage.ld[f"_{tensor_name}"].backward(
            storage.ld[tensor_name].grad,
            inputs=inputs,
            retain_graph=retain_graph,
        )

    return fct


def fct_generate_fake_data(storage, tensor_name):
    def fct():
        m = (
            storage.gd["cmeta"]
            if storage.dtypes[tensor_name].is_complex
            else storage.gd["meta"]
        )
        x = m.expand(np.prod(storage.shapes[tensor_name]))
        storage.ld[tensor_name].data = x.view(storage.shapes[tensor_name])

    return fct


def fct_del_tensor_data(storage, tensor_name):
    def fct():
        storage.ld[tensor_name].data = torch.empty(
            0, device=storage.gd["device"]
        )

    return fct


def fct_del_tensor_base(storage, tensor_name):
    def fct():
        storage.ld[f"_{tensor_name}"]._base.data = torch.empty(
            0, device=storage.gd["device"]
        )

    return fct


def fct_del_tensor_grad(storage, tensor_name):
    def fct():
        storage.ld[tensor_name].grad = None

    return fct


def fct_del_var(storage, var_name):
    def fct():
        storage.ld[var_name] = None

    return fct

# TODO: Currently, the swapin and swapout functions are synchronous. Hope to make them asynchronous in the future
def fct_swapin(storage, tensor_name, swap_stream):
    def fct():
        with torch.cuda.stream(swap_stream):
            storage.ld[tensor_name].data = storage.cpu_ld[tensor_name].data.cuda(non_blocking=False)
            storage.ld[f"_{tensor_name}"].data = storage.ld[tensor_name].data

    return fct

def fct_swapout(storage, tensor_name, swap_stream):
    def fct():
        with torch.cuda.stream(swap_stream):
            storage.cpu_ld[tensor_name] = torch.empty(storage.ld[tensor_name].size(), device="cpu")
            storage.cpu_ld[tensor_name].copy_(storage.ld[tensor_name], non_blocking=False)
            storage.cpu_ld[tensor_name] = storage.cpu_ld[tensor_name].detach().requires_grad_(True)

    return fct

    # endregion


class RngState:
    def __init__(self):
        self.cpu_states = {}
        self.gpu_states = {}

    def get(self, op_name):
        if op_name not in self.cpu_states.keys():
            self.cpu_states[op_name] = torch.get_rng_state()
            self.gpu_states[op_name] = torch.cuda.get_rng_state()

    def restore(self, op_name):
        # pass
        torch.set_rng_state(self.cpu_states[op_name])
        torch.cuda.set_rng_state(self.gpu_states[op_name])


class Storage:
    def __init__(self, device, nn_mod, dict_constants):
        self.gd = {
            **globals(),
            **dict_constants,
            # "original_mod": nn_mod,
            "self": nn_mod,
            "device": device,
            "torch": torch,
            "meta": torch.ones(1).to(device),
            "cmeta": torch.view_as_complex(torch.ones(2)).to(device),
        }
        self.ld = {}
        self.shapes = dict()
        self.dtypes = dict()
        self.rng_state = RngState() # rng (random number generator), used for producing same random numbers
        self.cpu_ld = {}
        self.gd["shapes"] = {}
        self.gd["rng_state"] = RngState()

    def add_val(self, val, x):
        self.ld[val] = x

    def get_val(self, val):
        try:
            return self.ld[val]
        except:
            try:
                return self.gd[val]
            except:
                raise Exception(f"{val} not in the storage")


class Compiler:
    """
    The compiler takes the full operation schedule as input,
    return the lists of Python functions.
    Each list corresponds to one operation.
    """

    def __init__(self, storage):
        self.storage = storage
        self.shapes = storage.shapes
        self.device = self.storage.gd["device"]
        self.swap_stream = torch.cuda.Stream()

    def is_alive(self, op, kdn_name):
        # if kdn_name in self.op_sched.kdn_names:
        #     return self.op_sched.alive_list[i][
        #         self.op_sched.kdn_names.index(kdn_name)
        #     ]

        if kdn_name in self.op_sched.kdn_names:
            return kdn_name in op.alive_datas
        else:
            return True

    # TODO: find_next_idx function deleted
    def find_next_idx(l, target, i):
        return i + l[i:].index(target)

    def compile_fwd(self, op, i):
        if "loss" in op.main_target:
            return [fct_run_forward_no_grad(self.storage, "")]

        not_first = op.name in self.op_sched.op_name_list[:i]

        if not op.proxy:
            last_before_bwd = False
        else:
            next_bwd_idx = i + self.op_sched.op_name_list[i:].index(
                op.name.replace("fwd", "bwd")
            )
            last_before_bwd = not (
                op.name in self.op_sched.op_name_list[i + 1 : next_bwd_idx]
            )
        r = []

        if op.is_rand:
            if not_first:
                r.append(fct_restore_rng_state(self.storage, op.name))
            else:
                r.append(fct_get_rng_state(self.storage, op.name))

        # compile inplace code
        inplace_code = make_str_list_assign(
            op.inplace_code, force_special_kwargs=not_first
        )

        # compile body code
        body_code = ""
        for bc in op.body_code:
            suffix = ""
            if not_first and (bc[0] in op.tensor_targets):
                suffix = ".data"
            body_code += (
                make_str_assign(bc, suffix=suffix, force_special_kwargs=not_first)
                + "\n"
            )

        # compile main code
        suffix = ""
        main_code = (
            make_str_assign(
                op.main_code, suffix=suffix, force_special_kwargs=not_first
            )
            + "\n"
        )
        main_code = main_code.replace(op.main_target, f"_{op.main_target}")

        # TODO: last_before_bwd needs to be True to avoid forward pass being executed in no_grad
        last_before_bwd = True

        if not last_before_bwd:
            for target in op.tensor_targets:
                inplace_code = inplace_code.replace(target, "_" + target)
            r.append(
                fct_run_forward_no_grad(
                    self.storage, main_code.replace("self", "original_mod"),
                )
            )
        else:
            # TODO: no_save_list needed ?
            no_save_list = []
            candidates = list(op.deps_global) + list(op.users_global)
            for kdn_name in candidates:
                if kdn_name in self.op_sched.op_name_list[i:next_bwd_idx]:
                    no_save_list.append(kdn_name.split(" ")[0])

            for target in op.tensor_targets:
                inplace_code = inplace_code.replace(target, "_" + target)

            # run main code
            r.append(
                fct_run_forward_with_grad(
                    self.storage,
                    main_code.replace("self", "original_mod"),
                    no_save_list=no_save_list,
                )
            )

        # run inplace code
        r.append(
            fct_run_forward_with_grad(
                self.storage, inplace_code.replace("self", "original_mod"),
            )
        )
        r.append(
            fct_run_detach(
                self.storage, op.main_target
            )
        )
        # run body code
        r.append(
            fct_run_forward_with_grad(
                self.storage, body_code.replace("self", "original_mod")
            )
        )

        # get the shape of tensors
        if not not_first:
            r.append(fct_get_shapes(self.storage, f"_{op.main_target}"))
            for target in op.tensor_targets:
                r.append(fct_get_shapes(self.storage, target))

        
        
        return r

    def compile_bwd(self, op, i):
        not_first = op.name in self.op_sched.op_name_list[:i]
        last = not (op.name in self.op_sched.op_name_list[i + 1 :])
        # TODO: normally, the bwd operation only appears once in the schedule. Hence, not_first is always False and last is always True

        r = []
        r2 = []

        if op.is_rand:
            if not_first:
                r.append(fct_restore_rng_state(self.storage, op.name))
            else:
                r.append(fct_get_rng_state(self.storage, op.name))

        temporary_tensor_names = [
            kdn_name.split(" ")[0] 
            for kdn_name in op.deps_fake
            if not self.is_alive(op, kdn_name)
        ]
        
        if op.main_target in temporary_tensor_names:
            temporary_tensor_names.append(f"_{op.main_target}")
        for tensor_name in temporary_tensor_names:
            r.append(fct_generate_fake_data(self.storage, tensor_name))
            r2.append(fct_del_tensor_data(self.storage, tensor_name))

        # print(f'{op.name} {temporary_tensor_names}')

        if not_first:
            prev_i = i - self.op_sched.op_name_list[:i][::-1].index(op.name) - 1
            input_names = []
            for kdn_name in op.users_global:
                if f"del {kdn_name}" in self.op_sched.op_name_list[prev_i:i]:
                    input_names.append(kdn_name.split(" ")[0])
            r.append(
                fct_run_backward_with_inputs(
                    self.storage,
                    op.main_target,
                    retain_graph=(not last),
                    input_names=input_names,
                )
            )
        else:
            r.append(
                fct_run_backward(
                    self.storage, op.main_target, retain_graph=(not last)
                )
            )

        return r + r2

    def compile_del_data(self, op):
        r = []
        r.append(fct_del_tensor_data(self.storage, op.main_target))
        if op.info is not None and op.info.requires_grad:
            r.append(fct_del_tensor_data(self.storage, f"_{op.main_target}"))
        if op.includes_base:
            r.append(fct_del_tensor_base(self.storage, op.main_target))
        for v in op.tensor_targets:
            r.append(fct_del_tensor_data(self.storage, v))
        for v in op.container_targets:
            r.append(fct_del_var(self.storage, v))

        return r

    def compile_del_grad(self, op):
        r = []
        r.append(fct_del_tensor_grad(self.storage, op.main_target))

        return r

    def compile_swapin(self, op):
        r = []
        r.append(fct_swapin(self.storage, op.main_target, self.swap_stream))

        return r

    def compile_swapout(self, op):
        r = []
        r.append(fct_swapout(self.storage, op.main_target, self.swap_stream))
        r.append(fct_del_tensor_data(self.storage, op.main_target))
        r.append(fct_del_tensor_data(self.storage, f"_{op.main_target}"))
        
        return r

    def compile_fwd2(self, op, i):
        not_first = op.name in self.op_sched.op_name_list[:i]

        if not op.proxy:
            last_before_bwd = False
        else:
            next_bwd_idx = i + self.op_sched.op_name_list[i:].index(
                op.name.replace("fwd", "bwd")
            )
            last_before_bwd = not (
                op.name in self.op_sched.op_name_list[i + 1 : next_bwd_idx]
            )

        r = []
        
        suffix = ""
        if not_first and not op.proxy and "loss" not in op.name:
            suffix = ".data"
        
        code = (
            make_str_assign(
                op.main_code, suffix=suffix, force_special_kwargs=not_first
            )
            + "\n"
        )

        code += (
            make_str_list_assign(
                op.inplace_code, force_special_kwargs=not_first
            )
            + "\n"
        )

        if op.proxy:
            mt = op.main_target
            for target in op.tensor_targets:
                code = code.replace(target, "_" + target)
            if not_first:
                code += f"{mt}.data = _{mt}.data;\n"
            else:
                code += (
                    f"{mt} = _{mt}.detach();{mt}.requires_grad_();\n"
                )

        # compile body code
        for bc in op.body_code:
            suffix = ""
            if not_first and (bc[0] in op.tensor_targets):
                suffix = ".data"
            code += (
                make_str_assign(bc, suffix=suffix, force_special_kwargs=not_first)
                + "\n"
            )

        # get the shape of tensors
        if not not_first:
            for target in op.tensor_targets:
                if "loss" not in target:
                    code += f"shapes['{target}'] = {target}.shape;"
            for phantom_name in op.phantom_names:
                code += (
                    f"shapes['{phantom_name}'] = _{phantom_name}.shape;"
                )

        # rand
        if op.is_rand:
            code = f"rng_state.get('{op.name}');rng_state.restore('{op.name}')\n{code}"

        # print(f'{i}: {op.name}')
        # print(code)

        return [code]
    
    def _generate_fake_data(self, mt, is_self=False):
        # return code for generate the target fake tensor (only for data/grad)
        prep_code = ""
        after_code = ""

        target_tensor = f"meta.clone().expand(np.prod(shapes['{mt}']))"
        prep_code += f"{mt}.data = {target_tensor}.view(shapes['{mt}']);"

        # for v in kdn.tensor_targets:
        after_code += f"{mt}.data = torch.empty(0,device=device); "

        if is_self:
            prep_code += (
                f"_{mt}.data = {target_tensor}.view(shapes['{mt}']);"
            )
            after_code += f"_{mt}.data = torch.empty(0,device=device);"

        return prep_code, after_code

    def compile_bwd2(self, op, i):
        mt = op.main_target
        not_first = op.name in self.op_sched.op_name_list[:i]
        last = not (op.name in self.op_sched.op_name_list[i + 1 :])
        
        prep_code = ""
        after_code = ""

        # print(f'{i}: {op.name} {op.main_target} {op.deps_fake}')
        for kdn in op.deps_fake:
            kdn_name = kdn.split(" ")[0]
            if (
                not self.is_alive(op, kdn)
                # or op_sched.input_size[0] in kdn.name
            ):
                fake_code = self._generate_fake_data(kdn_name, is_self=(kdn_name == op.main_target))
                prep_code += fake_code[0]
                after_code += fake_code[1]
        
        if not_first:
            prev_i = i - self.op_sched.op_list[:i][::-1].index(op) - 1
            input_names = []
            for kdn in op.users_global:
                if f"del {kdn}" in self.op_sched.op_name_list[prev_i:i]:
                    input_names += [kdn.main_target]  # kdn.tensor_targets

            inputs = ",".join(input_names)
            code = f"_{mt}.backward({mt}.grad, inputs=[{inputs}], retain_graph={not last})"
        else:
            code = f"_{mt}.backward({mt}.grad, retain_graph={not last})"
        bwd_code = f"{prep_code}\n" f"{code}\n" f"{after_code}"
        
        if op.is_rand:
            bwd_code = f"rng_state.get('{op.name}');rng_state.restore('{op.name}')\n{bwd_code}"
            
        return [bwd_code]

    def del_op2(self, op, i):
        code = ""
        if op.kdn_type == "data":
            if (
                op.info is not None
                and op.info.requires_grad
                # and _is_alive(op.name.replace("data", "phantoms"), i)
                and op.proxy
            ):
                code += f"_{op.main_target}.data = torch.empty(0,device=device);"
                for inp in op.inplace_targets:
                    # code += f"_{inp}.data = torch.empty(0,device=device);"
                    code += f"del _{inp};"

                if op.includes_phantoms:
                    code += f"del _{op.main_target};"

                if op.includes_base:
                    if op.proxy:
                        code += f"_{op.main_target}._base.data = torch.empty(0,device=device);"
                    else:
                        code += f"{op.main_target}._base.data = torch.empty(0,device=device);"

            for v in op.tensor_targets:
                code += f"{v}.data = torch.empty(0,device=device); "

            for v in op.container_targets:
                code += f"del {v};"

        if op.kdn_type == "grad":
            code += f"{op.main_target}.grad = None;"
            
        if op.kdn_type == "phantoms":
            code += f"del _{op.main_target};"

        return [code]

    def compile(self, op_sched):
        self.op_sched = op_sched
        # TODO: op_sched renamed

        fct_list = []
        fwd_code = []
        bwd_code = []

        for i, op in enumerate(op_sched.op_list):
            if "fwd" in op.name:
                if op.is_swap:
                    fct_list.append(self.compile_swapin(op))
                else:
                    fct_list.append(self.compile_fwd(op, i))
                    fwd_code.append(self.compile_fwd2(op, i))
            elif "bwd" in op.name:
                fct_list.append(self.compile_bwd(op, i))
                bwd_code.append(self.compile_bwd2(op, i))
            elif "data" in op.name:
                if op.is_swap:
                    fct_list.append(self.compile_swapout(op))
                else:
                    fct_list.append(self.compile_del_data(op))
                    bwd_code.append(self.del_op2(op, i))
            elif "grad" in op.name:
                fct_list.append(self.compile_del_grad(op))
                bwd_code.append(self.del_op2(op, i))
            else:
                fct_list.append([])

        # print(f'fwd_code: {fwd_code}')
        # print(f'bwd_code: {bwd_code}')

        return fct_list, fwd_code, bwd_code

