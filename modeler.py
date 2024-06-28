from asuta import Asuta
import torch
import torchvision.models as models

class Modeler():
    def __init__(
        self,
        model,
    ):
        super().__init__()
        self.model = model
        

    def build(self, inputs, mem_constraint=16, mode="default"):
        self.inputs = inputs
        self.mem_constraint = mem_constraint
        self.batch_size = 0

        if isinstance(self.inputs, list):
            self.batch_size = self.inputs[0].shape[0]
            print(self.inputs[0].shape)
        else:
            self.batch_size = self.inputs.shape[0]
            print(self.inputs.shape)

        greedy_start = time.time()

        self.debug_model = Asuta(self.model, self.inputs)
        candidates, init_peak_memory = self.gen_candidates()

        if mode == "default":
            evict_list, evict_tensor_mode = self.gen_eviction_plan(candidates, init_peak_memory)
        elif mode == "maximum":
            evict_list, evict_tensor_mode = self.gen_maximum_eviction_plan(candidates, init_peak_memory)
        elif mode == "max_swap":
            evict_list, evict_tensor_mode = self.gen_max_swap_eviction_plan(candidates, init_peak_memory)

        # print(f'evict_list: {evict_list}')
        # print(f'evict_tensor_mode: {evict_tensor_mode}')


        # r_mode = 0
        # s_mode = 0
        # for mode in evict_tensor_mode.values():
        #     if mode == "recompute":
        #         r_mode += 1
        #     else:
        #         s_mode += 1
        
        # print(f'evict_list: {evict_list}')
        # print(f'evict_tensor_mode: {evict_tensor_mode}')
        # print(f'total: {r_mode + s_mode}, recompute: {r_mode}, swap: {s_mode}')
            

        self.new_model = Asuta(self.model, self.inputs, evict_list, evict_tensor_mode) 
        self.new_model.build()

        greedy_end = time.time()
        print(f'greedy_time (sec): {greedy_end - greedy_start}')

        return self.new_model

    def gen_eviction_plan(self, candidates, init_peak_memory):
        evict_list = []
        evict_tensor_mode = {}
        swap_data = []
        base_candidates = {}
        residual_candidates = []

        def rerun():
            self.debug_model.eviction_list = evict_list
            self.debug_model.evict_tensor_mode = evict_tensor_mode
            self.debug_model.gen_op_list_evict()
            self.debug_model.compile_function(evict=True)
            self.debug_model.peak_memory_usage = [0, 0, 'fw']
            torch.cuda.reset_peak_memory_stats()
            outputs = self.debug_model.forward(*self.inputs)
            loss = outputs.mean()
            loss.backward()
            self.debug_model.backward() 
            peak_memory = self.debug_model.peak_memory_usage
            # print(peak_memory)

            return peak_memory

        # print(evict_list)
        # print(evict_tensor_mode)

        def greedy_replace(peak_memory):
            if peak_memory[-1] == "fw":
                op = self.debug_model.fwd_op_list_evict[peak_memory[1]]
            else:
                op = self.debug_model.bwd_op_list_evict[peak_memory[1]]

            for user_name in op.users_global:
                if "fv" not in user_name and "input" not in user_name and "out" not in user_name: continue
                assert "grad" in user_name
                
                user_name = user_name.replace("grad", "data")
                swap_data.append(user_name)

            # for data in candidates.keys():
            for data in evict_list:
                if "out" in data or data in swap_data:# or "fv" in data:
                    evict_tensor_mode[data] = "swap"
                else:
                    evict_tensor_mode[data] = "recompute"
            
            # print(f"evict_tensor_mode: {evict_tensor_mode}")
            # print(f"swap_data: {swap_data}")

        def compare_lists(list1, list2):
            if list1[1] == list2[1] and list1[2] == list2[2]: return True 
            return False 
        
        
        for data, mem in candidates.items():
            if init_peak_memory > self.mem_constraint:
                base_candidates[data] = mem
                init_peak_memory -= mem/1000**3
            else:
                residual_candidates.append((data, mem))
                
        # print(f'base_candidates: {base_candidates}')
        # print(f'residual_candidates: {residual_candidates}')

        # test the base_candidates (all swap plan)
        for data in base_candidates.keys():
            evict_list.append(data)
            evict_tensor_mode[data] = "swap"
            # if "out" in data or "__211_input" in data or "__80_input" in data or "__154_input" in data:
            #     evict_tensor_mode[data] = "swap"
            # else:
            #     evict_tensor_mode[data] = "recompute"
        
        test_peak_mem = rerun()[0]
        while test_peak_mem > self.mem_constraint:
            d, m = residual_candidates.pop(0)
            # print(f"evicting {d} with {m} memory")
            base_candidates[d] = m
            evict_list.append(d)
            evict_tensor_mode[d] = "swap"
            test_peak_mem = rerun()[0]

        for data in base_candidates.keys():
            evict_list.append(data)
            evict_tensor_mode[data] = "swap"
        

        min_peak_mem = rerun() 

        

        greedy_replace(min_peak_mem)
        print(f'min_peak_mem: {min_peak_mem}')

        tt = 0
        while True:
            peak_mem = rerun()
            # print(f'peak_mem: {peak_mem}')
            if compare_lists(min_peak_mem, peak_mem):
                break
            greedy_replace(peak_mem)
            tt += 1
        print(f"rerun {tt}")

        print(f'evict_tensor_mode: {evict_tensor_mode}')

        return evict_list, evict_tensor_mode
    
    def gen_maximum_eviction_plan(self, candidates, init_peak_memory):
        evict_list = []
        evict_tensor_mode = {}
        swap_data = []

        def rerun():
            self.debug_model.eviction_list = evict_list
            self.debug_model.evict_tensor_mode = evict_tensor_mode
            self.debug_model.gen_op_list_evict()
            self.debug_model.compile_function(evict=True)
            self.debug_model.peak_memory_usage = [0, 0, 'fw']
            torch.cuda.reset_peak_memory_stats()
            outputs = self.debug_model.forward(*self.inputs)
            loss = outputs.mean()
            loss.backward()
            self.debug_model.backward() 
            peak_memory = self.debug_model.peak_memory_usage
            # print(peak_memory)

            return peak_memory

        # print(evict_list)
        # print(evict_tensor_mode)

        def greedy_replace(peak_memory, for_expr=False):
            if peak_memory[-1] == "fw":
                op = self.debug_model.fwd_op_list_evict[peak_memory[1]]
            else:
                op = self.debug_model.bwd_op_list_evict[peak_memory[1]]


            if not for_expr:
                for user_name in op.users_global:
                    if "fv" not in user_name and "input" not in user_name and "out" not in user_name: continue
                    assert "grad" in user_name
                    
                    user_name = user_name.replace("grad", "data")
                    swap_data.append(user_name)

            for data in candidates.keys():
                if "out" in data or data in swap_data:# or "fv" in data:
                    evict_tensor_mode[data] = "swap"
                else:
                    evict_tensor_mode[data] = "recompute"

            # print(f"evict_tensor_mode: {evict_tensor_mode}")
            # print(f"swap_data: {swap_data}")

        def compare_lists(list1, list2):
            if list1[1] == list2[1] and list1[2] == list2[2]: return True 
            return False 

        
        for data, mem in candidates.items():
            evict_list.append(data)
            evict_tensor_mode[data] = "swap"

        min_peak_mem = rerun() 
        print(f'min_peak_mem: {min_peak_mem}')

        
        # greedy_replace(min_peak_mem, for_expr=True)
        # peak_expr = rerun()
        # print(f'peak_expr: {peak_expr}')
        
        greedy_replace(min_peak_mem)
        # evict_list.pop()
        # evict_tensor_mode.popitem()

        tt = 0
        while True:
            peak_mem = rerun()
            print(f'peak_mem: {peak_mem}')
            if compare_lists(min_peak_mem, peak_mem):
                break
            greedy_replace(peak_mem)
            tt += 1
        
        print(f"rerun {tt}")

        print(f'evict_tensor_mode: {evict_tensor_mode}')

        return evict_list, evict_tensor_mode

    def gen_max_swap_eviction_plan(self, candidates, init_peak_memory):
        evict_list = []
        evict_tensor_mode = {}

        for data, mem in candidates.items():
            evict_list.append(data)
            evict_tensor_mode[data] = "swap"

        # print(f'evict_list: {len(evict_list)}')
        print(f'evict_list: {evict_list}')

        # evict_list.pop()
        # evict_tensor_mode.popitem()

        self.debug_model.eviction_list = evict_list
        self.debug_model.evict_tensor_mode = evict_tensor_mode
        self.debug_model.gen_op_list_evict()
        self.debug_model.compile_function(evict=True)
        self.debug_model.peak_memory_usage = [0, 0, 'fw']
        torch.cuda.reset_peak_memory_stats()
        outputs = self.debug_model.forward(*self.inputs)
        loss = outputs.mean()
        loss.backward()
        self.debug_model.backward() 
        peak_memory = self.debug_model.peak_memory_usage
        print(peak_memory)

        return evict_list, evict_tensor_mode

    def gen_candidates(self):
        self.debug_model.gen_op_list()
        self.debug_model.compile_function(evict=False)
        
        torch.cuda.reset_peak_memory_stats()
        outputs = self.debug_model.forward(*self.inputs)
        loss = outputs.mean()
        loss.backward()
        self.debug_model.backward() 
        
        peak_memory = self.debug_model.peak_memory_usage
        print(peak_memory)
        candidates = {} 

        expr_total_memory = 0
        

        op_name = ""
        if peak_memory[-1] == "fw":
            op_name = self.debug_model.fwd_op_list[peak_memory[1]].name     
            op_name = op_name.replace("fwd_", "") + " data"
        else:
            op_name = self.debug_model.bwd_op_list[peak_memory[1]].name 
            op_name = op_name.replace("bwd_", "") + " data"
        
        for data, mem in self.debug_model.data_memory.items():
            if op_name in data: break
            if "data" not in data: continue
            expr_total_memory += mem
            if "fv" not in data and "input" not in data and "out" not in data: continue
            # if "fv" not in data and "input" not in data: continue
            candidates[data] = mem
        
        if peak_memory[1] == 0:
            candidates.popitem()

        # print(candidates)
        # print(f'fwd_op: {[op.name for op in self.debug_model.fwd_op_list]}')
        # print(f'bwd_op: {[op.name for op in self.debug_model.bwd_op_list]}')
        # print(op_name)
        print(f'expr_total_memory: {expr_total_memory/1000**3} GB')

        return candidates, peak_memory[0]
            
import time
import torch.optim as optim

if __name__ == "__main__":
    device = torch.device("cuda")
    batch_size = 275 

    # net = models.resnet101().to(device)
    # sample = [torch.rand(batch_size, 3, 224, 224).to(device)]

    from gpt import get_GPT
    net = get_GPT(model="GPT2-small").to(device)
    s = [torch.randint(0, 600, [17, 512]).to(device)]
    sample = [torch.randint(0, 600, [17, 512]).to(device)]

    # net = models.vgg16().to(device)
    # net = models.resnet50().to(device)
    # net = models.resnet152().to(device)
    # s = [torch.rand(275, 3, 128, 128).to(device)]
    # sample = [torch.rand(batch_size, 3, 128, 128).to(device)]

    md = Modeler(net)
    new_model = md.build(s, 15, "max_swap")

    del md
    torch.cuda.empty_cache()

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    torch.cuda.reset_peak_memory_stats()
    max_before = torch.cuda.max_memory_allocated()/1000/1000/1000
    print(f"Before: {max_before}, {torch.cuda.memory_reserved()/1000/1000/1000}")

    for _ in range(2):
        # optimizer.zero_grad()

        start_time = time.time()
        # outputs = net(sample[0])
        outputs = new_model(*sample)

        # loss = criterion(outputs, y.to(device))
        loss = outputs.mean()
        loss.backward()
        new_model.backward() 
        # optimizer.step()

        # running_loss += loss.item()
        # print(f'loss: {running_loss}')
        # running_loss = 0.0
        
        end_time = time.time()
        train_time = end_time - start_time
        print(f'training_time (sec): {train_time}')

        peak_mem = torch.cuda.max_memory_allocated() - max_before
    print(f'peak_mem (GB): {peak_mem/1000/1000/1000}')
        
    