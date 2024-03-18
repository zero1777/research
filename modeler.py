from asuta import Asuta
import torch
import torchvision.models as models

class Modeler():
    def __init__(
        self,
        model,
        inputs,
    ):
        super().__init__()
        self.model = model
        self.inputs = inputs
        self.mem_constraint = 16
        self.batch_size = 0

        if isinstance(self.inputs, list):
            self.batch_size = self.inputs[0].shape[0]
            print(self.inputs[0].shape)
        else:
            self.batch_size = self.inputs.shape[0]
            print(self.inputs.shape)

    def build(self, mem_constraint=16):
        self.debug_model = Asuta(self.model, self.inputs)
        candidates = self.gen_candidates()
        evict_list, evict_tensor_mode = self.gen_eviction_plan(candidates)

        # self.new_model = Asuta(self.model, self.inputs, evict_list, evict_tensor_mode) 
        # self.new_model.build()

        # return self.new_model

    def gen_eviction_plan(self, candidates):
        evict_list = []
        evict_tensor_mode = {}
        for data in candidates.keys():
            evict_list.append(data)
            # evict_tensor_mode[data] = "swap"
            if "out" in data or "__61_input" in data or "__54_input" in data:   
                evict_tensor_mode[data] = "swap"
            else:
                evict_tensor_mode[data] = "recompute"

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
            print(peak_memory)


        # print(evict_list)
        # print(evict_tensor_mode)

        rerun() 
        

        def greedy_replace():
            # op_name = 
            pass


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
            if "fv" not in data and "input" not in data and "out" not in data: continue
            candidates[data] = mem
        
        return candidates
            
        # print(candidates)
        # print(f'fwd_op: {[op.name for op in self.debug_model.fwd_op_list]}')
        # print(f'bwd_op: {[op.name for op in self.debug_model.bwd_op_list]}')
        # print(op_name)
            


if __name__ == "__main__":
    device = torch.device("cuda")
    batch_size = 275

    net = models.resnet101().to(device)
    sample = [torch.rand(batch_size, 3, 224, 224).to(device)]

    net = models.vgg16().to(device)
    sample = [torch.rand(batch_size, 3, 128, 128).to(device)]

    md = Modeler(net, sample)
    md.build()
    