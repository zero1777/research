import torch
import time


tensor_gpu = torch.rand(10*26214400, requires_grad=True, device='cuda')
# tensor_cpu = torch.rand(10*26214400, device='cpu', pin_memory=True)

torch.cuda.synchronize()
start = time.time()
tensor_cpu = torch.empty(10*26214400, device="cpu", pin_memory=True)
tensor_cpu.copy_(tensor_gpu)
tensor_cpu = tensor_cpu.detach()
# tensor_gpu = torch.empty(10*26214400, requires_grad=True, device="cuda")
# tensor_gpu.data.copy_(tensor_cpu)
torch.cuda.synchronize()

end = time.time()
# print('gpu to cpu', end - start)
print(torch.cuda.memory_allocated())
print('cpu to gpu', end - start)
