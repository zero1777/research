import torch
import time

tensor_gpu = torch.randn(13107200, requires_grad=True, device='cuda')

start = time.time()

tensor_cpu = torch.empty(tensor_gpu.size(), device="cpu")
tensor_cpu.copy_(tensor_gpu)
# tensor_cpu = tensor_gpu.cpu()
tensor_cpu = tensor_cpu.detach()
# tensor_gpu = tensor_cpu.cuda()
torch.cuda.synchronize()

end = time.time()
print(torch.cuda.memory_allocated())
print('gpu to cpu', end - start)

torch.cuda.synchronize()

tensor_cpu = torch.randn(13107200, requires_grad=True, device='cpu')
start = time.time()
# tensor_cpu_ = tensor_cpu
tensor_gpu = tensor_cpu.cuda()
tensor_gpu_ = tensor_gpu
torch.cuda.synchronize()

end = time.time()
print('cpu to gpu', end - start)