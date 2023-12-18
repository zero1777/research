import torch
import ast

tensor_gpu = torch.rand(128).cuda()
# tensor_cpu = torch.empty(tensor_gpu.shape, pin_memory=True, device=torch.device('cpu'))
print(torch.cuda.memory_allocated())
stream = torch.cuda.Stream()
str = f"tensor_cpu = torch.empty(tensor_gpu.shape, pin_memory=True, device='cpu')\nwith torch.cuda.stream(stream):\n\ttensor_cpu.copy_(tensor_gpu, non_blocking=False); tensor_cpu = tensor_cpu.detach()\n\ttensor_gpu.data = torch.empty(0, device=torch.device('cuda'))"
str = [str]
str2 = compile(ast.parse("\n".join(str)), "", "exec")
exec(str2)
# print(tensor_gpu, tensor_cpu)
print(torch.cuda.memory_allocated())
print(torch.cuda.max_memory_allocated())
