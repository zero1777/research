import torch

a = torch.tensor([1.]*514, requires_grad=True, device='cuda:0')

print(torch.cuda.memory_allocated())