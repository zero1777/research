import torch
import time

tgpu = torch.rand(32768000, device="cuda")
tcpu = torch.empty(32768000, device="cpu", pin_memory=True)
# print(tgpu[0][0].element_size())

start = time.time()
tcpu.copy_(tgpu, non_blocking=False)
tcpu.detach()
end = time.time()

print(f"Time taken: {end-start}")