import torch

swap_stream = torch.cuda.Stream()
cpu_tensor = torch.randn(10000, 10000)
gpu_tensor = torch.randn(10000, 10000).cuda()
with torch.cuda.stream(swap_stream):
    a_tensor = cpu_tensor.cuda(non_blocking=True)
    gpu_tensor = cpu_tensor.cuda(non_blocking=True)
    event = swap_stream.record_event() 

# Perform some computation
# c_result = cpu_tensor + 1
g_result = gpu_tensor + 1

# Wait for the swap stream to complete
# torch.cuda.default_stream().wait_event(event)
print(f'CPU Tensor: {cpu_tensor[0, 0]}')
print(g_result[0, 0])
