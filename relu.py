import torch
import gc

def pack(x):
    return x

def unpack(x):
    return x

x = torch.randn(5, requires_grad=True)
with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    y = torch.relu(x)

# print(y.equal(y.grad_fn._saved_result))  # True
# print(y is y.grad_fn._saved_result)  # False

a = torch.randn(5, requires_grad=True).to('cuda')
# b = a.exp()
with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
    b = torch.nn.functional.relu(a)
c = torch.empty(b.size(), device='cpu')
c.copy_(b, non_blocking=True)
c = c.detach()
print(torch.cuda.memory_allocated())

print(f'a: {a}')
print(f'b: {b}')
print(f'c: {c}')

b.data = torch.empty(0, device='cuda')

print(torch.cuda.memory_allocated())

# b.backward(torch.ones_like(a))