import torch

def active_bytes():
    stats = torch.cuda.memory_stats()
    current_active_byte =  stats['active_bytes.all.current']
    return current_active_byte
a = torch.tensor([1., 2., 3.], requires_grad=True, device='cuda:0')
t = a.size()
print('GPU memory: ', active_bytes())
# b = torch.empty(t, device='cpu')
# b = a.cpu()
# b.copy_(a, non_blocking=True)
# b = b.detach().requires_grad_(True)

# print('a: ', a)
# print('b: ', b)

# # del a
c = a ** 2
d = c ** 3
e = d
print('d: ', d)
print('e: ', e)

with torch.cuda.stream(torch.cuda.Stream()):
    b = torch.empty(d.size(), device='cpu:0')
    b.copy_(d, non_blocking=True)
    b = b.detach().requires_grad_(True)
    print('d: ', d)
d.data = torch.empty(0, device='cuda:0')

print('d: ', d)
print('e: ', e)

d.data = b.data.cuda(non_blocking=True)
print('d: ', d)
print('e: ', e)


# e.sum().backward()
# a.data = torch.empty(0, device='cuda:0')
# print('GPU memory: ', active_bytes())

# # print(type(a), a.data_ptr())
# # print(type(b), b.data_ptr())



# # d = b ** 3
# # a = b.cuda(non_blocking=True)
# # a = a.detach().requires_grad_(True)
# # a.data = torch.empty(t, device='cuda:0')
# c.data = b.data.cuda(non_blocking=True)
# # print('a: ', a)
# c.sum().backward()
# # print('a.grad: ', a.grad)
# # d.sum().backward()
# # print('b.grad: ', b.grad)

