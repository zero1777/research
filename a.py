import torch
import ast
import torch.nn as nn

# def fct_get_pack():
#     def pack(x):
#         return x
#     return pack

# def fct_get_unpack():
#     def unpack(x):
#         return x
#     return unpack

# a0 = torch.tensor([1.,2.,3.], requires_grad=True)
# a00 = torch.tensor([1.,2.,3.], requires_grad=True)
# with torch.autograd.graph.saved_tensors_hooks(
#     fct_get_pack(), fct_get_unpack()
# ):
#     a = torch.Tensor.add(a0, a00)
# with torch.autograd.graph.saved_tensors_hooks(
#     fct_get_pack(), fct_get_unpack()
# ):
#     b = torch.relu_(a)
    
sample =  torch.randint(0,600, [8, 500]).to('cuda')
pos_ids = torch.arange(
                0, sample.size(-1), dtype=torch.long, device='cuda'
            ).unsqueeze(0)
wte = nn.Embedding(50257, 768).to('cuda')
wte(pos_ids)
# print(pos_ids.dtype)