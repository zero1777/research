import torch
import ast

def fct_get_pack():
    def pack(x):
        return x
    return pack

def fct_get_unpack():
    def unpack(x):
        return x
    return unpack

a0 = torch.tensor([1.,2.,3.], requires_grad=True)
a00 = torch.tensor([1.,2.,3.], requires_grad=True)
with torch.autograd.graph.saved_tensors_hooks(
    fct_get_pack(), fct_get_unpack()
):
    a = torch.Tensor.add(a0, a00)
with torch.autograd.graph.saved_tensors_hooks(
    fct_get_pack(), fct_get_unpack()
):
    b = torch.relu_(a)