import torch
import time
import os
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import logging

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

log_fname = sys.argv[1] if len(sys.argv) > 1 else None
logging.basicConfig(filename=log_fname, filemode='w', level=logging.INFO)

def fit_regression(x,y):
    ''' Fits linear regression and returns slope'''
    x = np.array(x).reshape(-1,1)
    y = np.array(y).reshape(-1,1)
    model = LinearRegression().fit(x,y)
    return model.coef_[0][0]

def benchmark_cpu_to_gpu(method=1,pin_memory=False):
# Transfer Tensors from ~373MB to ~7.45GB
    tgpu = torch.rand(200,200, device=torch.device(torch.device("cuda")))
    x = []
    y = []
    for i in range(1,21):
        del tgpu
        torch.cuda.empty_cache()
        # source Tensor on CPU
        tcpu = torch.empty(10000,10000,i, device=torch.device(torch.device("cpu")),pin_memory=pin_memory)
        y.append(tcpu.nelement() * tcpu.element_size()/1024/1024/1024) # Tensor size in GB
        # destination tensor on GPU for Method 2
        if method == 2:
            tgpu = torch.rand(10000,10000,i, device=torch.device(torch.device("cuda")))
        torch.cuda.synchronize()

        start_time = time.time()
        if method == 1:
            tgpu = tcpu.to(torch.device("cuda"),non_blocking=False)
        elif method == 2:
            tgpu.copy_(tcpu,non_blocking=False)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        x.append(elapsed_time)
    logging.info(f"CPU->GPU Method {method}, pin_memory={pin_memory}, Transfer rate: {fit_regression(x,y):.4f} GB/s.")

def benchmark_gpu_to_cpu(method=1,pin_memory=False):
# Transfer Tensors from ~373MB to ~7.45GB
    tgpu = torch.rand(200,200, device=torch.device(torch.device("cuda")))
    x = []
    y = []
    for i in range(1,21):
        del tgpu
        torch.cuda.empty_cache()
        # source Tensor on GPU
        tgpu = torch.rand(10000,10000,i, device=torch.device(torch.device("cuda")))
        y.append(tgpu.nelement() * tgpu.element_size()/1024/1024/1024) # Tensor size in GB
        # destination tensor on CPU for Method 2
        if method == 2:
            tcpu = torch.empty(10000,10000,i, device=torch.device(torch.device("cpu")),pin_memory=pin_memory)
        torch.cuda.synchronize()

        start_time = time.time()
        if method == 1:
            tcpu = tgpu.to(torch.device("cpu"),non_blocking=False)
        elif method == 2:
            tcpu.copy_(tgpu,non_blocking=False)
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        x.append(elapsed_time)
    logging.info(f"GPU->CPU Method {method}, pin_memory={pin_memory}, Transfer rate: {fit_regression(x,y):.4f} GB/s.")

benchmark_cpu_to_gpu(1,False)
benchmark_cpu_to_gpu(2,False)
benchmark_cpu_to_gpu(1,True)
benchmark_cpu_to_gpu(2,True)

benchmark_gpu_to_cpu(1,False)
benchmark_gpu_to_cpu(2,False)
benchmark_gpu_to_cpu(1,True)
benchmark_gpu_to_cpu(2,True)