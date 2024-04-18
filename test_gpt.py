from asuta import Asuta
import rkgb.src as rkgb
from gpt import get_GPT
import torch
import time
import argparse
import sys

device = torch.device("cuda")

model_name = "GPT2-small"
print(f"--- Using Model: {model_name}\n")


batch_size = 10
model = get_GPT(model=model_name).to(device)
sample = [ torch.randint(0,600, [batch_size, 500]).to(device) ]
# print(len(sample[0]))

new_model = Asuta(model, sample)

torch.cuda.reset_peak_memory_stats()
max_before = torch.cuda.max_memory_allocated()/1000/1000/1000
print(f"Before: {max_before}, {torch.cuda.memory_reserved()/1000/1000/1000}")
 
repeat = 2
for _ in range(repeat):
    # start_time = time.time()
    
    # outputs = model(sample[0])
    outputs = new_model(*sample)
    
    loss = outputs.mean()
    loss.backward()
    new_model.backward() 
    
    # end_time = time.time()
    # train_time = end_time - start_time
    # print(f'train_time (sec): {train_time}') 

    peak_mem = torch.cuda.max_memory_allocated() - max_before
print(f"Peak memory (GB): {peak_mem/1000/1000/1000}")
