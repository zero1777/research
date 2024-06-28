import sys

sys.path.append("..")

from pathlib import Path
import matplotlib.pyplot as plt
import torch

from model.llm import LLM
from model.tokenizer import Tokenizer, train_tokenizer

from helpers.dataset import NextTokenPredictionDataset
from helpers.trainer import train
from helpers.config import LLMConfig, TrainingConfig

from asuta import Asuta
from modeler import Modeler

llm_config = LLMConfig(
    vocab_size=2_000,
    context_size=256,
    dim_emb=512,
    num_layers=8,
    num_heads=8,
    emb_dropout=0.0,
    ffd_bias=True,
    ffd_dropout=0.0,
)

train_config = TrainingConfig(
    retrain_tokenizer=True, batch_size=64, learning_rate=1e-4, weight_decay=1e-5, max_steps=10, log_frequency=1
)

input_file = "tinyshakespeare.txt"
output_file = Path(input_file).with_suffix(".model")

if not output_file.exists() or train_config.retrain_tokenizer:
    train_tokenizer(input_file, llm_config.vocab_size)

tokenizer = Tokenizer(str(output_file))

sentence = "The role of the tokenizer is to build a mapping between a sentences represented as a string and token indices"
print(tokenizer.sp.EncodeAsPieces(sentence))

assert tokenizer.decode(tokenizer.encode(sentence)) == sentence

# This helper class allow to generate batches of inputs and targets where targets last element is the next token to predict
ds_train = NextTokenPredictionDataset(input_file, llm_config.context_size, tokenizer)

X, y = ds_train.get_batch(batch_size=1)

print(X.shape, y.shape)

model = LLM(
    vocab_size=tokenizer.vocab_size,
    context_size=llm_config.context_size,
    dim_emb=llm_config.dim_emb,
    num_layers=llm_config.num_layers,
    attn_num_heads=llm_config.num_heads,
    emb_dropout=llm_config.emb_dropout,
    ffd_bias=llm_config.ffd_bias,
    ffd_dropout=llm_config.ffd_dropout
)


param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2

print(f"total params: {sum(p.numel() for p in model.parameters()):,d}")
print(f"learnable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}")
print(f"model size: {size_all_mb:.3f}MB")

#print(model)

# loss_history = train(
#     model,
#     ds_train,
#     batch_size=train_config.batch_size,
#     lr=train_config.learning_rate,
#     max_steps=train_config.max_steps,
#     weight_decay=train_config.weight_decay,
#     log_every=train_config.log_frequency,
# )

# torch.cuda.reset_peak_memory_stats()
# max_before = torch.cuda.max_memory_allocated()/1000/1000/1000
# print(f"Before: {max_before}, {torch.cuda.memory_reserved()/1000/1000/1000}")

inputs, y = ds_train.get_batch(230)
x, y = ds_train.get_batch(250)
x = [x.to("cuda")]
inputs = [inputs.to("cuda")]
model = model.to("cuda")

# inputs = [torch.randint(0, 600, [100, 64]).to("cuda")]

# new_model = Asuta(model, inputs)


md = Modeler(model)
# new_model = md.build(inputs, 11.5, "max_swap")
new_model = md.build(inputs, 13, "max_swap")
# new_model = md.build(inputs, 13.6, "maximum")
# new_model = md.build(inputs, 13.6, "max_swap")

del md
torch.cuda.empty_cache()

torch.cuda.reset_peak_memory_stats()
max_before = torch.cuda.max_memory_allocated()/1000/1000/1000
print(f"Before: {max_before}, {torch.cuda.memory_reserved()/1000/1000/1000}")

for _ in range(3):
    outputs = new_model(*x)
    loss = outputs.mean()
    loss.backward()
    new_model.backward()

    peak_mem = torch.cuda.max_memory_allocated() - max_before

print(f'peak_mem (GB): {peak_mem/1000/1000/1000}')
# 9.1833

# import time
# bs = 270
# steps = 12800//bs
# print(f'steps: {steps}')
# start = time.time()

# for _ in range(steps):
#     a, b = ds_train.get_batch(bs)
#     a = a.to("cuda")
#     outputs = new_model(a)
#     loss = outputs.mean()
#     loss.backward()
#     new_model.backward()

# end = time.time()
# print(f"Training time (sec): {end-start}")
