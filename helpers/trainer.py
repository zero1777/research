from collections import defaultdict

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(step, max_steps, lr, metrics, mode="train"):
    metrics_print = " - ".join([f"{m}: {v[-1]:.3f}" for m, v in metrics.items()])

    if mode == "train":
        print(f"Step {step + 1}/{max_steps} - LR:{lr:.4f} -", metrics_print, end="\r")
    if mode == "eval":
        print(f"\n\Step {step + 1}/{max_steps} -", metrics_print)


def train(
    model: Module,
    ds_train,
    batch_size: int,
    lr: float,
    max_steps: int,
    weight_decay: float = 1e-2,
    device: str = DEVICE,
    log_every: int = 10,
    with_optimizer: bool = False,
) -> defaultdict:
    metrics_tracker = defaultdict(list)
    # val_loss_tracker = defaultdict(list)

    model.to(device)
    optimizer = Adam(model.parameters(), lr=10 * lr, weight_decay=weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr, verbose=False)
    model.train()
    
    torch.cuda.reset_peak_memory_stats()
    max_before = torch.cuda.max_memory_allocated()/1000/1000/1000
    print(f"Before: {max_before}, {torch.cuda.memory_reserved()/1000/1000/1000}")
    print(f'max_step: {max_steps}')

    start = time.time()

    for step in range(max_steps):
        if with_optimizer:
            optimizer.zero_grad(set_to_none=True)

        inputs, labels = ds_train.get_batch(batch_size)
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)

        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
        loss = logits.mean()
        loss.backward()

        if with_optimizer:
            optimizer.step()
        # scheduler.step()

        # print(f'Step: {step}')
        # metlossrics_tracker["train_loss"].append(loss.detach().cpu().item())
        # if step % log_every == 0 or step == max_steps - 1:
        #     log(step, max_steps, scheduler.get_last_lr()[-1], metrics_tracker)

        # val_loss = evaluate(model, dl_val, device)
        # val__tracker["val_loss"].append(val_loss)

    end = time.time()
    print(f'Training time (sec): {end-start}')

    peak_mem = torch.cuda.max_memory_allocated() - max_before
    print(f'peak_mem (GB): {peak_mem/1000/1000/1000}')

    return metrics_tracker


@torch.inference_mode()
def evaluate(model: Module, dl_val: DataLoader, device: DEVICE) -> float:
    model.eval()
    running_loss = 0.0
    num_steps = 0

    for sequence, labels in dl_val:
        sequence, labels = sequence.to(device), labels.to(device)
        logits = model(sequence)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)

        running_loss += loss.cpu().item()
        num_steps += 1

    return running_loss / num_steps
