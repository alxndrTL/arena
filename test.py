import os
import json

import torch
import torch.nn.functional as F

from models.lm import load_model
from models.transformer.transformer import TransformerConfig
from models.mamba.mamba import MambaConfig
from models.mamba.mamba2 import Mamba2Config

from data.dataloader import DataLoader

# ------

vocab_size = 50257 # gpt2 enc
ctx_len = 1024

batch_size = 16

load_dir = "runs/royal-frog-94/"

device = "cuda"

# ------

model = load_model(load_dir, vocab_size, device)
model.eval()

val_loader = DataLoader("data/fineweb10B/fineweb_val_*.bin",batch_size, ctx_len, 1, 1)

eval_loss = 0.0
total_batches = 0

with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        
        _, loss = model(x, y)
        eval_loss += loss.item()
        total_batches += 1

        if total_batches >= 10:
            break

eval_loss /= total_batches
print(f"Evaluation loss: {eval_loss:.4f}")
