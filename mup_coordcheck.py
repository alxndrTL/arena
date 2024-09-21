"""
Runs a coord check on the model defined in config.py.
Data is dummy.
"""

# todo : mamba et mamba2

import os
from contextlib import nullcontext

import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch._inductor.config as torch_ind_config

from models.lm import LM

from utils.mup.coord_check import get_coord_data, plot_coord_data
from config import *

# --------------------------

output_dir = ""

lr = 2e-3

batch_size = 16

max_value = 100

widths = [64, 128, 256, 512, 768]
# check that for all these widths, d_model is divisible by d_head

# --------------------------

d_head_0 = config.d_head # fixed through scaling
config.mup_base_width = widths[0]

seed = 123456789 + seed

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"
torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
dtype_ctx = (nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type, torch_dtype))

if use_torch_compile:
    if hasattr(torch_ind_config, "coordinate_descent_tuning"):
        torch_ind_config.coordinate_descent_tuning = True

class RandomDataset(Dataset):
    def __len__(self):
        return 9999999

    def __getitem__(self, idx):
        data = torch.randint(low=0, high=max_value, size=(batch_size, ctx_len))
        x = data[:, :-1].int()
        y = data[:, 1:].long()
        return x, y

def lazy_model(width):
    config.d_model = width
    if architecture == "Transformer":
        config.d_ff = int((8/3) * width)
    
    if architecture == "Transformer" or architecture == "Mamba2":
        config.n_heads = width//d_head_0
        config.n_kv_heads = config.n_heads

    config.__post_init__()

    return lambda: LM(config, vocab_size=max_value).to(device) # todo

models = {width: lazy_model(width) for width in widths}

dataset = RandomDataset()
loader = DataLoader(dataset, batch_size=None, shuffle=True)
iter_ = iter(loader)

# todo : warning "optimizer" not taken into account
optcls = lambda model: model.configure_optimizers(weight_decay, lr, (adam_b1, adam_b2), device_type)

df = get_coord_data(models, iter_, optcls, dtype_ctx, nsteps=10)

if use_mup:
    name = "mup.png"
else:
    name = "no_mup.png"

plot_coord_data(df, legend="auto", save_to=os.path.join(output_dir, name))
