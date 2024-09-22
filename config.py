"""
default config is 124M Transformer
"""

from models.transformer.transformer import TransformerConfig
from models.mamba.mamba import MambaConfig
from models.mamba.mamba2 import Mamba2Config

# ---------------------------------------------

seed = 0 # 0, 1, 2...

# --- downstream eval parameters ---
#eval_interval = 1000
#num_tasks = 100

# --- data ---
vocab_size = 50257 # gpt2 enc
ctx_len = 1024

# --- model parameters ---
architecture = "Transformer" # "Transformer" or "Mamba" or "Mamba2"
d_model = 768
n_layers = 12
base_std = 0.02

# Mamba specific
use_cuda = True # choose True if you can (mamba-ssm installed). else, fallbacks to mamba.py (https://github.com/alxndrTL/mamba.py)

# Mamba2 specific
bias = False
d_head = 64
d_state = 128

# Transformer specific
d_ff = 2048
n_heads = 12
n_kv_heads = 12 # n_heads is MHA, 1 is MQA (multi query), in between is GQA (grouped query attention)
dropout = 0.

pos_emb = "rope" # "absolute" or "rope"
rope_theta = 10000

use_flash_attention = True

# --- muP parameters ---
use_mup = False
mup_base_width = 288

# --- training parameters ---
num_iters = 4768 # 2.5B tokens
total_batch_size = 512
micro_batch_size = 16

# LR and scheduler
schedule = "wsd" # "cosine" or "wsd"

lr = 1.8e-3
lr_warmup_iters = 200

# cosine schedule specific
lr_min = 1.8e-4

# wsd schedule specific
lr_decay_iters = 1000 # 10-20% of num_iters

optimizer = "Ademamix" # "AdamW" or "Ademamix"

weight_decay = 0.1
adam_b1 = 0.9
adam_b2 = 0.95

# Ademamix specific (we set T_alpha_beta3 to T)
adam_b3 = 0.999 # heuristic: half of informations comes from last ln(0.5)/ln(b3) gradients
alpha = 5
T_ab3 = num_iters - lr_decay_iters

max_grad_norm = 1.0

use_torch_compile = True # do not toggle if using Mamba

device = "cuda" # "cpu", "cuda:0", "cuda:1", ...
dtype = "bfloat16" # "float32" or "bfloat16"

# --- saving/checkpointing parameters ---
save_dir = "runs/" # where to save to
ckpt_interval = 1000 # None if you don't want checkpointing
# size of each checkpointing file, in MB : 12 * number of parameters (in M)

ckpt = "" # if you want to restart training from a checkpoint (path/to/model.pth)
start_iter = 0 # specify starting iter (if loading from ckpt_60000, put 60001)

# --- logging and eval parameters ---
log_wandb = True

train_log_interval = 12
eval_val_interval = 12 # also the printing period
eval_val_iters = 50

# ---------------------------------------------

if architecture == "Transformer":
    config = TransformerConfig(d_model=d_model, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_kv_heads, d_ff=d_ff, pos_emb=pos_emb, rope_theta=rope_theta, base_std=base_std, mup=use_mup, mup_base_width=mup_base_width, dropout=dropout, max_len=ctx_len, flash=use_flash_attention)
elif architecture == "Mamba":
    config = MambaConfig(d_model=d_model, n_layers=n_layers, bias=bias, base_std=base_std, mup=use_mup, mup_base_width=mup_base_width, use_cuda=use_cuda)
elif architecture == "Mamba2":
    config = Mamba2Config(d_model=d_model, n_layers=n_layers, d_state=d_state, d_head=d_head, n_groups=1, max_len=ctx_len, bias=bias, base_std=base_std, mup=use_mup, mup_base_width=mup_base_width)
else:
    raise NotImplementedError
