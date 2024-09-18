"""
Universal language model, which accepts as its core a Transformer or a Mamba.

The Transformer is implemented in PyTorch and supports FlashAttention-2/
For Mamba, you have the choice : use mamba.py's pure PyTorch implementation (cf mamba/mamba.py) or use the CUDA implementation.
"""

from typing import Union, List
import inspect
import math

import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from models_bis.transformer.transformer import TransformerConfig, rmsnorm, Transformer #RMSNorm
#from models.transformer.transformer_gpt import Transformer, TransformerGPTConfig as TransformerConfig, RMSNorm
from models.mamba.mamba2 import Mamba2, Mamba2Config
from models.mamba.mamba import Mamba, MambaConfig

class LM(nn.Module):
    def __init__(self, model_config: Union[TransformerConfig, MambaConfig, Mamba2Config], vocab_size: int, rng: torch.Generator):
        super().__init__()

        self.config = model_config
        self.vocab_size = vocab_size
        self.rng = rng

        self.embedding = nn.Embedding(self.vocab_size, self.config.d_model)
        
        if isinstance(self.config, TransformerConfig):
            self.core = Transformer(self.config)
        elif isinstance(self.config, MambaConfig):
            self.core = Mamba(self.config)
        elif isinstance(self.config, Mamba2Config):
            self.core = Mamba2(self.config)
        else:
            raise NotImplementedError

        #self.out_norm = RMSNorm(self.config.d_model, self.config.norm_eps, self.config.mup)

        self.lm_head = nn.Linear(self.config.d_model, self.vocab_size, bias=False)
        self.embedding.weight = self.lm_head.weight

        if self.config.mup and isinstance(self.config, TransformerConfig):
            for pn, p in self.named_parameters():
                if any(pn.endswith(w) for w in ['sa.key_proj.weight','sa.value_proj.weight', 'sa.c_proj.weight', 'mlp.fc_1.weight', 'mlp.fc_2.weight', 'mlp.fc_3.weight']):
                    std = self.config.base_std

                    if any(pn.endswith(w) for w in ['sa.c_proj.weight', 'mlp.fc_3.weight']):
                        std = std / math.sqrt(2 * self.config.n_layers)
                    
                    torch.nn.init.normal_(p, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult), generator=self.rng)
                elif pn.endswith('sa.query_proj.weight'):
                    torch.nn.init.zeros_(p) # init query proj to zeros
                elif pn == "embedding.weight":
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std, generator=self.rng)
                elif pn == "lm_head.weight":
                    torch.nn.init.zeros_(p)
                elif pn == "core.PE.weight":
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std, generator=self.rng)
                else:
                    # here, we only have biases and rotary_emb.freqs
                    assert p.dim() == 1, f"a 2d param ({pn}) has not been filtered out for init. please check."

                    if "bias" in pn:
                        torch.nn.init.zeros_(p)

        elif self.config.mup and isinstance(self.config, MambaConfig):
            for pn, p in self.named_parameters():
                if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.x_delta_proj.weight', 'mixer.dt_proj.weight', 'mixer.out_proj.weight', 'mixer.x_proj.weight']):
                    std = self.config.base_std

                    if 'mixer.out_proj.weight' in pn:
                        std = std / math.sqrt(2 * self.config.n_layers)

                    if 'mixer.dt_proj.weight' in pn:
                        std = self.config.dt_rank**-0.5 * self.config.dt_scale

                    torch.nn.init.normal_(p, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult), generator=self.rng)
                elif 'mixer.x_BC_proj.weight' in pn:
                    torch.nn.init.zeros_(p[self.config.dt_rank:])
                elif 'mixer.conv1d.weight' in pn:
                    torch.nn.init.zeros_(p)
                elif pn == "embedding.weight":
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std, generator=self.rng)
                elif pn == "lm_head.weight":
                    torch.nn.init.zeros_(p)
                elif any(pn.endswith(w) for w in ['mixer.A_log', 'mixer.D']):
                    pass
                else:
                    # here, we only have biases
                    assert p.dim() == 1, f"a 2d param ({pn}) has not been filtered out for init. please check."

                    if ("in_proj.bias" in pn) or ("out_proj.bias" in pn):
                        torch.nn.init.zeros_(p)
            
        elif self.config.mup and isinstance(self.config, Mamba2Config):
            for pn, p in self.named_parameters():
                
                if any(pn.endswith(w) for w in ['mixer.in_proj.weight', 'mixer.out_proj.weight']):
                    std = self.config.base_std

                    if 'mixer.out_proj.weight' in pn:
                        std = std / math.sqrt(2 * self.config.n_layers)

                    torch.nn.init.normal_(p, mean=0.0, std=std / math.sqrt(self.config.mup_width_mult), generator=self.rng)
                
                elif 'mixer.conv1d.weight' in pn:
                    torch.nn.init.zeros_(p)
                
                elif pn == "embedding.weight":
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std, generator=self.rng)
                
                elif pn == "lm_head.weight":
                    torch.nn.init.zeros_(p)
                
                elif any(pn.endswith(w) for w in ['mixer.A_log', 'mixer.D', 'mixer.dt_bias']):
                    pass
                else:
                    # here, we only have biases
                    assert p.dim() == 1, f"a 2d param ({pn}) has not been filtered out for init. please check."

                    if ("in_proj.bias" in pn) or ("out_proj.bias" in pn):
                        torch.nn.init.zeros_(p)

        else: # transformer and mamba
            self.apply(self._init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith('fc_3.weight') or pn.endswith('c_proj.weight') or pn.endswith('mixer.out_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0, std=self.config.base_std/math.sqrt(2 * self.config.n_layers), generator=self.rng)

    def forward(self, tokens, targets=None, caches=None, return_logits=False, seq_pos=0):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)
        if caches is None:
            x = self.core(x)
        else:
            x, caches = self.core(x, caches, seq_pos)
        #x = self.out_norm(x)
        x = rmsnorm(x)

        if self.config.mup:
            x = x / self.config.mup_width_mult

        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return None, loss
        else:
            return None, caches
    
    # non-muP init
    # taken from llama2.c
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std, generator=self.rng)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.base_std, generator=self.rng)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
        return optimizer

def load_model(load_dir, device="cuda"):
    config_dir = os.path.join(load_dir, 'config.json')
    checkpoint_dir = os.path.join(load_dir, 'model.pth')

    config_json = json.load(open(config_dir))
    architecture = config_json['architecture']
    del config_json['architecture']

    if architecture == "Transformer":
        config = TransformerConfig(**config_json)
    elif architecture == "Mamba":
        config = MambaConfig(**config_json)
    elif architecture == "Mamba2":
        config = Mamba2Config(**config_json)
    else:
        raise NotImplementedError

    model = LM(config, vocab_size=52+3).to(device)
    checkpoint = torch.load(checkpoint_dir, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model
