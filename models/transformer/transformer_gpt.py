import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
caching is WIP
"""

#todo : mettre des optional la ou on peut
@dataclass
class TransformerGPTConfig:
    d_model: int # D or d_model in comments
    n_layers: int
    n_heads: int
    max_len: int # maximum sequence length (for positional embedding, super attn and mask if no FA)
    dropout: float = 0.
    bias: bool = False
    norm_eps: float = 1e-5
    base_std: float = 0.02
    
    optimised_attn: bool = False
    efficient_attn: bool = False
    super_attn: bool = False # overwrites flash to False

    pos_emb: str = "absolute" # absolute, rope
    rope_theta: float = 10000

    mup: bool = False
    mup_base_width: float = 128 # width=d_model

    flash: bool = True

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be a multiple of n_heads"
        self.d_head = self.d_model // self.n_heads

        # eff/opt/super attn
        self.optimised_attn = self.optimised_attn or self.efficient_attn or self.super_attn
        self.efficient_attn = self.efficient_attn or self.super_attn

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width
            self.mup_attn_mult = math.sqrt(self.d_head) # base_d_head=d_head (kept constant)

class Transformer(nn.Module):
    def __init__(self, config: TransformerGPTConfig):
        super().__init__()

        self.config = config

        if self.config.pos_emb == "absolute":
            self.PE = nn.Embedding(config.max_len, config.d_model)

        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])
        
        self.in_dropout = nn.Dropout(config.dropout)

    def forward(self, X, caches=None, seq_pos=0):
        # X : (B, L, D)

        # Y : (B, L, D)

        _, T, _ = X.size()

        if self.config.pos_emb == "absolute":
            pos_emb = self.PE(torch.arange(seq_pos, seq_pos+T, dtype=torch.long, device=X.device))
            X = self.in_dropout(X + pos_emb)
        else:
            X = self.in_dropout(X)

        for i, layer in enumerate(self.layers):
            X, c = layer(X, caches[i] if caches is not None else None) # (B, L, d_model)

            if caches is not None:
                caches[i] = c
        
        if caches is None:
            return X
        else:
            return X, caches
    
class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerGPTConfig):
        super().__init__()

        self.config = config

        self.attention_norm = RMSNorm(config.d_model, config.norm_eps, config.mup)
        self.sa = SelfAttentionMultiHead(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps, config.mup)
        self.mlp = MLP(config)
        
    def forward(self, X, cache=None):
        # X : (B, L, D)

        # Y : (B, L, D)

        residual = X
        X, cache = self.sa(self.attention_norm(X), cache)
        X = residual + X
        X = X + self.mlp(self.mlp_norm(X))

        return X, cache
    
    def get_empty_cache(self, batch_size):
        return (None, None)
    
class MLP(nn.Module):
    def __init__(self, config: TransformerGPTConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.c_proj  = nn.Linear(4 * config.d_model, config.d_model, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class SelfAttentionMultiHead(nn.Module):
    def __init__(self, config: TransformerGPTConfig):
        super().__init__()

        self.config = config

        # key, query, value projections for all heads
        self.query_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False) # d_query = n_heads*d_head = d_model as in the Transformer paper

        if not self.config.efficient_attn:
            self.key_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)

        if not self.config.optimised_attn:
            self.value_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False)

        # LxL super attention matrix params
        if config.super_attn:
            self.k_in_v_proj = nn.Linear(config.max_len, config.max_len, bias=False)

        # RoPE embedding

        if not config.flash or config.super_attn:
            # compute the mask once and for all here 
            # registrer treats it like a parameter (device, state_dict...) without training
            mask = torch.full((1, 1, config.max_len, config.max_len), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer('mask', mask)

        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.rotary = Rotary(self.config.d_head)

        # regularization
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, X, cache=None):
        # X : (B, L, d_model)

        B, L, _ = X.size()

        # Q,K,V projections
        Q = self.query_proj(X).view(B, L, self.config.n_heads, self.config.d_head)#.transpose(1, 2) # (B, n_heads, L, d_query)

        if not self.config.efficient_attn:
            K = self.key_proj(X).view(B, L, self.config.n_heads, self.config.d_head)#.transpose(1, 2) # (B, n_kv_heads, L, d_key)
        else:
            K = X.view(B, L, self.config.n_heads, self.config.d_head)#.transpose(1, 2) # (B, n_kv_heads, L, d_key)

        if not self.config.optimised_attn:
            V = self.value_proj(X).view(B, L, self.config.n_heads, self.config.d_head)#.transpose(1, 2) # (B, n_heads, L, d_head=d_value)
        else:
            V = X.view(B, L, self.config.n_heads, self.config.d_head)#.transpose(1, 2) # (B, n_heads, L, d_head=d_value)

        # kv cache implementation
        if cache is not None:
            past_keys, past_values = cache
            
            # not first in the sequence
            if past_keys is not None:
                K = torch.cat([past_keys, K], dim=2)
                V = torch.cat([past_values, V], dim=2)
            
            cache = (K, V) # prepare cache for next token
        
        # RoPE
        if self.config.pos_emb == "rope" and cache is None:
            #Q = self.rotary_emb.rotate_queries_or_keys(Q)
            #K = self.rotary_emb.rotate_queries_or_keys(K)

            cos, sin = self.rotary(Q)
            Q = apply_rotary_emb(Q, cos, sin)
            K = apply_rotary_emb(K, cos, sin)

        elif self.config.pos_emb == "rope":
            Q, K = self.rotary_emb.rotate_queries_with_cached_keys(Q, K)

        # attn computation (torch or manual)
        scale = self.config.mup_attn_mult/self.config.d_head if self.config.mup else 1/math.sqrt(self.config.d_head)

        if self.config.flash and not self.config.super_attn:
            attention = F.scaled_dot_product_attention(Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=not(L==1), scale=scale)
        else:
            QK_T = Q @ torch.transpose(K, 2, 3) # (B, n_heads, L, L)
            QK_T = QK_T + self.mask[:, :, :L, :L]

            attention_scores = torch.softmax(scale * QK_T, dim=3) # (B, n_heads, L, L)

            if self.config.super_attn:
                assert L == self.config.max_len, "Super Attention only currently supports a seq len of max_len"
                attention = self.attn_drop(attention_scores) @ self.k_in_v_proj.weight[:L, :L] @ V # (B, n_h, L, d_value=d_head)
            else:
                attention = self.attn_drop(attention_scores) @ V # (B, n_h, L, d_value=d_head)

        attention = attention.transpose(1, 2) # (B, L, n_heafs, d_head)
        y = attention.contiguous().view(B, L, self.config.d_model) # n_heads * d_head = d_model

        y = self.resid_dropout(self.c_proj(y))

        return y, cache

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, use_mup: bool):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)

        if not self.use_mup:
            return output * self.weight
        else:
            return output


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
