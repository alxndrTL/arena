import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.rotary_embedding import RotaryEmbedding

"""
caching is WIP
"""

#todo : mettre des optional la ou on peut
@dataclass
class TransformerConfig:
    d_model: int # D or d_model in comments
    n_layers: int
    n_heads: int
    max_len: int # maximum sequence length (for positional embedding, super attn and mask if no FA)
    dropout: float = 0.
    bias: bool = False
    norm_eps: float = 1e-5
    base_std: float = 0.02
    
    d_ff: int = None
    n_kv_heads: Optional[int] = None # None=n_heads is MHA, 1 is MQA (multi query attention), in between is GQA (grouped)
    
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

        self.n_kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0, "number of kv heads must divide the number of heads"
        self.kv_rep = self.n_heads // self.n_kv_heads

        if self.d_ff is None:
            self.d_ff = 4*self.d_model

        # eff/opt/super attn
        self.optimised_attn = self.optimised_attn or self.efficient_attn or self.super_attn
        self.efficient_attn = self.efficient_attn or self.super_attn

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width
            self.mup_attn_mult = math.sqrt(self.d_head) # base_d_head=d_head (kept constant)

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        if self.config.pos_emb == "absolute":
            self.PE = nn.Embedding(config.max_len, config.d_model)
            self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])

        elif self.config.pos_emb == "rope":
            PE = RotaryEmbedding(dim=(self.config.d_model//self.config.n_heads)//2, theta=self.config.rope_theta)
            self.layers = nn.ModuleList([DecoderLayer(config, PE) for _ in range(config.n_layers)])

        else:
            raise NotImplementedError
        
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
    def __init__(self, config: TransformerConfig, rotary_emb: RotaryEmbedding = None):
        super().__init__()

        self.config = config

        self.attention_norm = RMSNorm(config.d_model, config.norm_eps, config.mup)
        self.sa = SelfAttentionMultiHead(config, rotary_emb)
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
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.fc_1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc_2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.fc_3 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc_2(F.silu(self.fc_1(x)) * self.fc_3(x)))

class SelfAttentionMultiHead(nn.Module):
    def __init__(self, config: TransformerConfig, rotary_emb: RotaryEmbedding = None):
        super().__init__()

        self.config = config

        # key, query, value projections for all heads
        self.query_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False) # d_query = n_heads*d_head = d_model as in the Transformer paper

        if not self.config.efficient_attn:
            self.key_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)

        if not self.config.optimised_attn:
            self.value_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)

        # LxL super attention matrix params
        if config.super_attn:
            self.k_in_v_proj = nn.Linear(config.max_len, config.max_len, bias=False)

        # RoPE embedding
        self.rotary_emb = rotary_emb

        if not config.flash or config.super_attn:
            # compute the mask once and for all here 
            # registrer treats it like a parameter (device, state_dict...) without training
            mask = torch.full((1, 1, config.max_len, config.max_len), float('-inf'))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer('mask', mask)

        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # regularization
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, X, cache=None):
        # X : (B, L, d_model)

        B, L, _ = X.size()

        # Q,K,V projections
        Q = self.query_proj(X).view(B, L, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, L, d_query)

        if not self.config.efficient_attn:
            K = self.key_proj(X).view(B, L, self.config.n_kv_heads, self.config.d_head).transpose(1, 2) # (B, n_kv_heads, L, d_key)
        else:
            K = X.view(B, L, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_kv_heads, L, d_key)

        if not self.config.optimised_attn:
            V = self.value_proj(X).view(B, L, self.config.n_kv_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, L, d_head=d_value)
        else:
            V = X.view(B, L, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, L, d_head=d_value)

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
            Q = self.rotary_emb.rotate_queries_or_keys(Q)
            K = self.rotary_emb.rotate_queries_or_keys(K)
        elif self.config.pos_emb == "rope":
            Q, K = self.rotary_emb.rotate_queries_with_cached_keys(Q, K)

        # GQA : expand K and V to compute standard attention
        if not self.config.efficient_attn:
            K = repeat_kv(K, self.config.kv_rep)
        if not self.config.optimised_attn:
            V = repeat_kv(V, self.config.kv_rep)

        # attn computation (torch or manual)
        scale = self.config.mup_attn_mult/self.config.d_head if self.config.mup else 1/math.sqrt(self.config.d_head)

        if self.config.flash and not self.config.super_attn:
            attention = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=not(L==1), scale=scale)
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

# taken from modeling_jamba.py (jamba official implementation)
# the same as the one in llama2.c model.py, but dim of repeat is 1 instead of 2
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim).
    """

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
