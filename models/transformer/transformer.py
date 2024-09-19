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

    pos_emb: str = "absolute" # absolute, rope
    rope_theta: float = 10000

    mup: bool = False
    mup_base_width: float = 128 # width=d_model

    flash: bool = True

    def __post_init__(self):
        self.architecture = "Transformer"
        
        assert self.d_model % self.n_heads == 0, "d_model must be a multiple of n_heads"
        self.d_head = self.d_model // self.n_heads

        self.n_kv_heads = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0, "number of kv heads must divide the number of heads"
        self.kv_rep = self.n_heads // self.n_kv_heads

        if self.d_ff is None:
            self.d_ff = 4*self.d_model

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

        for layer in self.layers:
            X = layer(X) # (B, L, d_model)
        
        return X

class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.sa_scale = (1 / math.sqrt(2 * config.n_layers))

        #self.attention_norm = RMSNorm(config.d_model, config.norm_eps, config.mup)
        self.sa = SelfAttentionMultiHead(config)
        #self.mlp_norm = RMSNorm(config.d_model, config.norm_eps, config.mup)
        self.mlp = MLP(config)
        
    def forward(self, X):
        # X : (B, L, D)
        # -> Y : (B, L, D)

        #X, cache = self.sa(self.attention_norm(X), cache)
        X = X + self.sa_scale * self.sa(rmsnorm(X))
        #X = X + self.mlp(self.mlp_norm(X))
        X = X + self.mlp(rmsnorm(X))

        return X
    
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

"""
class SelfAttentionMultiHead(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        # key, query, value projections for all heads
        self.query_proj = nn.Linear(config.d_model, config.n_heads * config.d_head, bias=False) # d_query = n_heads*d_head = d_model as in the Transformer paper
        self.key_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.value_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)

        # RoPE embedding
        #self.rotary_emb = rotary_emb

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
            K = self.key_proj(X).view(B, L, self.config.n_kv_heads, self.config.d_head)#.transpose(1, 2) # (B, n_kv_heads, L, d_key)
        else:
            K = X.view(B, L, self.config.n_heads, self.config.d_head)#.transpose(1, 2) # (B, n_kv_heads, L, d_key)

        if not self.config.optimised_attn:
            V = self.value_proj(X).view(B, L, self.config.n_kv_heads, self.config.d_head)#.transpose(1, 2) # (B, n_heads, L, d_head=d_value)
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

        # GQA : expand K and V to compute standard attention
        K = repeat_kv(K, self.config.kv_rep)
        V = repeat_kv(V, self.config.kv_rep)

        # attn computation (torch or manual)
        scale = self.config.mup_attn_mult/self.config.d_head if self.config.mup else 1/math.sqrt(self.config.d_head)

        if self.config.flash:
            attention = F.scaled_dot_product_attention(Q, K, V, attn_mask=None, dropout_p=self.config.dropout if self.training else 0, is_causal=not(L==1), scale=scale)
        else:
            QK_T = Q @ torch.transpose(K, 2, 3) # (B, n_heads, L, L)
            QK_T = QK_T + self.mask[:, :, :L, :L]

            attention_scores = torch.softmax(scale * QK_T, dim=3) # (B, n_heads, L, L)
            attention = self.attn_drop(attention_scores) @ V # (B, n_h, L, d_value=d_head)

        attention = attention.transpose(1, 2) # (B, L, n_heafs, d_head)
        y = attention.contiguous().view(B, L, self.config.d_model) # n_heads * d_head = d_model

        y = self.resid_dropout(self.c_proj(y))

        return y, cache
"""

class SelfAttentionMultiHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_heads
        self.n_embd = config.d_model
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x, cache=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

"""
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
"""

def rmsnorm(x0, eps=1e-5):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

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
