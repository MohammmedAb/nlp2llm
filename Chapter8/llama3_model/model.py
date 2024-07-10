import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 1024
    ffn_dim_multiplier: Optional[float] = 1.3
    norm_eps: float = 1e-05
    rope_theta: float = 500000

    #for the kv cach
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6):
        super().__init__
        self.eps = eps # small epsilon value
        self.weight = nn.Parameter(torch.ones) # learnable parameters

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # number of heads for the queries
        self.local_heads = args.n_heads
        # indicates how many times the heads of keys and value should be repeated to match the head of the query 
        self.n_rep = self.local_heads // self.n_kv_heads
        # indicated the dimention of each head
        self.head_dim = args.dim // args.n_heads

        # input = embedding dim, output = we multiply the head dim by the number of heads to have multiple attention heads to learn different representations and focus on different parts of the input sequence for each head.   
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, dias = False) 
        # here we use kv cach heads 
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias= False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        # for the output layer, we concatenated space of all attention heads back into the model's dimensionality space.]
        self.wo = nn.Linear(args.n_heads * self.head_dim, bias = False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.tensor, start_pos: int, freq_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape

        xq, = self.wq(x) # (B, 1, H_q * head_size)
        xk, xv = self.wk(x), self.wv(x)

        #reshape the tensors

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = Attention(args)
        slef.feed

class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(self.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()

        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock())





