import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # number of heads for the queries 
    n_kv_heads: Optional[int] = 8 # n of heads for key and value  
    vocab_size: int = 1024
    multiple_of: int = 256 #used in the ffn
    ffn_dim_multiplier: Optional[float] = 1.3 # used in the ffn
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

class FeedForward(nn.Module):
    def __init__(
        self, 
        dim:int,
        hidden_dim: int, 
        multiple_of: int,
        ffn_dim_multiplier: Optional[float]
    ):
        super.__init__()
        hidden_dim = int(2* hidden_dim / 3) 

        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)

        hidden_dim = multiple_of * ((hidden_dim + multiple_of -1) // multiple_of)
        


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = Attention(args)

        self.feed_forward = FeedForward(
            dim = args.dim,
            hidden_dim = 4 * args.dim,
            multiple_of= args.multiple_of,
            ffn_dim_multiplier= args.ffn_dim_multiplier 
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_pos: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_pos, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


## The base model
class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        #first layer: input embeddings
        self.tok_embeddings = nn.Embedding(self.vocab_size, params.dim)
        self.layers = torch.nn.ModuleList()

        # Then the hidden layers
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock())

        # the normalization layer
        self.norm = RMSNorm(params.dim, params.norm_eps)
        # to the output
        self.output = nn.Linear(params.dim, self.vocab_size, bias=False)

        # To precompute the frequencies of the Rotary Positional Encodings
        self.freq_cis = precomputed_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.params.rope_theta)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        """ 
        Because we are using the KV chache, we don't need to pass the complete prompt in each itteration
        with the KV Cahce, we only need to pass the latest token becaue the previos ones are already computed and saved in the vram
        (We are gonna use KV chahe just for traning)
        """
        # (B, seq_len)
        _bsz, seqlen = tokens.shape
        # assert seqlen == 1, "Only one token at a time"
        # (B, seq_len) => (B, seq_len, dim)
        h = self.tok_embeddings() # token => embedding

        # Retrive the paris (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freq_cis = self.freq_cis[start_pos: start_pos + seqlen] # we need to compute tht positional encoding
        
        mask = None
        # for traning
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)
        
        for layer in self.layers:
            h = layer(h, start_pos, freq_cis, mask)
        
        h = self.norm
        output  =self.output(h).float()
        return output



