import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
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

    device: str = None


def precomputed_theta_pos_frequencies(head_dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(seq_len, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  
    return freqs_cis

def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    device: str   
):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x).to(device)


def repeat_kv(x: torch.Tensor, n_rep: int)-> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            # (B, seq_len, n_kv_heads, 1, head_dim)
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )

    

class RMSNorm(nn.Module):
    def __init__(self, dim:int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps # small epsilon value
        self.weight = nn.Parameter(torch.ones(dim)) # learnable parameters

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Attention(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        self.device = args.device
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # number of heads for the queries
        self.local_heads = args.n_heads
        # indicates how many times the heads of keys and value should be repeated to match the head of the query 
        self.n_rep = self.local_heads // self.n_kv_heads
        # indicated the dimention of each head
        self.head_dim = args.dim // args.n_heads

        # input = embedding dim, output = we multiply the head dim by the number of heads to have multiple attention heads to learn different representations and focus on different parts of the input sequence for each head.   
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias = False) 
        # here we use kv cach heads 
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias= False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        # for the output layer, we concatenated space of all attention heads back into the model's dimensionality space.]
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim ,bias = False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (B, 1, Dim)

        xq, = self.wq(x) # (B, 1, H_q * head_size)
        xk, xv = self.wk(x), self.wv(x)

        #reshape the tensors
        xq = xq.view(batch_size, seq_len, self.local_heads)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).to(x.device)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).to(x.device)

        xq = apply_rotary_emb(xq, freq_complex, device = x.device)
        xq = apply_rotary_emb(xq, freq_complex, device = x.device)

        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, h_q, 1, head_dim) @ (B, h_kv, seq_len-kv, head_dim) -> (B, h_q, 1, seq_len-kv)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, h_q, 1, seq_len) @ (B, h_q, seq_len-kv, head_dim) --> (b, h-q, q, head_dim)
        output = torch.matmul(scores, values)

        # (B, h_q, 1, head_dim) -> (B, 1, h_q, head_dim) -> ()
        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, dim) -> (B, 1, dim)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Assuming 'hidden_dim' is calculated as per your specifications
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)  # Applying your specific transformation
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        #hidden_dim = int(2 * hidden_dim / 3)  # Applying your specific transformation
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)  # This layer seems to be missing in your original setup
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)  # Corrected to match checkpoint

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))  # Apply first transformation
        x_V = self.w3(x) 
        x = swish * x_V        # Apply contraction to original dimension
        x = self.w2(x)  # Apply optional additional transformation
        return x

        


class TransformerBlock(nn.Module):
    def __init__(self,  args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = Attention(args)

        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_pos: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


## The base model
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        #first layer: input embeddings
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()

        # Then the hidden layers
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

        # the normalization layer
        self.norm = RMSNorm(args.dim, args.norm_eps)
        # to the output
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # To precompute the frequencies of the Rotary Positional Encodings
        self.freq_cis = precomputed_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, theta=self.args.rope_theta)

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
            h = layer(h, start_pos, freq_cis)
        
        h = self.norm
        output  =self.output(h).float()
        return output

print('All worked well')

