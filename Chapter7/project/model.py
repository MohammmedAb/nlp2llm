import torch.nn as nn
import inspect
from dataclasses import dataclass
import torch
import torch.nn.functional as F

"""
Model Info:
- GPT model with 12 layers, 12 heads, 768 hidden size
- Number of parameters: 162M
- Dataset: Bigcode/StarCoder dataset
- Tokenizer: cl100k_base
- Training data: 185M tokens
- Batch size: 512k tokens
"""

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd*4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd*4, config.n_embd)

        self.c_proj.NANOGPT_SCALE_INIT = 1 # init scale for the weights

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CasualSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd) # turn the input embedding to query, key, value vectors
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # project the output back to the embedding size

        # Attributes
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        

@dataclass
class GPTConfig:
    block_size: int= 1024 # maximum length of the input sequence
    vocab_size: int = 100277 # size of the vocabulary
    n_layer: int = 12 # Transformer layers
    n_head: int = 12 # Attention heads
    n_embd: int = 768 # Embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2* self.config.n_layer) ** -0.5 # scale initialization by the number of layers
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        


    def forward(self, idx, target = None):
        B, T = idx.size()
        assert T <= self.config.block_size, 'Cannot forward, model block size is exhausted'

        pos = torch.arange(0, T, dtype=torch.long)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss
    
    def config_optimizer(self, weight_decay, learning_rate, device):
        # start with all the parameters that require a gradient
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_parameters = [p for n, p in param_dict.items() if p.dim() >= 2] # emdeddings and metrices in the linear layers
        nondecay_parameters = [p for n, p in param_dict.items() if p.dim() < 2] # biases (one dimentional tensors)
        optim_groups = [
            {'params': decay_parameters, 'weight_decay': weight_decay},
            {'params': nondecay_parameters, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_parameters)
        num_nondecay_params = sum(p.numel() for p in nondecay_parameters)
        print(f"number decayed parameter tensor: {len(decay_parameters)} with {num_decay_params:,} parameters")
        print(f"number non-decayed parameter tensor: {len(nondecay_parameters)} with {num_nondecay_params:,} parameters")

        fused_available =  'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused adam: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

