from dataclasses import dataclass
import inspect
import math
import time
import torch
import torch.nn as nn
from torch.nn import functional as F


class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # projecting the input vector into key query value, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) 
        
        # output projection => projecting the output of the self-attention computation back into the original embedding space
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

        

class MLP(nn.Module):
    def __init__(self, config):    
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd) # fully connected layer
        self.gelu = nn.GELU(approximate='tanh') # activation function
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd) # fully connected layer
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # layernorm 1
        self.attn = CasualSelfAttention(config) # multi-head self-attention
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # feedforward neural network
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
        

@dataclass
class GPTConfig:
    block_size: int= 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer block (hidden layers)
            ln_f = nn.LayerNorm(config.n_embd) # layernorm after the last block
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # language model head
        self.transformer.wte.weight = self.lm_head.weight # weight tying

        # init weights for each submodule
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
            

    # forward pass
    def forward(self, idx, target = None):
        # idx: [B, T] with B being the batch size and T the length of the sequence

        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is {self.config.block_size}"

        pos = torch. arange(0, T, dtype=torch.long ,device=idx.device) # tensor of positions
        pos_emb = self.transformer.wpe(pos) # Positional embedding
        tok_emb = self.transformer.wte(idx) # Token embedding
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x) # apply each block to the input x
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss



    # Load weights from a pretrained GPT2 model to initialize our model
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])  # copy from hf to our model

        return model
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
    
import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# gradient accumulation: we use it to simulate a larger batch size
total_batch_size = 1024 # is supposed to be => 2^19, ~0.5M tokens
B = 4 # Batch size  
T = 32 # sequence length
assert total_batch_size % (B * T) == 0, "Make sure total_batch_size is divisible by B * T"
grad_acc_steps = total_batch_size // (B * T)
print(f"Total desired batch size: {total_batch_size:,}")
print(f"Gradient accumulation steps: {grad_acc_steps}")

train_loader = DataLoaderLite(B=B, T=T)

# get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1) linear warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) min learning rate 
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
# optimize
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.config_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)
for step in range(max_steps):
    # model.train()
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0
    for micro_step in range(grad_acc_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y) # the inital loss should be: -log(1/vocab_size) = -log(1/50257) = ~10.82
        loss = loss / grad_acc_steps
        loss_accum += loss.detach() 
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # cosine learning rate schedule
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    t1 = time.time()
    dt = t1 - t0
    print(f'iter {step} | loss: {loss_accum.item()} | lr: {lr:.4e} | norm: {norm: .4f} | dt: {dt*10000: .2f}ms | ') 
import sys; sys.exit(0)


#prefix tokens as starting point for the model
model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello i'm a language model, ")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens
x = tokens.to(device)

# generate some text
torch.manual_seed(42)
# torch.cuda.manual_seed(42)
while x.size(1) < max_length:

    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :] # take the logits at the last position

        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 10, dim=-1)

        ix = torch.multinomial(topk_probs, 1) # sample from the top-k probabilities

        xcol = torch.gather(topk_indices, -1, ix) 
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print('>',decoded)
