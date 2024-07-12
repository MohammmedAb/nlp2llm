import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class LLama:
    def __init__(self, model: Transformer, model_args: ModelArgs):
        self.model = model
        # self.tokenizer = tokenizer
        self.args = model_args
        self.device = model_args.device

    @staticmethod
    def build(checkpoint_dir: str, load_model: bool,
              max_seq_len: int, batch_size: int, device: str):
        prev_time = time.time()

        # load the checkpoint of the model
        if load_model:
            checkpoints = sorted(Path(checkpoint_dir).glob('*.pth'))
            assert len(checkpoints) > 0
            chk_path = checkpoints[0]
            print(f'loading checkpoint {chk_path}')
            checkpoint = torch.load(chk_path)
            print(f'Loaded checkpoint in {(time.time() - prev_time):.2f} seconds')
            prev_time = time.time()
        if os.path.exists(Path(checkpoint_dir) / 'params.json'):
            with open(Path(checkpoint_dir) / 'params.json', 'r') as f:
                params = json.loads(f.read())
        else:
            params = {}

        model_args: ModelArgs = ModelArgs(
            max_batch_size=batch_size,
            device=device,
            max_seq_len=max_seq_len,
            **params
        )

        ## load the tokenizer
        # tokenizer = SentencePieceProcessor()
        # tokenizer.load(tokenizer_path)
        # model_args.vocab_size = tokenizer.vocab_size()

        # set the tensor type as instructed in the paper
        # if we use GPU, we change the precision to  16-bit half-precision floating-point numbers (also known
        # as float16 or half) on CUDA-enabled GPUs.
        if device == 'cuda':
            torch.set_default_dtype(torch.float16)  # Set default to half precision for CUDA
        else:
            torch.set_default_dtype(torch.bfloat16)  # Set default to bfloat16 for other devices

        model = Transformer(model_args).to(device)

        if load_model:
            # we don't need to load the Rope embeddings
            if 'rope.freqs' in checkpoint:
                del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f'Loaded state dict in {(time.time() - prev_time):.2f}')
        return LLama(model, model_args)

if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LLama.build(
        checkpoint_dir= 'llama/Meta-Llama-3-8B/',
        # tokenizer_path= 'llama/Meta-Llama-3-8B/tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        batch_size=2,
        device=device
    )

    print('Everything is ready!')