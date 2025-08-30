from __future__ import annotations
from typing import Callable, Optional
import torch, random

def _rand_tokens(batch: int, T: int, vocab: int, pad_id: int) -> torch.Tensor:
    x = torch.randint(1, max(2, vocab), (batch, T), dtype=torch.long)
    # ensure no accidental pad symbols if pad_id==0
    if pad_id == 0:
        return x
    x[x == pad_id] = 1
    return x

def build_copy_gen(max_len: int = 128, vocab: int = 100, pad_id: int = 0) -> Callable[[int], torch.Tensor]:
    def gen(b: int) -> torch.Tensor:
        T = max_len
        x = _rand_tokens(b, T, vocab, pad_id)
        # Concatenate "copy" marker? Not needed for LM loss; plain x works.
        return x
    return gen

def build_repeat_copy_gen(max_len: int = 128, vocab: int = 100, pad_id: int = 0) -> Callable[[int], torch.Tensor]:
    def gen(b: int) -> torch.Tensor:
        T = max_len
        half = max(1, T // 2)
        a = _rand_tokens(b, half, vocab, pad_id)
        # repeat "a" → [a || a]
        x = torch.cat([a, a], dim=1)
        if x.size(1) < T:
            pad = torch.full((b, T - x.size(1)), pad_id, dtype=torch.long)
            x = torch.cat([x, pad], dim=1)
        return x[:, :T]
    return gen

def build_nback_gen(max_len: int = 128, n: int = 5, vocab: int = 50, pad_id: int = 0) -> Callable[[int], torch.Tensor]:
    """
    LM-style n-back: emit a stream; the model benefits from memory to compress patterns.
    """
    def gen(b: int) -> torch.Tensor:
        T = max_len
        x = _rand_tokens(b, T, vocab, pad_id)
        # Optionally inject periodic structure that the DNC path likes
        for i in range(n, T, n+1):
            x[:, i-1] = x[:, i-n]  # echo every n steps
        return x
    return gen
