# dncformer/data/synthetic.py
from __future__ import annotations
import random, torch

def make_copy_task(batch: int, T: int, vocab: int = 50) -> torch.Tensor:
    return torch.randint(5, vocab, (batch, T), dtype=torch.long)

def make_repeat_copy(batch: int, T: int, repeat_min=2, repeat_max=4,
                     vocab=100, pad_id: int = 0, device: str = "cpu") -> torch.Tensor:
    L = max(1, T // 2)
    x = torch.randint(1, vocab, (batch, L), device=device, dtype=torch.long)
    r = torch.randint(repeat_min, repeat_max + 1, (batch,), device=device)
    out = torch.full((batch, T), pad_id, dtype=torch.long, device=device)
    for i in range(batch):
        seq = x[i].repeat_interleave(int(r[i].item()))
        out[i, :min(T, seq.numel())] = seq[:T]
    return out

def make_n_back(batch: int, T: int, n: int = 3, vocab=50) -> torch.Tensor:
    return torch.randint(1, vocab, (batch, T), dtype=torch.long)

def make_haystack_batch(batch: int, T: int = 256, vocab: int = 1024, sentinel: int = 3):
    assert T >= 12
    x = torch.randint(5, vocab, (batch, T), dtype=torch.long)
    K = torch.randint(5, vocab, (batch,), dtype=torch.long)
    V = torch.randint(5, vocab, (batch,), dtype=torch.long)

    p1 = torch.randint(low=T//8, high=T//2 - 2, size=(batch,))
    x[torch.arange(batch), p1] = K; x[torch.arange(batch), p1 + 1] = V
    p2 = torch.randint(low=3*T//4, high=T - 2, size=(batch,))
    x[torch.arange(batch), p2] = K; x[torch.arange(batch), p2 + 1] = sentinel
    qpos = p2 + 1
    return x, V, qpos
