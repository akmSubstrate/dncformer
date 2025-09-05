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

def make_n_back(batch: int, T: int, n: int = 3, vocab: int = 50, q_token: int = 0, token_min: int = 5,) -> torch.Tensor:
    """
    Emit a random token stream but insert (Q, A) pairs where A == token seen n steps earlier
    - Q is a special marker token (default=0)
    - Random symbols are drawn from [token_min, vocab)
    """
    assert n >= 1, "n must be >= 1"
    assert vocab > token_min + 1, "increase vocab or lower token_min"
    out = torch.full((batch, T), 0, dtype=torch.long)

    for b in range(batch):
        seq: list[int] = []
        # seed with at least n random symbols
        seed_len = min(n, T)
        for _ in range(seed_len):
            seq.append(int(torch.randint(token_min, vocab, (1,)).item()))
        # fill the rest, occasionally inserting a (Q, A) pair
        while len(seq) < T:
            can_insert_qa = (len(seq) >= n) and (len(seq) <= T - 2)
            if can_insert_qa and (random.random() < 0.30):
                seq.append(q_token)
                seq.append(seq[-n])
            else:
                seq.append(int(torch.randint(token_min, vocab, (1,)).item()))
        out[b, :] = torch.tensor(seq[:T], dtype=torch.long)
    return out

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
