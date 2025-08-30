# dncformer/data/tasks_synth.py
from __future__ import annotations
import torch, random
from .registry import TaskRegistry

@TaskRegistry.register("copy")
def build_copy(T=128, vocab=100):
    def gen(b): return torch.randint(1, vocab, (b, T), dtype=torch.long)
    return gen

@TaskRegistry.register("repeat")
def build_repeat(T=128, vocab=100, pad_id=0, repeat_min=2, repeat_max=4):
    def gen(b):
        L = max(1, T // 2)
        x = torch.randint(1, vocab, (b, L))
        r = torch.randint(repeat_min, repeat_max + 1, (b,))
        out = torch.full((b, T), pad_id, dtype=torch.long)
        for i in range(b):
            seq = x[i].repeat_interleave(int(r[i].item()))
            out[i, :min(T, seq.numel())] = seq[:T]
        return out
    return gen

@TaskRegistry.register("nback")
def build_nback(T=128, n=5, vocab=50):
    def gen(b): return torch.randint(1, vocab, (b, T), dtype=torch.long)
    return gen

@TaskRegistry.register("add_long")
def build_add_long(T=128, digits=6, base=10, pad_id=0):
    """
    Build sequences like:  'a + b = ?' embedded in a token stream.
    Emits (B,T) ids; model should predict next tokens.
    """
    def gen(b):
        def rnd(d): return ''.join(str(random.randrange(base)) for _ in range(d))
        seqs = []
        for _ in range(b):
            a, b_ = rnd(digits), rnd(digits)
            s = f"{a}+{b_}="
            ids = torch.tensor([ord(c) for c in s], dtype=torch.long)  # simple ASCII for now
            if ids.numel() < T:
                pad = torch.full((T - ids.numel(),), pad_id, dtype=torch.long)
                ids = torch.cat([ids, pad], dim=0)
            else:
                ids = ids[:T]
            seqs.append(ids.unsqueeze(0))
        return torch.cat(seqs, dim=0)
    return gen

@TaskRegistry.register("dyck")
def build_dyck(T=128, depth=4):
    """
    Balanced parentheses strings up to `depth` (Dyck-1). Promotes stack-like memory.
    """
    def gen(b):
        def one():
            s=[]
            cur=0
            while len(s)<T-1:
                if cur<depth and random.random()<0.6: s.append(0); cur+=1       # '('
                else:
                    if cur==0: s.append(0); cur+=1
                    else: s.append(1); cur-=1                                   # ')'
            return torch.tensor(s+[1], dtype=torch.long)[:T]
        xs=[one().unsqueeze(0) for _ in range(b)]
        return torch.cat(xs, dim=0)
    return gen
