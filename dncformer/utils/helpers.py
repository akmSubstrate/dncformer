# dncformer/utils/helpers.py
from __future__ import annotations
from typing import Optional, List, Dict, Any
import contextlib, warnings, numbers
import torch, torch.nn as nn, torch.nn.functional as F

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def requires_grad_(module: nn.Module, flag: bool):
    for p in module.parameters(): p.requires_grad_(flag)
    return module

def causal_mask(T: int, device=None):
    return torch.full((T, T), float("-inf"), device=device).triu(1)

def reduce_gate_tensor(g: torch.Tensor) -> torch.Tensor:
    if g.dim() == 3: return g.mean(dim=-1)
    return g

def gate_metrics(g: torch.Tensor):
    g2 = reduce_gate_tensor(g.detach())
    mean_val = float(g2.mean().item())
    frac = float((g2 > 0.5).float().mean().item())
    eps = 1e-6
    p = g2.clamp(eps, 1 - eps)
    ent = float((-(p * (p + eps).log() + (1 - p) * (1 - p + eps).log())).mean().item())
    return mean_val, frac, ent

def free_head_and_cache(globs: Dict[str, Any] | None = None):
    import gc
    G = globs if globs is not None else globals()
    with contextlib.suppress(Exception): del G['head']
    with contextlib.suppress(Exception): del G['tok']
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def mean_safely(seq):
    xs = [x for x in seq if isinstance(x, (int,float)) and not (x != x)]
    return sum(xs)/len(xs) if xs else float("nan")
