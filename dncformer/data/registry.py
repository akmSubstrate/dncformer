# dncformer/data/registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import random, torch

from ..config import CFG
from .tasks_synthetic import build_copy_gen, build_repeat_copy_gen, build_nback_gen
from .tasks_hf import build_hf_gen

@dataclass
class TaskSpec:
    name: str
    weight: float
    builder: Callable[..., Callable[[int], torch.Tensor]]

# Registry of known tasks → builder
TASK_BUILDERS: Dict[str, Callable[..., Callable[[int], torch.Tensor]]] = {
    "copy": build_copy_gen,
    "repeat": build_repeat_copy_gen,
    "nback": build_nback_gen,
    "hf": build_hf_gen,
}

class StickyMixtureSampler:
    """
    Sticky/chunked mixture over named batch generators.
    Will hold to the chosen task for `chunk_len` calls, then re-sample.
    """
    def __init__(self, gens, weights, names, chunk_len: int = 0):
        self.gens = gens
        t = torch.tensor(list(map(float, weights)), dtype=torch.float32)
        self.p = t / (t.sum() + 1e-8)
        self.names = names
        self.last_name = None
        self._sticky_count = 0
        self._sticky_name = None
        self._sticky_len = max(0, int(chunk_len or 0))
        if self._sticky_len < 0:
            self._sticky_len = 0

    def set_weights(self, weights):
        ws = list(map(float, weights))
        t = torch.tensor(ws, dtype=torch.float32)
        s = float(t.sum().item())
        if s <= 0:
            raise ValueError("Mixture weights must sum to > 0")
        self.p = t / s
        if hasattr(self, "names") and len(self.names) != len(ws):
            print(f"[MixtureSampler] Warning: len(names)={len(self.names)} != len(weights)={len(ws)}")

    def __call__(self, batch: int) -> torch.Tensor:
        if self._sticky_len > 0 and self._sticky_count > 0:
            # re-use sticky choice
            idx = self.names.index(self._sticky_name)
            self._sticky_count -= 1
        else:
            idx = torch.multinomial(self.p, 1).item()
            self._sticky_name = self.names[idx]
            self._sticky_count = self._sticky_len - 1 if self._sticky_len > 0 else 0

        self.last_name = self.names[idx]
        return self.gens[idx](batch)

def build_mixed_sampler(
    tok,
    mixture: Tuple[float, float, float, float],
    *,
    chunk_len: int = 0,
    hf_dataset: Optional[str] = "tatsu-lab/alpaca",
    hf_max_items: int = 5000,
):
    """
    Create a StickyMixtureSampler using registered task builders.
    Task order: ('hf','copy','repeat','nback'), matching legacy plots.
    """
    mx = int(getattr(tok, "model_max_length", 256) or 256)
    pad_id = getattr(tok, "pad_token_id", 0) or 0
    names, gens, wts = [], [], []

    # HF
    if hf_dataset:
        try:
            g_hf = TASK_BUILDERS["hf"](tok, dataset_id=hf_dataset, max_items=hf_max_items, max_len=mx, pad_id=pad_id)
            names.append("hf"); gens.append(g_hf); wts.append(mixture[0])
        except Exception as e:
            print(f"[data] HF loader failed: {type(e).__name__}: {e}. Falling back to synthetics only.")
            names.append("hf"); gens.append(lambda b: torch.full((b, mx), pad_id, dtype=torch.long)); wts.append(0.0)
    else:
        names.append("hf"); gens.append(lambda b: torch.full((b, mx), pad_id, dtype=torch.long)); wts.append(0.0)

    # Synthetics
    g_copy   = TASK_BUILDERS["copy"](max_len=min(mx, 128), vocab=100, pad_id=pad_id)
    g_repeat = TASK_BUILDERS["repeat"](max_len=min(mx, 128), vocab=100, pad_id=pad_id)
    g_nback  = TASK_BUILDERS["nback"](max_len=min(mx, 128), n=5,  vocab=50,  pad_id=pad_id)
    names += ["copy","repeat","nback"]
    gens  += [g_copy, g_repeat, g_nback]
    wts   += [mixture[1], mixture[2], mixture[3]]

    return StickyMixtureSampler(gens, wts, names, chunk_len=int(chunk_len or getattr(CFG, "sticky_mix", 0)))
