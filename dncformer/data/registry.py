# dncformer/data/registry.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Any, Iterable

import torch
import random

TaskFn = Callable[[int], torch.Tensor]  # returns (B, T) long tensor

@dataclass
class DatasetSpec:
    name: str                 # e.g. "hf_text", "hf_instruct", "copy", "repeat", "nback", "add_long"
    params: dict = field(default_factory=dict)
    weight: float = 1.0       # relative mixer weight
    alias: Optional[str] = None  # nice name for TB (“copy”, “hf/tinystories”, etc.)

@dataclass
class MixerSpec:
    sticky_chunk: int = 0     # 0 disables stickiness (per-step sampling)
    order: Optional[List[str]] = None  # optional fixed rotation order by dataset alias
    normalize: bool = True

class TaskRegistry:
    _TASKS: Dict[str, Callable[..., TaskFn]] = {}

    @classmethod
    def register(cls, name: str):
        def deco(builder: Callable[..., TaskFn]):
            cls._TASKS[name] = builder
            return builder
        return deco

    @classmethod
    def build(cls, spec: DatasetSpec) -> Tuple[str, TaskFn]:
        if spec.name not in cls._TASKS:
            raise KeyError(f"Unknown task '{spec.name}'. Registered: {sorted(cls._TASKS)}")
        fn = cls._TASKS[spec.name](**spec.params)
        alias = spec.alias or spec.name
        return alias, fn

def build_pipeline(specs: List[DatasetSpec], mixer: MixerSpec):
    """
    Returns (sampler_call, names) where sampler_call(batch) -> torch.LongTensor[B,T].
    Exposes sampler_call.last_name for logging.
    """
    names, fns, wts = [], [], []
    for s in specs:
        alias, fn = TaskRegistry.build(s)
        names.append(alias); fns.append(fn); wts.append(float(s.weight))

    # normalize weights
    S = sum(wts) + 1e-8
    p = [w/S for w in wts]

    if mixer.sticky_chunk > 0:
        return _make_sticky_mixture(fns, names, p, mixer.sticky_chunk, mixer.order), names
    else:
        return _make_weighted_mixture(fns, names, p), names

def _make_weighted_mixture(fns, names, p):
    p_t = torch.tensor(p, dtype=torch.float32)

    class _Sampler:
        last_name = None
        def __call__(self, batch: int) -> torch.Tensor:
            idx = torch.multinomial(p_t, 1).item()
            self.last_name = names[idx]
            return fns[idx](batch)
    return _Sampler(), names

def _make_sticky_mixture(fns, names, p, chunk_len: int, order: Optional[List[str]]):
    p_t = torch.tensor(p, dtype=torch.float32)
    idx_map = {n: i for i, n in enumerate(names)}

    seq: List[int]
    if order:
        seq = [idx_map[n] for n in order]
    else:
        seq = list(range(len(names)))

    class _StickySampler:
        last_name = None
        _step_in_chunk = 0
        _cursor = 0
        def __call__(self, batch: int) -> torch.Tensor:
            nonlocal seq
            # rotate on boundaries
            if self._step_in_chunk == 0:
                self._cursor %= len(seq)
                self._active = seq[self._cursor]
                self._cursor += 1
            self.last_name = names[self._active]
            out = fns[self._active](batch)
            self._step_in_chunk = (self._step_in_chunk + 1) % chunk_len
            return out
    return _StickySampler(), names
