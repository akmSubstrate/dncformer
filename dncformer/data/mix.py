from __future__ import annotations
from typing import List, Optional, Tuple
import random, torch

class MixtureSampler:
    def __init__(self, gens: List, weights: List[float], names: Optional[List[str]] = None):
        self.gens = gens
        self.weights = list(map(float, weights))
        p = torch.tensor(self.weights, dtype=torch.float32)
        self.p = p / (p.sum() + 1e-8)
        self.names = names if names is not None else [f"g{i}" for i in range(len(gens))]
        self.last_name = None

    def __call__(self, batch: int) -> torch.Tensor:
        idx = torch.multinomial(self.p, 1).item()
        self.last_name = self.names[idx]
        return self.gens[idx](batch)

    def set_weights(self, weights):
        """
        Update mixture probabilities at runtime (used by schedules)

        Accepts weights of arbitrary length and aligns them to the number of
        active generators
        """
        import torch as _t

        ws = list(map(float, weights))
        n = len(self.gens)

        if len(ws) != n:
            # Case 1: schedule includes HF but mixer has only synthetics
            if len(ws) == n + 1 and ("hf" not in self.names):
                ws = ws[1:]  # drop HF

            # Case 2: schedule longer than gens; truncate sensibly
            elif len(ws) > n:
                # If 'hf' is absent, prefer to keep the *last* n entries (copy, repeat, nback)
                ws = ws[-n:] if ("hf" not in self.names) else ws[:n]

            # Case 3: schedule shorter than gens; pad uniformly
            else:
                pad_val = (sum(ws) / len(ws)) if ws else 1.0
                ws = ws + [pad_val] * (n - len(ws))

            print(f"[MixtureSampler] aligned schedule weights to {n} gens (names={self.names}): {ws}")

        # normalize and update tensor
        t = _t.tensor(ws, dtype=_t.float32)
        s = float(t.sum().item())
        if s <= 0:
            raise ValueError("Mixture weights must sum to > 0")
        self.p = t / s
        self.weights = ws

class StickyMixtureSampler:
    """
    Wraps a MixtureSampler and holds on to the same index for `chunk_steps` batches.
    - Respects MixtureSampler.set_weights() updates (probabilities change for the *next* pick).
    - Does not reset mid-chunk on schedule changes (keeps things stable).
    """
    def __init__(self, base_sampler, chunk_steps: int, reset_on_set_weights: bool = False):
        assert chunk_steps >= 1
        self.base = base_sampler
        self.chunk_steps = int(chunk_steps)
        self.reset_on_set_weights = bool(reset_on_set_weights)
        self._left = 0
        self._idx = None
        self.last_name = None

    def __len__(self):
        return len(self.base.gens)

    def set_weights(self, weights):
        # propagate to underlying sampler
        self.base.set_weights(weights)
        if self.reset_on_set_weights:
            self._left = 0  # force a new choice next call

    def _pick_new_index(self):
        # draw one from the base sampler's categorical distribution
        import torch as _t
        p = getattr(self.base, "p", None)
        if p is None:
            # fall back to uniform if base has a different representation
            self._idx = 0
        else:
            self._idx = int(_t.multinomial(p, 1).item())
        self._left = self.chunk_steps

    def __call__(self, batch: int):
        if self._left <= 0 or self._idx is None:
            self._pick_new_index()
        self._left -= 1
        # delegate to the underlying generator
        self.last_name = self.base.names[self._idx]
        return self.base.gens[self._idx](batch)
