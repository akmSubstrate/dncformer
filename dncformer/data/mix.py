# dncformer/data/mix.py
from __future__ import annotations
from typing import List, Optional, Tuple
import random, torch
from .synthetic import make_copy_task, make_repeat_copy, make_n_back
from .hf import hf_instruction_loader, make_hf_batch

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
        Update mixture probabilities at runtime (used by schedules).

        Accepts weights of arbitrary length and aligns them to the number of
        active generators. If a 4-tuple (HF, copy, repeat, nback) is supplied
        but the HF generator is *not* present, we drop the first element.
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

def build_mixer(tok, weights, hf_dataset="tatsu-lab/alpaca", hf_max_items=2000, sticky_chunk_steps: int | None = None):
    mx = int(getattr(tok, "model_max_length", 256) or 256)
    pad_id = getattr(tok, "pad_token_id", 0) or 0
    gens, wts, names = [], [], []
    gen_hf, hf_ok, hf_reason = None, False, ""

    if hf_dataset:
        try:
            if "alpaca" in hf_dataset.lower():
                pairs = hf_instruction_loader(hf_dataset, "train", ("instruction","output"), max_items=hf_max_items)
                if pairs:
                    def gen_hf(b): return make_hf_batch(tok, pairs, b, max_len=mx)
                    hf_ok = True
                else:
                    hf_reason = "no pairs"
            else:
                from datasets import load_dataset
                try:
                    ds = load_dataset(hf_dataset, split="train", streaming=True)
                except Exception:
                    ds = load_dataset(hf_dataset, split="train")
                text_key = None
                for k in ("text","content","story","document","body","article"):
                    if getattr(ds, "features", None) and k in ds.features:
                        text_key = k; break
                if text_key is None:
                    try:
                        first_ex = next(iter(ds))
                        for k in ("text","content","story","document","body","article"):
                            if k in first_ex: text_key = k; break
                    except StopIteration:
                        pass
                if text_key is None:
                    hf_reason = "no text field"
                else:
                    samples = []
                    for ex in ds:
                        txt = ex.get(text_key, None)
                        if not txt: continue
                        ids = tok(txt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
                        if ids.numel() < 8: continue
                        samples.append(ids.cpu())
                        if len(samples) >= int(hf_max_items): break
                    if samples:
                        def gen_hf(b: int):
                            out = []
                            for _ in range(b):
                                ids = random.choice(samples); n = ids.numel()
                                if n >= mx:
                                    s = random.randint(0, n - mx); seq = ids[s:s+mx]
                                else:
                                    pad = torch.full((mx - n,), pad_id, dtype=torch.long); seq = torch.cat([ids, pad], dim=0)
                                out.append(seq.unsqueeze(0))
                            return torch.cat(out, dim=0)
                        hf_ok = True
                    else:
                        hf_reason = "0 samples"
        except Exception as e:
            hf_reason = f"{type(e).__name__}: {e}"

    if hf_ok and gen_hf is not None:
        gens.append(gen_hf); wts.append(weights[0]); names.append("hf")
        s_w = list(weights[1:])
    else:
        if hf_dataset:
            print(f"HF dataset unavailable or empty; using synthetic only. reason={hf_reason}")
        s_w = list(weights[1:])

    def gen_copy(b):   return make_copy_task(b, T=min(mx, 128), vocab=100)
    def gen_repeat(b): return make_repeat_copy(b, T=min(mx, 128), vocab=100, pad_id=pad_id, device="cpu")
    def gen_nback(b):  return make_n_back(b, T=min(mx, 128), n=5, vocab=50)

    gens.extend([gen_copy, gen_repeat, gen_nback])
    wts.extend(s_w); names.extend(["copy","repeat","nback"])

    sampler = MixtureSampler(gens, wts, names=names)
    # optional chunking (prefer explicit arg; else CFG.mixture_chunk_steps if present)
    chunk = None
    if isinstance(sticky_chunk_steps, int) and sticky_chunk_steps > 1:
        chunk = sticky_chunk_steps
    else:
        # optional: pull from global config if present (correct package level)
        try:
            from ..config import CFG  # fixed import
            c = int(getattr(CFG, "mixture_chunk_steps", 0))
            if c > 1:
                chunk = c
        except Exception:
            pass

    if chunk:
        sampler = StickyMixtureSampler(sampler, chunk_steps=int(chunk), reset_on_set_weights=False)

    return sampler
