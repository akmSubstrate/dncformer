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
        ws = list(map(float, weights))
        t = torch.tensor(ws, dtype=torch.float32, device=self.p.device)
        s = float(t.sum().item())
        if s <= 0: raise ValueError("Mixture weights must sum to > 0")
        self.p = t / s; self.weights = ws
        if hasattr(self, "names") and len(self.names) != len(ws):
            print(f"[MixtureSampler] Warning: len(names)={len(self.names)} != len(weights)={len(ws)}")

def build_mixer(tok, weights, hf_dataset="tatsu-lab/alpaca", hf_max_items=2000) -> MixtureSampler:
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
    return MixtureSampler(gens, wts, names=names)
