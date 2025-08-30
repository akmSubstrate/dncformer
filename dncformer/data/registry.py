from __future__ import annotations
from typing import Any, Dict, List
import random, torch
from .mix import MixtureSampler, StickyMixtureSampler, build_mixer
from .synthetic import make_copy_task, make_repeat_copy, make_n_back
from .hf import hf_instruction_loader, make_hf_batch
from ..config import CFG

def _normalize_weights(ws: List[float]) -> List[float]:
    w = [float(x) for x in ws]
    s = sum(x for x in w if x > 0)
    return [x / s for x in w] if s > 0 else [1.0/len(w)] * len(w)

def _build_hf_gen(tok, dataset: str, mx: int, pad_id: int, max_items: int, text_key: str|None=None):
    """
    Returns a callable gen(b) or None if loading failed.
    Two paths:
      - 'alpaca' style instruction/output via hf_instruction_loader
      - generic text datasets via datasets.load_dataset
    """
    if dataset and "alpaca" in dataset.lower():
        pairs = hf_instruction_loader(dataset, "train", ("instruction","output"), max_items=max_items)
        if pairs:
            def gen_hf(b, _pairs=pairs): return make_hf_batch(tok, _pairs, b, max_len=mx)
            return gen_hf
        else:
            print(f"[registry] HF/alpaca '{dataset}' empty → skipping")
            return None
    # generic
    try:
        from datasets import load_dataset
        ds = None
        try:
            ds = load_dataset(dataset, split="train", streaming=True)
        except Exception:
            ds = load_dataset(dataset, split="train")
    except Exception as e:
        print(f"[registry] load_dataset failed for '{dataset}': {e}")
        return None

    # figure out text key if not provided
    if text_key is None:
        if getattr(ds, "features", None):
            for k in ("text","content","story","document","body","article","code"):
                if k in ds.features:
                    text_key = k; break

    samples = []
    try:
        it = iter(ds)
        if text_key is None:
            first = next(it)
            for k in ("text","content","story","document","body","article","code"):
                if k in first:
                    text_key = k; break
            if text_key:
                ids = tok(first[text_key], return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
                if ids.numel() >= 8: samples.append(ids.cpu())
        for ex in it:
            if not text_key: break
            txt = ex.get(text_key, None)
            if not txt: continue
            ids = tok(txt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
            if ids.numel() >= 8:
                samples.append(ids.cpu())
            if len(samples) >= int(max_items): break
    except Exception:
        pass

    if not samples:
        print(f"[registry] HF '{dataset}' yielded 0 samples → skipping")
        return None

    def gen_hf_generic(b, _samples=samples, _mx=mx, _pad=pad_id):
        out = []
        for _ in range(b):
            ids = random.choice(_samples); n = ids.numel()
            if n >= _mx:
                s = random.randint(0, n - _mx); seq = ids[s:s+_mx]
            else:
                pad = torch.full((_mx - n,), _pad, dtype=torch.long); seq = torch.cat([ids, pad], dim=0)
            out.append(seq.unsqueeze(0))
        return torch.cat(out, dim=0)
    return gen_hf_generic

def build_sampler_from_cfg(tok, data_cfg: Dict[str, Any]):
    """
    Build a sampler from YAML:
      data:
        sticky_mix: 200
        tasks:
          - { name: alpaca, type: hf, dataset: tatsu-lab/alpaca, weight: 0.25, max_items: 4000 }
          - { name: code, type: hf, dataset: bigcode/some-corpus, weight: 0.25, max_items: 4000, text_key: code }
          - { name: copy, type: synth, weight: 0.20, params: { T: 128, vocab: 200 } }
          - { name: nback, type: synth, weight: 0.30, params: { T: 128, n: 5, vocab: 100 } }
    """
    tasks = data_cfg.get("tasks", []) or []
    sticky = int(data_cfg.get("sticky_mix", getattr(CFG,"sticky_mix", 0)) or 0)

    if not tasks:
        # fallback: the legacy 4-slot builder
        return build_mixer(tok,
                           tuple(getattr(CFG,"mixture",(0.4,0.2,0.2,0.2))),
                           hf_dataset=getattr(CFG,"hf_dataset","tatsu-lab/alpaca"),
                           hf_max_items=int(getattr(CFG,"hf_max_items", 5000)))

    mx = int(getattr(tok, "model_max_length", 256) or 256)
    pad_id = int(getattr(tok, "pad_token_id", 0) or 0)

    gens, weights, names = [], [], []
    for t in tasks:
        ttype  = (t.get("type","synth") or "synth").lower()
        name   = t.get("name", ttype)
        weight = float(t.get("weight", 1.0))
        params = t.get("params", {}) or {}
        if ttype == "synth":
            kind = (t.get("kind") or name).lower()
            if kind in ("copy",):
                T = int(params.get("T", min(mx, 128))); vocab = int(params.get("vocab", 100))
                def gen_copy(b, _T=T, _V=vocab): return make_copy_task(b, T=_T, vocab=_V)
                gens.append(gen_copy); weights.append(weight); names.append(name)
            elif kind in ("repeat","repeat_copy"):
                T = int(params.get("T", min(mx, 128))); vocab = int(params.get("vocab", 100))
                def gen_rep(b, _T=T, _V=vocab, _pad=pad_id): return make_repeat_copy(b, T=_T, vocab=_V, pad_id=_pad, device="cpu")
                gens.append(gen_rep); weights.append(weight); names.append(name)
            elif kind in ("nback","n_back","n-back"):
                T = int(params.get("T", min(mx, 128))); n = int(params.get("n", 5)); vocab = int(params.get("vocab", 50))
                def gen_nb(b, _T=T, _n=n, _V=vocab): return make_n_back(b, T=_T, n=_n, vocab=_V)
                gens.append(gen_nb); weights.append(weight); names.append(name)
            else:
                print(f"[registry] unknown synth kind '{kind}' → skipping '{name}'")
        elif ttype == "hf":
            dataset   = t.get("dataset", None)
            max_items = int(t.get("max_items", data_cfg.get("hf_max_items", getattr(CFG,"hf_max_items", 5000))))
            text_key  = t.get("text_key", None)
            gen = _build_hf_gen(tok, dataset, mx, pad_id, max_items, text_key=text_key)
            if gen is not None:
                gens.append(gen); weights.append(weight); names.append(name)
        else:
            print(f"[registry] unknown task type '{ttype}' → skipping '{name}'")

    if not gens:
        print("[registry] no usable tasks; falling back to legacy 4-slot mixture.")
        return build_mixer(tok,
                           tuple(getattr(CFG,"mixture",(0.4,0.2,0.2,0.2))),
                           hf_dataset=getattr(CFG,"hf_dataset","tatsu-lab/alpaca"),
                           hf_max_items=int(getattr(CFG,"hf_max_items", 5000)))

    weights = _normalize_weights(weights)
    if sticky > 1:
        base = MixtureSampler(gens, weights, names=names)
        return StickyMixtureSampler(base, chunk_steps=sticky, reset_on_set_weights=False)
    return MixtureSampler(gens, weights, names=names)
