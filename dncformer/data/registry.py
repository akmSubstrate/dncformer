from __future__ import annotations
from typing import Any, Dict, List
import random, torch

from .mix import MixtureSampler, StickyMixtureSampler
from .synthetic import make_copy_task, make_repeat_copy, make_n_back
from .hf import hf_instruction_loader, make_hf_batch

from .tasks_code import make_mbpp_gen, make_apps_gen
from .tasks_math import make_gsm8k_gen, make_aqua_rat_gen
from .tasks_logic import make_clutrr_gen # depreciated: make_logiqa_gen, make_wikihop_gen
from .tasks_algo import make_dyck_gen, make_stack_ops_gen, make_sort_list_gen

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
    YAML-driven sampler - example
      data:
        sticky_mix: 200
        tasks:
          - { name: mbpp,   type: hf,  dataset: mbpp,                 weight: 0.25, max_items: 4000 }
          - { name: apps,   type: hf,  dataset: codeparrot/apps,      weight: 0.25, max_items: 2000 }
          - { name: gsm8k,  type: hf,  dataset: gsm8k,                weight: 0.25 }
          - { name: aqua,   type: hf,  dataset: aqua_rat,             weight: 0.25 }
          - { name: babi3,  type: hf,  dataset: babi,                 weight: 0.25, params: { task_id: 3 } }
          - { name: clutrr, type: hf,  dataset: clutrr,               weight: 0.25 }
          - { name: dyck,   type: synth, kind: dyck,                  weight: 0.20, params: { depth: 4, T: 80 } }
          - { name: stack,  type: synth, kind: stack_ops,             weight: 0.20, params: { ops: 24 } }
          - { name: sort,   type: synth, kind: sort_list,             weight: 0.20, params: { length: 12, vocab: 100 } }
          - { name: copy,   type: synth, kind: copy,                  weight: 0.10, params: { T: 128, vocab: 200 } }
          - { name: nback,  type: synth, kind: nback,                 weight: 0.10, params: { T: 128, n: 5, vocab: 100 } }
    """
    tasks = data_cfg.get("tasks", []) or []
    sticky = int(data_cfg.get("sticky_mix", getattr(CFG, "sticky_mix", 0)) or 0)

    if not tasks:
        raise ValueError("CFG.data.tasks is missing or empty. Provide tasks in your YAML config.")

    mx = int(getattr(tok, "model_max_length", 256) or 256)
    pad_id = int(getattr(tok, "pad_token_id", 0) or 0)

    gens, weights, names = [], [], []
    for t in tasks:
        ttype  = (t.get("type","synth") or "synth").lower()
        name   = t.get("name", ttype)
        weight = float(t.get("weight", 1.0))
        params = t.get("params", {}) or {}
        max_items = int(t.get("max_items", data_cfg.get("hf_max_items", getattr(CFG, "hf_max_items", 5000))))

        if ttype == "logic":
            kind = (t.get("kind") or name).lower()
            raise ValueError(
                f"[registry] 'type: logic' is reserved; please use 'type: hf' and dataset 'clutrr' for CLUTRR.\n"
                f"Offending task: '{name}' (kind='{kind}')."
            )

        # ALGORITHMIC TASKS
        elif ttype == "synth":
            kind = (t.get("kind") or name).lower()
            # COPY
            if kind in ("copy",):
                T = int(params.get("T", min(mx, 128))); vocab = int(params.get("vocab", 100))
                def gen_copy(b, _T=T, _V=vocab): return make_copy_task(b, T=_T, vocab=_V)  # from synthetic
                gens.append(gen_copy); weights.append(weight); names.append(name)
            # REPEAT
            elif kind in ("repeat","repeat_copy"):
                T = int(params.get("T", min(mx, 128))); vocab = int(params.get("vocab", 100))
                def gen_rep(b, _T=T, _V=vocab):
                    from .synthetic import make_repeat_copy
                    return make_repeat_copy(b, T=_T, vocab=_V, pad_id=pad_id, device="cpu")
                gens.append(gen_rep); weights.append(weight); names.append(name)
            # N-BACK
            elif kind in ("nback","n_back","n-back"):
                T = int(params.get("T", min(mx, 128))); n = int(params.get("n", 5)); vocab = int(params.get("vocab", 50))
                def gen_nb(b, _T=T, _n=n, _V=vocab): return make_n_back(b, T=_T, n=_n, vocab=_V)  # from synthetic
                gens.append(gen_nb); weights.append(weight); names.append(name)
            # DYCK
            elif kind in ("dyck",):
                depth = int(params.get("depth", 4)); T = int(params.get("T", 80))
                gens.append( make_dyck_gen(tok, max_len=mx, pad_id=pad_id, depth=depth, T=T) ); weights.append(weight); names.append(name)  # tasks_algo
            # STACK
            elif kind in ("stack_ops","stack","stackops"):
                ops = int(params.get("ops", 24))
                gens.append( make_stack_ops_gen(tok, max_len=mx, pad_id=pad_id, ops=ops) ); weights.append(weight); names.append(name)  # tasks_algo
            # SORT
            elif kind in ("sort_list","sort"):
                length = int(params.get("length", 12)); vocab = int(params.get("vocab", 100))
                gens.append( make_sort_list_gen(tok, max_len=mx, pad_id=pad_id, length=length, vocab=vocab) ); weights.append(weight); names.append(name)  # tasks_algo
            else:
                raise ValueError(f"[registry] unknown synth kind '{kind}' in task '{name}'")

        elif ttype == "hf":
            dataset = (t.get("dataset") or "").lower()
            # CODE
            if dataset in ("mbpp","google-research-datasets/mbpp"):
                gen = make_mbpp_gen(tok, max_len=mx, pad_id=pad_id, max_items=max_items)
                gens.append(gen); weights.append(weight); names.append(name)
            elif "codeparrot/apps" in dataset or dataset.endswith("/apps") or dataset == "apps":
                gen = make_apps_gen(tok, max_len=mx, pad_id=pad_id, max_items=max_items)
                gens.append(gen); weights.append(weight); names.append(name)
            # MATH
            elif dataset == "gsm8k":
                gen = make_gsm8k_gen(tok, max_len=mx, pad_id=pad_id, split=t.get("split","train"), max_items=max_items)
                gens.append(gen); weights.append(weight); names.append(name)
            elif dataset in ("aqua_rat","aqua"):
                gen = make_aqua_rat_gen(tok, max_len=mx, pad_id=pad_id, split=t.get("split","train"), max_items=max_items)
                gens.append(gen); weights.append(weight); names.append(name)
            # LOGIC (CLUTRR only)
            elif dataset in ("clutrr","facebook/clutrr"):
                gen = make_clutrr_gen(tok, max_len=mx, pad_id=pad_id, split=t.get("split","train"), max_items=max_items)
                gens.append(gen); weights.append(weight); names.append(name)
            else:
                # Generic text fallback for explicitly HF tasks, mostly for future expansion/quick testing
                from .hf import build_generic_text_batcher
                gen = build_generic_text_batcher(tok, dataset, max_len=mx, pad_id=pad_id, max_items=max_items)
                if gen:
                    gens.append(gen); weights.append(weight); names.append(name)
                else:
                    raise ValueError(f"[registry] HF dataset '{dataset}' unsupported or returned 0 samples (task '{name}').")

        else:
            raise ValueError(f"[registry] unknown task type '{ttype}' in task '{name}'")

    if not gens:
        raise RuntimeError("[registry] no usable tasks from YAML. Aborting.")

    # Weight policy: 'auto' (default), 'uniform', or 'yaml'
    policy = str(data_cfg.get("weight_policy", "auto")).lower().strip()
    # detect whether user explicitly supplied any weight in YAML
    had_yaml_weights = any(("weight" in (t or {})) for t in tasks)

    if policy == "uniform" or (policy == "auto" and not had_yaml_weights):
        # Force equal weighting across all constructed generators
        weights = [1.0] * len(gens)
    # else: keep 'weights' list collected from YAML
    #       assigned earlier for tasks that omitted 'weight') and normalize below
    weights = _normalize_weights(weights)

    # --- Wrap with sticky chunking if requested ---
    weights = _normalize_weights(weights)
    base = MixtureSampler(gens, weights, names=names)
    sticky = int(data_cfg.get("sticky_mix", getattr(CFG, "sticky_mix", 0)) or 0)
    return StickyMixtureSampler(base, chunk_steps=sticky, reset_on_set_weights=False) if sticky > 1 else base
