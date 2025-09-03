from __future__ import annotations
from typing import Callable, Optional
import random, torch
from datasets import load_dataset

def _pad_or_trim(ids: torch.Tensor, T: int, pad_id: int) -> torch.Tensor:
    n = ids.numel()
    if n == T: return ids
    if n > T:  return ids[:T]
    pad = torch.full((T - n,), pad_id, dtype=torch.long)
    return torch.cat([ids, pad], dim=0)

def _batchize(tok, texts, B: int, max_len: int, pad_id: int) -> torch.Tensor:
    outs = []
    for _ in range(B):
        s = random.choice(texts)
        ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        outs.append(_pad_or_trim(ids, max_len, pad_id).unsqueeze(0))
    return torch.cat(outs, dim=0)

def make_mbpp_gen(tok, max_len: int, pad_id: int, max_items: int = 4000) -> Callable[[int], torch.Tensor]:
    """
    MBPP: problem + (optionally) signature + examples → '### Solution:' prompt.
    We accept either 'mbpp' or 'google-research-datasets/mbpp' layouts.
    """
    try:
        # Try the sanitized split first; fallback to raw
        try:
            ds = load_dataset("mbpp", "sanitized", split="train", streaming=True)
        except Exception:
            ds = load_dataset("google-research-datasets/mbpp", split="train", streaming=True)
    except Exception as e:
        raise RuntimeError(f"[mbpp] load failed: {e}")

    buf = []
    for ex in ds:
        # robust field search across variants
        prob = ex.get("text") or ex.get("prompt") or ex.get("task_description") or ex.get("problem") or ""
        sig  = ex.get("code", "")  # not ideal—sometimes holds solution; we won’t leak code
        # Some versions include 'code' as solution; avoid leaking by not including it when it seems long.
        sig_str = ""
        if isinstance(sig, str) and sig.strip() and sig.count("\n") <= 1 and len(sig) < 80 and "def " in sig:
            sig_str = f"\n### Function signature:\n{sig.strip()}\n"
        if not prob:
            continue
        prompt = f"### Problem:\n{prob.strip()}{sig_str}\n### Solution:\n"
        buf.append(prompt)
        if len(buf) >= int(max_items): break

    if not buf:
        raise RuntimeError("[mbpp] 0 samples found")

    def gen_mbpp(B: int) -> torch.Tensor:
        return _batchize(tok, buf, B, max_len, pad_id)
    return gen_mbpp

def make_apps_gen(tok, max_len: int, pad_id: int, max_items: int = 4000) -> Callable[[int], torch.Tensor]:
    """
    APPS: competitive programming problems.
    Fields vary; try 'question' plus optional IO examples.
    """
    try:
        ds = load_dataset("codeparrot/apps", split="train", streaming=True, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"[apps] load failed: {e}")

    buf = []
    for ex in ds:
        q = ex.get("question") or ex.get("problem") or ""
        in_out = ex.get("input_output", "")
        exm = f"\n### Examples:\n{in_out}" if isinstance(in_out, str) and in_out.strip() else ""
        if not q:
            continue
        prompt = f"### Problem:\n{q.strip()}{exm}\n### Solution (code):\n"
        buf.append(prompt)
        if len(buf) >= int(max_items): break
    if not buf:
        raise RuntimeError("[apps] 0 samples found")

    def gen_apps(B: int) -> torch.Tensor:
        return _batchize(tok, buf, B, max_len, pad_id)
    return gen_apps
