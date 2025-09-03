from __future__ import annotations
from typing import Callable
import random, torch
from datasets import load_dataset

def _pad_or_trim(ids, T, pad_id):
    n = ids.numel()
    if n == T: return ids
    if n > T:  return ids[:T]
    return torch.cat([ids, torch.full((T-n,), pad_id, dtype=torch.long)], dim=0)

def _batchize(tok, texts, B, max_len, pad_id):
    outs=[]
    for _ in range(B):
        s = random.choice(texts)
        ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        outs.append(_pad_or_trim(ids, max_len, pad_id).unsqueeze(0))
    return torch.cat(outs, dim=0)

def make_gsm8k_gen(tok, max_len: int, pad_id: int, split="train", max_items: int = 8000) -> Callable[[int], torch.Tensor]:
    """
    GSM8K: 'question' + optional 'answer' rationale formatting (we omit answers for training).
    """
    try:
        ds = load_dataset("gsm8k", "main", split=split, streaming=True)
    except Exception as e:
        raise RuntimeError(f"[gsm8k] load failed: {e}")

    buf=[]
    for ex in ds:
        q = ex.get("question", "").strip()
        if not q: continue
        prompt = f"### Problem:\n{q}\n### Solution:\n"
        buf.append(prompt)
        if len(buf) >= int(max_items): break
    if not buf: raise RuntimeError("[gsm8k] 0 samples found")
    def gen(B: int) -> torch.Tensor:
        return _batchize(tok, buf, B, max_len, pad_id)
    return gen

def make_aqua_rat_gen(tok, max_len: int, pad_id: int, split="train", max_items: int = 8000) -> Callable[[int], torch.Tensor]:
    """
    AQuA-RAT: question + options. We format as NL multiple choice without including the provided rationale.
    """
    try:
        ds = load_dataset("aqua_rat", split=split, streaming=True)
    except Exception as e:
        raise RuntimeError(f"[aqua_rat] load failed: {e}")

    buf=[]
    for ex in ds:
        q = ex.get("question", "").strip()
        ops = ex.get("options", [])
        if not q or not ops: continue
        ops_txt = "\n".join([f"{chr(65+i)}. {o}" for i, o in enumerate(ops)])
        prompt = f"### Question:\n{q}\n### Options:\n{ops_txt}\n### Explanation and Answer:\n"
        buf.append(prompt)
        if len(buf) >= int(max_items): break
    if not buf: raise RuntimeError("[aqua_rat] 0 samples found")
    def gen(B: int) -> torch.Tensor:
        return _batchize(tok, buf, B, max_len, pad_id)
    return gen
