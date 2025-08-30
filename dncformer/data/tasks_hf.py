from __future__ import annotations
from typing import Callable, Optional
import torch, random

def build_hf_gen(tok, dataset_id: str, max_items: int, max_len: int, pad_id: int) -> Callable[[int], torch.Tensor]:
    """
    Build a function that returns (B, max_len) token batches from a HF dataset.
    Prefers 'text'/'content'/'story', falls back to scanning the first example.
    """
    from datasets import load_dataset
    # Try streaming first; fall back to full download if necessary
    try:
        ds = load_dataset(dataset_id, split="train", streaming=True)
    except Exception:
        ds = load_dataset(dataset_id, split="train")

    # identify a likely text field
    text_key = None
    if getattr(ds, "features", None):
        for k in ("text","content","story","document","body","article"):
            if k in ds.features:
                text_key = k; break
    if text_key is None:
        try:
            first = next(iter(ds))
            for k in ("text","content","story","document","body","article"):
                if k in first: text_key = k; break
        except StopIteration:
            pass
    if text_key is None:
        raise RuntimeError("No text field found in the HF dataset")

    samples = []
    for ex in ds:
        txt = ex.get(text_key, None)
        if not txt: continue
        ids = tok(txt, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        if ids.numel() < 8: continue
        samples.append(ids.cpu())
        if len(samples) >= int(max_items): break
    if not samples:
        raise RuntimeError("0 usable HF samples")

    def gen(b: int) -> torch.Tensor:
        out = []
        for _ in range(b):
            ids = random.choice(samples); n = ids.numel()
            if n >= max_len:
                s = random.randint(0, n - max_len)
                seq = ids[s:s+max_len]
            else:
                pad = torch.full((max_len - n,), pad_id, dtype=torch.long)
                seq = torch.cat([ids, pad], dim=0)
            out.append(seq.unsqueeze(0))
        return torch.cat(out, dim=0)
    return gen
