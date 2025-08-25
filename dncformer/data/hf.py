# dncformer/data/hf.py
from __future__ import annotations
import random
from typing import List, Tuple
import torch

def hf_instruction_loader(dataset_name="tatsu-lab/alpaca", split="train",
                          text_field=("instruction","output"), max_items=5000):
    try:
        from datasets import load_dataset
    except Exception:
        print("Install 'datasets' to enable HF loading: pip install datasets -q")
        return []
    ds = load_dataset(dataset_name, split=split)
    pairs = []
    i_field, o_field = text_field
    for ex in ds:
        instr = ex.get(i_field, ""); out = ex.get(o_field, "")
        if instr and out: pairs.append((instr, out))
        if len(pairs) >= max_items: break
    random.shuffle(pairs); return pairs

def format_instruction(tok, instr: str, resp: str, max_len=256) -> torch.Tensor:
    prompt = f"### Instruction:\n{instr}\n\n### Response:\n{resp}"
    return tok(prompt, return_tensors="pt", truncation=True, max_length=max_len).input_ids[0]

def make_hf_batch(tok, pairs: List[Tuple[str,str]], batch: int, max_len=256) -> torch.Tensor:
    if not pairs:
        return torch.full((batch, max_len), tok.pad_token_id, dtype=torch.long)
    batch_ids = []
    for _ in range(batch):
        instr, out = random.choice(pairs)
        ids = format_instruction(tok, instr, out, max_len=max_len)
        batch_ids.append(ids)
    maxL = min(max(x.size(0) for x in batch_ids), max_len)
    out_ids = torch.full((batch, maxL), tok.pad_token_id, dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        ids = ids[:maxL]; out_ids[i, :ids.size(0)] = ids
    return out_ids
