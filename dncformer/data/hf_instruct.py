# dncformer/data/hf_instruct.py
from __future__ import annotations
from datasets import load_dataset
import torch, random
from .registry import TaskRegistry

@TaskRegistry.register("hf_instruct")
def build_hf_instruct(dataset="tatsu-lab/alpaca", split="train",
                      instr_key="instruction", out_key="output",
                      max_items=5000, pad_id=0, max_len=256, tokenizer=None):
    ds = load_dataset(dataset, split=split)
    pairs = []
    for ex in ds:
        i, o = ex.get(instr_key, ""), ex.get(out_key, "")
        if i and o: pairs.append((i, o))
        if len(pairs) >= max_items: break
    random.shuffle(pairs)

    def encode_one(instr, out):
        prompt = f"### Instruction:\n{instr}\n\n### Response:\n{out}"
        return tokenizer(prompt, return_tensors="pt", truncation=True,
                         max_length=max_len).input_ids[0]

    def gen(b):
        out=[]
        for _ in range(b):
            i, o = random.choice(pairs)
            ids = encode_one(i, o)
            if ids.numel() < max_len:
                pad = torch.full((max_len - ids.numel(),), pad_id, dtype=torch.long)
                ids = torch.cat([ids, pad], dim=0)
            else:
                ids = ids[:max_len]
            out.append(ids.unsqueeze(0))
        return torch.cat(out, dim=0)
    return gen
