# dncformer/train/scheduler.py
from __future__ import annotations
import math
from torch.optim.lr_scheduler import LambdaLR

def make_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.10):
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))
    def lr_lambda(step_idx: int):
        s = step_idx + 1
        if s <= warmup_steps:
            return s / float(warmup_steps)
        progress = (s - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)
