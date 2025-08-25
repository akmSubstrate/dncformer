# dncformer/train/optim.py
from __future__ import annotations
import torch
from ..utils.helpers import requires_grad_

def make_optimizer(model, lr: float, weight_decay: float):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
