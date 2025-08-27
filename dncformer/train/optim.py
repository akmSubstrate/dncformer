# dncformer/train/optim.py
from __future__ import annotations
import torch
from ..config import CFG

# Heuristic: what should NOT receive weight decay?
_NO_DECAY_KEYWORDS = ("bias", "norm", "layernorm", "ln_", "emb", "pos_emb", "rope", "rotary", "lora_")

def _use_decay(param_name: str, param: torch.nn.Parameter) -> bool:
    """Return True if weight decay should be applied to this parameter."""
    # 1D tensors (e.g., bias, LayerNorm weights) -> no decay
    if param.ndim == 1:
        return False
    # Name-based filters
    nm = param_name.lower()
    if any(k in nm for k in _NO_DECAY_KEYWORDS):
        return False
    return True

def make_optimizer(
    model: torch.nn.Module,
    lr: float | None = None,
    weight_decay: float | None = None,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    AdamW with two param groups:
      - decay:   all multi-dim weights not matched by the no‑decay filter
      - no_decay: biases, norms, embeddings, LoRA matrices, etc.
    lr/weight_decay default to CFG if not provided.
    """
    lr = float(lr if lr is not None else getattr(CFG, "lr", 2e-4))
    weight_decay = float(weight_decay if weight_decay is not None else getattr(CFG, "weight_decay", 0.01))

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if _use_decay(n, p) else no_decay).append(p)

    # Guard against empty groups (can happen when only adapters are trainable)
    param_groups = []
    if decay:
        param_groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        param_groups.append({"params": no_decay, "weight_decay": 0.0})

    # Fallback: if everything ended in one bucket, still construct a valid optimizer
    if not param_groups:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "weight_decay": weight_decay}]

    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
