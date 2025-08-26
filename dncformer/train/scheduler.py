# dncformer/train/scheduler.py
from __future__ import annotations
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR, SequentialLR, CosineAnnealingWarmRestarts, LinearLR
from ..config import CFG


def make_stepped_scheduler(optim, total_steps: int):
    t = getattr(CFG, "scheduler_type", "cosine")
    if t == "one_cycle":
        return OneCycleLR(
            optim, max_lr=CFG.lr, total_steps=total_steps, pct_start=CFG.warmup_steps/total_steps
        )
    elif t == "plateau":
        return ReduceLROnPlateau(
            optim, mode="min", factor=CFG.plateau_factor, patience=CFG.plateau_patience
        )
    else:  # cosine default
        return CosineAnnealingLR(
            optim, T_max=total_steps, eta_min=CFG.lr*CFG.min_lr_ratio
        )

def make_continuous_scheduler(
    optimizer: Optimizer,
    warmup_steps: int = 100,
    base_lr: float = 2e-4,
    min_lr_ratio: float = 0.10,
    cawr_T0: int = 200,         # first restart period in *steps* (not epochs)
    cawr_Tmult: int = 2,        # restart period multiplier
) -> SequentialLR:
    """
    Linear warm-up (0 -> base) for warmup_steps, then CosineAnnealingWarmRestarts forever.
    - Does *not* require knowing total steps up front.
    - Call `scheduler.step()` once per optimizer.step().
    """
    eta_min = max(1e-12, base_lr * float(min_lr_ratio))

    # Warm-up from ~0 to 1*base lr
    warm = LinearLR(optimizer, start_factor=1e-6, total_iters=max(1, int(warmup_steps)))

    # Cosine with restarts (per step)
    cawr = CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, int(cawr_T0)),
        T_mult=max(1, int(cawr_Tmult)),
        eta_min=eta_min
    )

    # Chain: warm-up first, then CAWR
    seq = SequentialLR(optimizer, schedulers=[warm, cawr], milestones=[max(1, int(warmup_steps))])
    return seq