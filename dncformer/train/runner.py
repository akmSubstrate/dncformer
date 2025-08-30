# dncformer/train/runner.py
from __future__ import annotations
import contextlib, math, torch
from typing import Optional, Dict, Any
from dncformer.config import CFG
from dncformer.train.loop import build_model_and_tokenizer, lm_shift_labels  # reuse
from dncformer.log.tb import TB_AVAILABLE, start_tb_run, tb
from dncformer.utils.env import choose_amp_dtype
from dncformer.train.optim import make_optimizer
from dncformer.train.scheduler import make_continuous_scheduler
from dncformer.data.registry import build_pipeline, DatasetSpec, MixerSpec

def train_model(exp: Dict[str, Any]):
    """
    exp = {
      "train": {"steps": 3000, "batch_size": 8, ...},
      "datasets": [DatasetSpec(...), ...],
      "mixer": MixerSpec(...),
      "lora": {...},
      "anti_collapse": {...},
      ...
    }
    """
    tok, head = build_model_and_tokenizer()
    ds_specs = [DatasetSpec(**d) for d in exp["datasets"]]
    mx_spec = MixerSpec(**exp.get("mixer", {}))
    sampler, _names = build_pipeline(ds_specs, mx_spec)

    steps = int(exp["train"].get("steps", CFG.train_steps))
    bs    = int(exp["train"].get("batch_size", CFG.batch_size))
    warm  = int(exp["train"].get("warmup_steps", max(10, steps//20)))
    logk  = int(exp["train"].get("log_every", CFG.log_every))

    optim = make_optimizer(head)
    sched = make_continuous_scheduler(
        optim, warmup_steps=warm,
        base_lr=getattr(CFG,"lr",2e-4),
        min_lr_ratio=float(getattr(CFG,"lr_min_ratio", 0.1)),
        cawr_T0=int(getattr(CFG,"lr_cawr_T0",200)),
        cawr_Tmult=int(getattr(CFG,"lr_cawr_Tmult",2)),
    )

    amp_dtype = choose_amp_dtype(CFG.precision)
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype in (torch.float16, torch.bfloat16)))
    head.train()

    for step in range(1, steps+1):
        x = sampler(bs).to(next(head.parameters()).device)
        with torch.autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype!=torch.float32)):
            logits, gates, aux = head(x, gate_override=getattr(CFG, "force_g", None))
            loss = lm_shift_labels(x, logits, tok)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), CFG.grad_clip)
        scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
        sched.step()

        if (step % logk == 0) and tb and tb.writer:
            mix_name = getattr(sampler, "last_name", "unknown")
            tb.writer.add_scalar("train/loss", float(loss.item()), step)
            tb.writer.add_scalar("train/lr", float(sched.get_last_lr()[0]), step)
            tb.writer.add_scalar(f"loss_by_task/{mix_name}", float(loss.item()), step)
            tb.flush()
