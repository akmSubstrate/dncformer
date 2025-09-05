from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import os, json, math, time, contextlib
import torch, torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..config import CFG
from ..log import tb as tblog
from ..model.head import DNCFormerHead
from ..utils.env import choose_amp_dtype
from ..utils.helpers import reduce_gate_tensor, gate_metrics
from dncformer.data.registry import build_sampler_from_cfg


def _ddp_env():
    """Return (use_ddp, world_size, rank, local_rank)."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    rk = int(os.environ.get("RANK", "0"))
    lr = int(os.environ.get("LOCAL_RANK", "0"))
    return (ws > 1), ws, rk, lr

def _is_main_process() -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return True
    return dist.get_rank() == 0

def _dist_allreduce_mean(x: float, device: torch.device) -> float:
    """Average a scalar across ranks; returns x if not initialized."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    t = torch.tensor([float(x)], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= float(dist.get_world_size())
    return float(t.item())

def lm_shift_labels(inp_ids: torch.Tensor, logits: torch.Tensor, tok) -> torch.Tensor:
    """
    Standard LM next-token loss with one-token shift:
      inputs:  x[ :, :-1]
      labels:  x[ :,  1:]
    """
    labels = inp_ids[:, 1:].contiguous()
    logits = logits[:, :-1].contiguous()
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=getattr(tok, "pad_token_id", 0)
    )

def _maybe_apply_lora(base: nn.Module):
    if not bool(getattr(CFG, "lora_enable", False)):
        return base
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        lcfg = LoraConfig(
            r=int(getattr(CFG, "lora_r", 8)),
            lora_alpha=int(getattr(CFG, "lora_alpha", 16)),
            lora_dropout=float(getattr(CFG, "lora_dropout", 0.05)),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=list(getattr(CFG, "lora_target_modules")),
        )
        base = get_peft_model(base, lcfg)
        print("[LoRA] enabled with target modules:", list(getattr(CFG, "lora_target_modules")))
    except Exception as e:
        print(f"[LoRA] skipped ({type(e).__name__}: {e})")
    return base

def build_model_and_tokenizer(device: torch.device):
    """Load tokenizer+base on the given device, then wrap into DNCFormerHead (still nn.Module)."""
    model_id = getattr(CFG, "base_model_id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token or "<|pad|>"
        tok.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)

    torch_dtype = torch.bfloat16 if str(getattr(CFG, "precision", "bf16")).lower() == "bf16" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=None
    ).to(device)
    base = _maybe_apply_lora(base)  # no-op if disabled

    head = DNCFormerHead(base, CFG).to(device)
    return tok, head


def _make_optimizer(model: nn.Module):
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(getattr(CFG, "lr", 2e-4))
    wd = float(getattr(CFG, "weight_decay", 0.01))
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

class WarmupCosine:
    """Simple warmup + cosine decay to min_ratio*lr (no restarts)."""
    def __init__(self, opt, warmup_steps: int, total_steps: int, min_lr_ratio: float):
        self.opt = opt
        self.warm = max(1, int(warmup_steps))
        self.total = max(self.warm+1, int(total_steps))
        self.min_ratio = float(min_lr_ratio)
        self.base = [g["lr"] for g in opt.param_groups]
        self.step_idx = 0
    def get_last_lr(self):
        return [g["lr"] for g in self.opt.param_groups]
    def step(self):
        self.step_idx += 1
        t = self.step_idx
        for i, g in enumerate(self.opt.param_groups):
            lr0 = self.base[i]
            if t <= self.warm:
                lr = lr0 * t / self.warm
            else:
                progress = (t - self.warm) / max(1, self.total - self.warm)
                cosv = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = lr0 * (self.min_ratio + (1.0 - self.min_ratio) * cosv)
            g["lr"] = lr

# NEW
def _infer_distributed_context():
    """Return (is_distributed, local_rank, world_size, is_main)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    is_distributed = (world_size > 1) and torch.cuda.is_available() and dist.is_available()
    is_main = True
    if is_distributed and dist.is_initialized():
        try:
            is_main = (dist.get_rank() == 0)
        except Exception:
            is_main = (local_rank == 0)
    else:
        is_main = True
    return is_distributed, local_rank, world_size, is_main


def _maybe_init_distributed(local_rank: int):
    """Initialize NCCL process group + set device if we’re in a DDP launch."""
    if not (torch.cuda.is_available() and dist.is_available()):
        return False
    if not dist.is_initialized():
        # torchrun/accelerate will set LOCAL_RANK; if absent, fall back to 0
        torch.cuda.set_device(local_rank if local_rank >= 0 else 0)
        dist.init_process_group(backend="nccl", timeout=torch.timedelta(seconds=1800))
        return True
    return False

def train_runner(
    *,
    steps: int,
    batch_size: int,
    mixture: Tuple[float, float, float, float],   # for CLI compatibility, ignored when CFG.data present
    warmup_steps: int = 100,
    min_lr_ratio: float = 0.1,
    hf_dataset: Optional[str] = "tatsu-lab/alpaca",   # unused when CFG.data present
    hf_max_items: int = 5000,                         # unused when CFG.data present
    chunk_len: int = 0,                               # prefer CFG.data.sticky_mix; left for CLI compat
    log_every: int = 10,
    label: Optional[str] = None
):
    # DDP bootstrap
    use_ddp, world_size, rank, local_rank = _ddp_env()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if use_ddp:
        # NCCL for CUDA, reasonable timeout for long runs
        from datetime import timedelta
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))

    # Model & optimizer
    tok, head = build_model_and_tokenizer(device)
    if use_ddp:
        # Wrap model head (contains base model)
        head = DDP(head, device_ids=[local_rank] if device.type == "cuda" else None,
                   output_device=local_rank if device.type == "cuda" else None,
                   find_unused_parameters=False)

    opt = _make_optimizer(head.module if isinstance(head, DDP) else head)
    sch = WarmupCosine(opt, warmup_steps=warmup_steps, total_steps=steps, min_lr_ratio=min_lr_ratio)

    # Data
    # 0.3.x runner requires a YAML `data:` spec
    data_cfg = getattr(CFG, "data", None)
    if not (isinstance(data_cfg, dict) and data_cfg.get("tasks")):
        if use_ddp:
            dist.barrier()
            if dist.get_rank() == 0:
                raise ValueError("CFG.data.tasks is missing or empty. 0.3.x runner requires a 'data:' section.")
            # ensure non-rank0 also exits
            raise SystemExit(1)
        raise ValueError("CFG.data.tasks is missing or empty. 0.3.x runner requires a 'data:' section.")

    # Build data mixture, handled by registry.py
    mixer = build_sampler_from_cfg(tok, data_cfg)

    # AMP / Scaler
    amp_dtype = choose_amp_dtype(getattr(CFG, "precision", "bf16"))
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    use_autocast = (device_type == 'cuda' and amp_dtype != torch.float32)
    use_scaler = (device_type == 'cuda' and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # TB logging (rank-0 only)
    run_name = label or time.strftime("dncformer-%Y%m%d-%H%M%S")
    if _is_main_process():
        tblog.start_tb_run(run_name)
        if tblog.tb and tblog.tb.writer:
            tblog.tb.add_text("run/config", json.dumps({
                "steps": steps, "batch_size": batch_size, "warmup_steps": warmup_steps,
                "data_tasks": [t.get("name","?") for t in data_cfg.get("tasks", [])],
                "sticky_mix": int(data_cfg.get("sticky_mix", 0) or 0),
            }, indent=2), 0)

    # Training loop
    module = head.module if isinstance(head, DDP) else head
    module.train()

    for step in range(1, steps + 1):
        ids = mixer(batch_size).to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device_type,
            dtype=amp_dtype if device_type == 'cuda' else torch.float32,
            enabled=use_autocast
        ):
            logits, gates, aux = module.forward(
                ids,
                gate_override=getattr(CFG, "force_g", None),
            )
            loss = lm_shift_labels(ids, logits, tok)

            # Soft expert balance - parallel experts configuration only (optional)
            lam = float(getattr(CFG, "router_balance_lambda", 0.0))
            if lam > 0.0 and isinstance(aux, dict) and "per_block" in aux:
                ents = [m.get("experts_pi_entropy") for m in aux["per_block"] if isinstance(m, dict)]
                ents = [float(x) for x in ents if isinstance(x, (float, int))]
                if ents:
                    loss = loss - lam * (sum(ents) / len(ents))

            # write-sparsity reg (optional)
            lam_w = float(getattr(CFG, "write_reg_lambda", 0.0))
            if lam_w > 0.0 and isinstance(aux, dict) and "blocks" in aux:
                bn = getattr(mixer, "last_name", "")
                apply_reg = (bn in ("copy", "repeat", "nback")) if bool(getattr(CFG, "reg_only_on_memory_batches", True)) else True
                if apply_reg:
                    wm = [m.get("write_gate_mean") for m in aux["blocks"] if isinstance(m, dict)]
                    wm = [float(x) for x in wm if isinstance(x, (float, int))]
                    if wm:
                        loss = loss + lam_w * (sum(wm) / len(wm))

        if use_scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), float(getattr(CFG, "grad_clip", 1.0)))
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), float(getattr(CFG, "grad_clip", 1.0)))
            opt.step(); opt.zero_grad(set_to_none=True)

        sch.step()

        # Reductions for logging
        # Loss avg across ranks for display/TB
        loss_avg = _dist_allreduce_mean(float(loss.item()), device)

        # Gate means for quick console summary
        gmeans_local = [float(reduce_gate_tensor(g).mean().item()) for g in gates] if isinstance(gates, (list, tuple)) else []
        gmeans_avg = []
        for gm in gmeans_local:
            gmeans_avg.append(_dist_allreduce_mean(gm, device))

        # Rank‑0 logging
        if (step % log_every == 0) and _is_main_process() and tblog.tb and tblog.tb.writer:
            tblog.tb.writer.add_scalar("train/loss", loss_avg, step)
            tblog.tb.writer.add_scalar("train/lr", float(sch.get_last_lr()[0]), step)

            # Gate metrics by block
            if isinstance(gates, (list, tuple)):
                for bi, g in enumerate(gates):
                    m, f, e = gate_metrics(g)               # <-- fixed (no enumerate(...))
                    # average each across ranks
                    m = _dist_allreduce_mean(float(m), device)
                    f = _dist_allreduce_mean(float(f), device)
                    e = _dist_allreduce_mean(float(e), device)
                    tblog.tb.writer.add_scalar(f"gates/block_{bi}_mean", m, step)
                    tblog.tb.writer.add_scalar(f"gates/block_{bi}_frac>0.5", f, step)
                    tblog.tb.writer.add_scalar(f"gates/block_{bi}_entropy", e, step)

            mix_name = getattr(mixer, "last_name", None) or "unknown"
            tblog.tb.writer.add_scalar(f"loss_by_task/{mix_name}", loss_avg, step)

            if isinstance(gates, (list, tuple)):
                for bi, g in enumerate(gates):
                    m, f, _ = gate_metrics(g)               # <-- fixed (no enumerate(...))
                    m = _dist_allreduce_mean(float(m), device)
                    f = _dist_allreduce_mean(float(f), device)
                    tblog.tb.writer.add_scalar(f"gates_by_task/block_{bi}_mean/{mix_name}", m, step)
                    tblog.tb.writer.add_scalar(f"gates_by_task/block_{bi}_frac>0.5/{mix_name}", f, step)

                # expert diagnostics - if parallel present
                for bi, bd in enumerate(aux.get("blocks", [])):
                    if isinstance(bd, dict) and "experts_pi_mean" in bd:
                        for j, v in enumerate(bd["experts_pi_mean"]):
                            tblog.tb.writer.add_scalar(f"experts/block_{bi}/pi_mean_{j}", float(v), step)
                    if isinstance(bd, dict) and "experts_pi_entropy" in bd:
                        tblog.tb.writer.add_scalar(f"experts/block_{bi}/pi_entropy", float(bd["experts_pi_entropy"]), step)
                    if isinstance(bd, dict) and "write_gate_mean" in bd:
                        tblog.tb.writer.add_scalar(f"reg/block_{bi}/write_gate_mean", float(bd["write_gate_mean"]), step)

            tblog.tb.flush()

        # Console on rank‑0
        if (step % log_every == 0) and _is_main_process():
            print(f"step {step} | loss {loss_avg:.4f} | lr {sch.get_last_lr()[0]:.2e} | gates={gmeans_avg} | mix={getattr(mixer,'last_name','?')}")

    if _is_main_process() and tblog.tb and tblog.tb.writer:
        tblog.tb.flush()

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

    # Return DDP-wrapped module and tokenizer.
    return head, tok
