from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import json, math, time, contextlib
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from .loop import lm_shift_labels
from ..config import CFG
from ..log import tb as tblog
from ..model.head import DNCFormerHead
from ..utils.env import choose_amp_dtype, report_cuda, sdpa_ctx
from ..utils.helpers import reduce_gate_tensor, gate_metrics
from ..data.mix import build_mixer
from dncformer.data.registry import build_sampler_from_cfg

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

def build_model_and_tokenizer():
    model_id = getattr(CFG, "base_model_id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token or "<|pad|>"
        tok.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)

    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, "bfloat16" if CFG.precision=="bf16" else "float32"),
        low_cpu_mem_usage=True, device_map=None
    )
    base = _maybe_apply_lora(base)  # no-op if disabled
    head = DNCFormerHead(base, CFG).to(getattr(CFG, "device", "cuda" if torch.cuda.is_available() else "cpu"))
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

def train_runner(
    *,
    steps: int,
    batch_size: int,
    mixture: Tuple[float, float, float, float],
    warmup_steps: int = 100,
    min_lr_ratio: float = 0.1,
    hf_dataset: Optional[str] = "tatsu-lab/alpaca",
    hf_max_items: int = 5000,
    chunk_len: int = 0,
    log_every: int = 10,
    label: Optional[str] = None
):
    # set up
    tok, head = build_model_and_tokenizer()
    opt = _make_optimizer(head)
    sch = WarmupCosine(opt, warmup_steps=warmup_steps, total_steps=steps, min_lr_ratio=min_lr_ratio)

    try:
        data_cfg = getattr(CFG, "data", None)
    except Exception:
        data_cfg = None

    if isinstance(data_cfg, dict) and data_cfg.get("tasks"):
        mixer = build_sampler_from_cfg(tok, data_cfg)
    else:
        # legacy 4-slot fallback
        mixer = build_mixer(tok, mixture, hf_dataset=hf_dataset, hf_max_items=hf_max_items)

    amp_dtype = choose_amp_dtype(getattr(CFG, "precision", "bf16"))
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_autocast = (device_type == 'cuda' and amp_dtype != torch.float32)
    use_scaler = (device_type == 'cuda' and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # TB run
    run_name = label or time.strftime("dncformer-%Y%m%d-%H%M%S")
    tblog.start_tb_run(run_name)
    if tblog.tb and tblog.tb.writer:
        tblog.tb.add_text("run/config", json.dumps({
            "steps": steps, "batch_size": batch_size, "warmup_steps": warmup_steps,
            "mixture": list(mixture), "chunk_len": int(chunk_len),
            "hf_dataset": hf_dataset, "hf_max_items": hf_max_items,
        }, indent=2), 0)

    head.train()
    for step in range(1, steps+1):
        ids = mixer(batch_size).to(next(head.parameters()).device)
        with torch.amp.autocast(
                device_type=device_type,
                dtype=amp_dtype if device_type == 'cuda' else torch.float32,
                enabled=use_autocast
        ):
            logits, gates, aux = head.forward(
                ids,
                gate_override=getattr(CFG, "force_g", None),
            )
            # standard LM loss
            loss = lm_shift_labels(ids, logits, tok)

            # A1: soft expert balance (parallel only)
            lam = float(getattr(CFG, "router_balance_lambda", 0.0))
            if lam > 0.0 and isinstance(aux, dict) and "per_block" in aux:
                ents = [m.get("experts_pi_entropy") for m in aux["per_block"] if isinstance(m, dict)]
                ents = [float(x) for x in ents if isinstance(x, (float,int))]
                if ents: loss = loss - lam * (sum(ents)/len(ents))

            # write-sparsity reg (optional)
            lam_w = float(getattr(CFG, "write_reg_lambda", 0.0))
            if lam_w > 0.0 and isinstance(aux, dict) and "blocks" in aux:
                bn = getattr(mixer, "last_name", "")
                apply_reg = (bn in ("copy","repeat","nback")) if bool(getattr(CFG, "reg_only_on_memory_batches", True)) else True
                if apply_reg:
                    wm = [m.get("write_gate_mean") for m in aux["blocks"] if isinstance(m, dict)]
                    wm = [float(x) for x in wm if isinstance(x, (float,int))]
                    if wm: loss = loss + lam_w * (sum(wm)/len(wm))

        if use_scaler:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(getattr(CFG, "grad_clip", 1.0)))
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(getattr(CFG, "grad_clip", 1.0)))
            opt.step(); opt.zero_grad(set_to_none=True)

        sch.step()

        if step % log_every == 0 and tblog.tb and tblog.tb.writer:
            # General loss/lr
            tblog.tb.writer.add_scalar("train/loss", float(loss.item()), step)
            tblog.tb.writer.add_scalar("train/lr", float(sch.get_last_lr()[0]), step)
            # Gate metrics
            if isinstance(gates, (list, tuple)):
                for bi, g in enumerate(gates):
                    m, f, e = gate_metrics(g)
                    tblog.tb.writer.add_scalar(f"gates/block_{bi}_mean", m, step)
                    tblog.tb.writer.add_scalar(f"gates/block_{bi}_frac>0.5", f, step)
                    tblog.tb.writer.add_scalar(f"gates/block_{bi}_entropy", e, step)
            # Per-task breakdown
            mix_name = getattr(mixer, "last_name", None) or "unknown"
            tblog.tb.writer.add_scalar(f"loss_by_task/{mix_name}", float(loss.item()), step)
            if isinstance(gates, (list,tuple)):
                for bi, g in enumerate(gates):
                    m, f, _ = enumerate(gate_metrics(g))
                    tblog.tb.writer.add_scalar(f"gates_by_task/block_{bi}_mean/{mix_name}", m, step)
                    tblog.tb.writer.add_scalar(f"gates_by_task/block_{bi}_frac>0.5/{mix_name}", f, step)
                # expert diagnostics (if present)
                for bi, bd in enumerate(aux.get("blocks", [])):
                    if isinstance(bd, dict) and "experts_pi_mean" in bd:
                        for j, v in enumerate(bd["experts_pi_mean"]):
                            tblog.tb.writer.add_scalar(f"experts/block_{bi}/pi_mean_{j}", float(v), step)
                    if isinstance(bd, dict) and "experts_pi_entropy" in bd:
                        tblog.tb.writer.add_scalar(f"experts/block_{bi}/pi_entropy", float(bd["experts_pi_entropy"]), step)
                    if isinstance(bd, dict) and "write_gate_mean" in bd:
                        tblog.tb.writer.add_scalar(f"reg/block_{bi}/write_gate_mean", float(bd["write_gate_mean"]), step)
            tblog.tb.flush()

        if step % log_every == 0:
            gmeans = [float(reduce_gate_tensor(g).mean().item()) for g in gates] if isinstance(gates, (list, tuple)) else []
            print(f"step {step} | loss {loss.item():.4f} | lr {sch.get_last_lr()[0]:.2e} | gates={gmeans} | mix={getattr(mixer,'last_name','?')}")

    if tblog and tblog.tb.writer: tblog.tb.flush()
    return head, tok
