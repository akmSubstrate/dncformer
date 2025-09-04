from __future__ import annotations
import os
from typing import Optional, Tuple, Dict
import json, math, time, contextlib

from dataclasses import dataclass
import torch, torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM


from ..config import CFG
from ..log import tb as tblog
from ..model.head import DNCFormerHead
from ..utils.env import choose_amp_dtype, report_cuda, sdpa_ctx
from ..utils.helpers import reduce_gate_tensor, gate_metrics
from ..utils.dist import init_distributed, get_world_size, get_local_rank, is_main_process, barrier, cleanup_distributed
from ..data.mix import StickyMixtureSampler
from dncformer.data.registry import build_sampler_from_cfg


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

def build_model_and_tokenizer(device: str | torch.device | None = None):
    model_id = getattr(CFG, "base_model_id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token or "<|pad|>"
        tok.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)

    # dtype per CFG
    prec = str(getattr(CFG, "precision", "bf16")).lower()
    dtype = torch.bfloat16 if (prec == "bf16" and torch.cuda.is_available()) else torch.float32

    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None
    )
    base = _maybe_apply_lora(base)  # no-op if disabled
    dev = device or getattr(CFG, "device", ("cuda" if torch.cuda.is_available() else "cpu"))
    head = DNCFormerHead(base, CFG).to(dev)
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
    mixture: Tuple[float, float, float, float],
    warmup_steps: int = 100,
    min_lr_ratio: float = 0.1,
    hf_dataset: Optional[str] = "tatsu-lab/alpaca",
    hf_max_items: int = 5000,
    chunk_len: int = 0,
    log_every: int = 10,
    label: Optional[str] = None
):
    # setup - init DDP if torchrun provided; else choose local device
    local_rank, world_size = init_distributed(backend=str(getattr(CFG, "ddp_backend", "nccl")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Seed: offset by global rank so each rank sees different batches
    seed = int(getattr(CFG, "seed", 1337))
    torch.manual_seed(seed + (0 if world_size == 1 else local_rank))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + (0 if world_size == 1 else local_rank))

    tok, head = build_model_and_tokenizer()
    head = head.to(device)

    # Wrap with DDP if multi-GPU
    if world_size > 1:
        head = _ddp.DistributedDataParallel(
            head,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=bool(getattr(CFG, "ddp_broadcast_buffers", False)),
            find_unused_parameters=bool(getattr(CFG, "ddp_find_unused_parameters", True)),
        )

    opt = _make_optimizer(head)
    sch = WarmupCosine(opt, warmup_steps=warmup_steps, total_steps=steps, min_lr_ratio=min_lr_ratio)

    # Data sampler / YAML loading logic
    try:
        data_cfg = getattr(CFG, "data", None)
    except Exception:
        data_cfg = None

    if not (isinstance(data_cfg, dict) and data_cfg.get("tasks")):
        raise ValueError(
            "CFG.data.tasks is missing or empty. 0.3.x runner requires a YAML config with a 'data:' section.\n"
            "Example:\n"
            "  data:\n"
            "    sticky_mix: 10\n"
            "    tasks:\n"
            "      - { name: alpaca, type: hf, dataset: tatsu-lab/alpaca,    weight: 0.25, max_items: 4000 }\n"
            "      - { name: apps,   type:hf,  dataset: codeparrot/apps,     weight:0.15,  max_items: 3000}\n"
            "      - { name: copy,   type: synth, weight: 0.20, params: { T: 128, vocab: 200 } }\n"
            "      - { name: stack,  type: synth, kind: stack_ops, weight: 0.05, params: {ops: 50}},"
        )
    mixer = build_sampler_from_cfg(tok, data_cfg)

    # Optional CLI override: make the sampler sticky even if YAML didn't set sticky_mix.
    if isinstance(chunk_len, int) and chunk_len > 1 and not hasattr(mixer, "base"):
        mixer = StickyMixtureSampler(mixer, chunk_steps=int(chunk_len), reset_on_set_weights=False)

    # Scheduler echo for context/debug
    try:
        base = mixer.base if hasattr(mixer, "base") else mixer
        p = getattr(base, "p", None)
        probs = (p.detach().cpu().tolist() if hasattr(p, "detach") else None)
        print(f"[data] using tasks={getattr(base, 'names', None)} | weights={getattr(base, 'weights', None)} | "
              f"sticky={getattr(mixer, 'chunk_steps', 1) if hasattr(mixer, 'chunk_steps') else 1} | p={probs}")
    except Exception:
        pass

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
            print("[tb] logging at step", step, "task=", getattr(mixer, 'last_name', '?'))
            w = tblog.tb.writer  # local alias
            # General loss/lr
            w.add_scalar("train/loss", float(loss.item()), step)
            w.add_scalar("train/lr", float(sch.get_last_lr()[0]), step)

            # Gate metrics
            if isinstance(gates, (list, tuple)):
                for bi, g in enumerate(gates):
                    m, f, e = gate_metrics(g)
                    w.add_scalar(f"gates/block_{bi}_mean", m, step)
                    w.add_scalar(f"gates/block_{bi}_frac>0.5", f, step)
                    w.add_scalar(f"gates/block_{bi}_entropy", e, step)

            # Per-task breakdown
            mix_name = getattr(mixer, "last_name", None) or "unknown"
            tblog.tb.writer.add_scalar(f"loss_by_task/{mix_name}", float(loss.item()), step)
            if isinstance(gates, (list, tuple)):
                for bi, g in enumerate(gates):
                    m, f, _ = gate_metrics(g)  # <-- fixed
                    tblog.tb.writer.add_scalar(f"gates_by_task/block_{bi}_mean/{mix_name}", m, step)
                    tblog.tb.writer.add_scalar(f"gates_by_task/block_{bi}_frac>0.5/{mix_name}", f, step)

                # Expert diagnostics (if present)
                for bi, bd in enumerate(aux.get("blocks", [])):
                    if isinstance(bd, dict) and "experts_pi_mean" in bd:
                        for j, v in enumerate(bd["experts_pi_mean"]):
                            w.add_scalar(f"experts/block_{bi}/pi_mean_{j}", float(v), step)
                    if isinstance(bd, dict) and "experts_pi_entropy" in bd:
                        w.add_scalar(f"experts/block_{bi}/pi_entropy", float(bd["experts_pi_entropy"]), step)
                    if isinstance(bd, dict) and "write_gate_mean" in bd:
                        w.add_scalar(f"reg/block_{bi}/write_gate_mean", float(bd["write_gate_mean"]), step)

            w.flush()

        if step % log_every == 0:
            gmeans = [float(reduce_gate_tensor(g).mean().item()) for g in gates] if isinstance(gates, (list, tuple)) else []
            print(f"step {step} | loss {loss.item():.4f} | lr {sch.get_last_lr()[0]:.2e} | gates={gmeans} | mix={getattr(mixer,'last_name','?')}")

    if tblog and tblog.tb.writer: tblog.tb.flush()
    return head, tok

# NEW
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
    # --- DDP context (no-op on single GPU) ---
    is_dist, local_rank, world_size, _ = _infer_distributed_context()
    if is_dist:
        _maybe_init_distributed(local_rank)
        is_main = (dist.get_rank() == 0)
        device = torch.device(f"cuda:{local_rank}")
    else:
        is_main = True
        device = torch.device(getattr(CFG, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    # per-rank seeding (keep CLI/CFG seed meaning stable)
    try:
        seed = int(getattr(CFG, "seed", 1337))
    except Exception:
        seed = 1337
    torch.manual_seed(seed + (dist.get_rank() if is_dist else 0))
    import random as _rnd
    _rnd.seed(seed + (dist.get_rank() if is_dist else 0))

    # --- Setup ---
    tok, head = build_model_and_tokenizer(device=device)
    if is_dist:
        # DDP wrap on the rank-local device
        head = DDP(head, device_ids=[device.index], output_device=device.index,
                   broadcast_buffers=False, find_unused_parameters=False)

    opt = _make_optimizer(head)
    sch = WarmupCosine(opt, warmup_steps=warmup_steps, total_steps=steps, min_lr_ratio=min_lr_ratio)

    # data config or legacy fallback
    try:
        data_cfg = getattr(CFG, "data", None)
    except Exception:
        data_cfg = None

    if isinstance(data_cfg, dict) and data_cfg.get("tasks"):
        mixer = build_mixed_sampler(tok, data_cfg)
    else:
        # legacy 4-slot fallback
        from ..data.mix import build_mixer
        mixer = build_mixer(tok, mixture, hf_dataset=hf_dataset, hf_max_items=hf_max_items)

    # batch per rank
    per_rank_batch = max(1, int(batch_size // (world_size if is_dist else 1)))

    # AMP/sdpa setup
    amp_dtype = choose_amp_dtype(getattr(CFG, "precision", "bf16"))
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    use_autocast = (device_type == 'cuda' and amp_dtype != torch.float32)
    use_scaler = (device_type == 'cuda' and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # TB only on rank-0
    if is_main and TB_AVAILABLE:
        run_name = label or time.strftime("dncformer-%Y%m%d-%H%M%S")
        start_tb_run(run_name)
        if tb and tb.writer:
            tb.add_text("run/config", json.dumps({
                "steps": steps, "batch_size": batch_size, "per_rank_batch": per_rank_batch,
                "world_size": world_size, "warmup_steps": warmup_steps,
                "hf_dataset": hf_dataset, "hf_max_items": hf_max_items,
                "chunk_len": int(chunk_len),
            }, indent=2), 0)

    # --- Train loop ---
    head.train()
    for step in range(1, steps + 1):
        ids = mixer(per_rank_batch).to(device)

        with torch.amp.autocast(
            device_type=device_type,
            dtype=amp_dtype if device_type == 'cuda' else torch.float32,
            enabled=use_autocast
        ):
            logits, gates, aux = head.forward(
                ids,
                gate_override=getattr(CFG, "force_g", None),
            )
            loss = lm_shift_labels(ids, logits, tok)

            # Optional: scale loss so DDP gradient magnitude matches single-GPU
            if is_dist:
                loss = loss / float(world_size)

            # A1: soft expert balance (parallel only)
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
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(getattr(CFG, "grad_clip", 1.0)))
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(getattr(CFG, "grad_clip", 1.0)))
            opt.step(); opt.zero_grad(set_to_none=True)

        sch.step()

        # rank-0 logging only
        if is_main and (step % log_every == 0) and tb and tb.writer:
            tb.writer.add_scalar("train/loss", float(loss.item()), step)
            tb.writer.add_scalar("train/lr", float(sch.get_last_lr()[0]), step)

            if isinstance(gates, (list, tuple)):
                for bi, g in enumerate(gates):
                    m, f, e = gate_metrics(g)
                    tb.writer.add_scalar(f"gates/block_{bi}_mean", m, step)
                    tb.writer.add_scalar(f"gates/block_{bi}_frac>0.5", f, step)
                    tb.writer.add_scalar(f"gates/block_{bi}_entropy", e, step)

            mix_name = getattr(mixer, "last_name", None) or "unknown"
            tb.writer.add_scalar(f"loss_by_task/{mix_name}", float(loss.item()), step)
            if isinstance(gates, (list, tuple)):
                for bi, g in enumerate(gates):
                    m, f, _ = gate_metrics(g)
                    tb.writer.add_scalar(f"gates_by_task/block_{bi}_mean/{mix_name}", m, step)
                    tb.writer.add_scalar(f"gates_by_task/block_{bi}_frac>0.5/{mix_name}", f, step)

                for bi, bd in enumerate(aux.get("blocks", [])):
                    if isinstance(bd, dict) and "experts_pi_mean" in bd:
                        for j, v in enumerate(bd["experts_pi_mean"]):
                            tb.writer.add_scalar(f"experts/block_{bi}/pi_mean_{j}", float(v), step)
                    if isinstance(bd, dict) and "experts_pi_entropy" in bd:
                        tb.writer.add_scalar(f"experts/block_{bi}/pi_entropy", float(bd["experts_pi_entropy"]), step)
                    if isinstance(bd, dict) and "write_gate_mean" in bd:
                        tb.writer.add_scalar(f"reg/block_{bi}/write_gate_mean", float(bd["write_gate_mean"]), step)
            tb.flush()

        if is_main and (step % log_every == 0):
            gmeans = [float(reduce_gate_tensor(g).mean().item()) for g in gates] if isinstance(gates, (list, tuple)) else []
            print(f"step {step} | loss {loss.item():.4f} | lr {sch.get_last_lr()[0]:.2e} | gates={gmeans} | mix={getattr(mixer,'last_name','?')}")

    # finalize
    if is_main and tb and tb.writer:
        tb.flush()
    if is_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return head, tok
