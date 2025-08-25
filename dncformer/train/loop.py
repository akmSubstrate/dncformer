# dncformer/train/loop.py
from __future__ import annotations
import math, time, json, contextlib
from typing import Optional, Tuple
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import CFG
from ..utils.env import sdpa_ctx, choose_amp_dtype
from ..utils.helpers import reduce_gate_tensor, gate_metrics, free_head_and_cache
from ..utils.helpers import causal_mask  # if needed elsewhere
from ..log.tb import TBLogger, TB_AVAILABLE, start_tb_run, tb
from ..data.mix import build_mixer
from ..data.synthetic import make_haystack_batch
from ..model.head import DNCFormerHead
from .optim import make_optimizer
from .scheduler import make_warmup_cosine_scheduler

def load_base_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                   if choose_amp_dtype(CFG.precision) != torch.float32 else None,
        device_map=None, trust_remote_code=True, attn_implementation="sdpa",
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    for p in model.parameters(): p.requires_grad_(False)
    model.config.output_hidden_states = True; model.config.use_cache = False
    return tok, model

def build_model_and_tokenizer():
    tok, base = load_base_model(CFG.base_model_id)
    if CFG.d_model is None:
        CFG.d_model = base.config.hidden_size
    head = DNCFormerHead(base, CFG).to(next(base.parameters()).device)
    if CFG.use_torch_compile and hasattr(torch, "compile"):
        head = torch.compile(head)
    return tok, head

def lm_shift_labels(input_ids, logits, tok):
    labels = input_ids[:, 1:].contiguous()
    logits = logits[:, :-1].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)),
                           labels.view(-1), ignore_index=tok.pad_token_id)

def evaluate_haystack(head, steps: int = 50, batch: int = 16, T: int = 256, vocab: int = 1024,
                      tb_step: int = None, fast: bool = False):
    head.eval()
    use_amp = (choose_amp_dtype(CFG.precision) in (torch.float16, torch.bfloat16)) and torch.cuda.is_available()
    accs, losses = [], []
    ctx = torch.inference_mode() if fast else torch.no_grad()
    with ctx:
        for _ in range(steps if not fast else min(steps, 10)):
            x, V, qpos = make_haystack_batch(batch if not fast else min(batch, 8),
                                             T if not fast else min(T, 128), vocab=vocab)
            x = x.to(next(head.parameters()).device); V = V.to(x.device); qpos = qpos.to(x.device)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=choose_amp_dtype(CFG.precision)):
                    logits, _g = head(x)
            else:
                logits, _g = head(x)
            idx = torch.arange(x.size(0), device=x.device)
            logits_q = logits[idx, qpos, :].float()
            loss = F.cross_entropy(logits_q, V)
            pred = logits_q.argmax(dim=-1)
            accs.append((pred == V).float().mean().item()); losses.append(loss.item())
    acc_m, loss_m = float(sum(accs)/len(accs)), float(sum(losses)/len(losses))
    if TB_AVAILABLE and ('tb' in globals()) and tb and tb.writer:
        tb.writer.add_scalar("eval/haystack_acc", acc_m, tb_step or 0)
        tb.writer.add_scalar("eval/haystack_loss", loss_m, tb_step or 0); tb.flush()
    head.train()
    print(f"[Haystack] acc={acc_m:.3f} | loss={loss_m:.3f} | fast={fast}")
    return acc_m, loss_m

def train_experiment(
    steps: int = None,
    batch_size: int = None,
    warmup_steps: int = None,
    min_lr_ratio: float = 0.1,
    mixture_weights=(0.4, 0.2, 0.2, 0.2),
    hf_dataset: str = "tatsu-lab/alpaca",
    hf_max_items: int = 5000,
    log_every: int = None,
    viz_memory_after: bool = False,
    viz_prompt: str = "### Instruction:\nSay hello in one word\n\n### Response:\n",
    viz_max_T: int = 64,
    mixture_schedule=None,
    gate_temp_schedule=None,
    gate_reg_schedule=None,
):
    steps = int(steps or CFG.train_steps)
    batch_size = int(batch_size or CFG.batch_size)
    warmup_steps = int(warmup_steps if warmup_steps is not None else max(10, steps//20))
    log_every = int(log_every if log_every is not None else CFG.log_every)

    tok, head = build_model_and_tokenizer()
    optim = make_optimizer(head, lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = make_warmup_cosine_scheduler(optim, warmup_steps, steps, min_lr_ratio=min_lr_ratio)
    mixer = build_mixer(tok, mixture_weights, hf_dataset=hf_dataset, hf_max_items=hf_max_items)

    def _apply_schedules(step: int):
        if mixture_schedule:
            for until, ws in mixture_schedule:
                if until is None or step <= int(until):
                    with contextlib.suppress(Exception): mixer.set_weights(ws); break
        if gate_temp_schedule:
            for until, temp in gate_temp_schedule:
                if until is None or step <= int(until):
                    CFG.gate_temp = float(temp); break
        if gate_reg_schedule:
            for until, lam in gate_reg_schedule:
                if until is None or step <= int(until):
                    CFG.gate_reg_lambda = float(lam); break

    if TB_AVAILABLE:
        try:
            tb  # may NameError
        except NameError:
            start_tb_run()
        if tb and tb.writer:
            tb.add_text("run/config", json.dumps({
                "steps": steps, "batch_size": batch_size, "warmup_steps": warmup_steps,
                "mixture_weights": list(mixture_weights),
                "mixture_schedule": mixture_schedule,
                "gate_temp_schedule": gate_temp_schedule,
                "gate_reg_schedule": gate_reg_schedule,
            }, indent=2), 0)

    head.train()
    amp_dtype = choose_amp_dtype(CFG.precision)
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype in (torch.float16, torch.bfloat16)))

    for step in range(1, steps + 1):
        _apply_schedules(step)
        in_ids = mixer(batch_size).to(next(head.parameters()).device)
        with torch.autocast('cuda', dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            logits, gates, aux = head.forward_with_metrics(in_ids, gate_override=getattr(CFG, "force_g", None))
            loss = lm_shift_labels(in_ids, logits, tok)

            div_lam = float(getattr(CFG, "expert_diversity_lambda", 0.0))
            if div_lam > 0.0 and isinstance(aux, dict) and "per_block" in aux:
                ent_vals = [float(m.get("experts_pi_entropy", float("nan"))) for m in aux["per_block"]]
                ent_vals = [x for x in ent_vals if not (x != x)]
                if ent_vals: loss = loss - div_lam * (sum(ent_vals)/len(ent_vals))

            lam = float(getattr(CFG, "gate_reg_lambda", 0.0))
            if lam > 0 and isinstance(gates, (list, tuple)) and len(gates) > 0:
                reg = 0.0
                for g in gates:
                    g2 = reduce_gate_tensor(g)
                    reg = reg + (g2.mean() * 0.0 + (g2 * (1 - g2)).mean())
                loss = loss + lam * reg

            lam_w = float(getattr(CFG, "write_reg_lambda", 0.0))
            if lam_w > 0.0 and isinstance(aux, dict) and "blocks" in aux:
                apply_reg = True
                if bool(getattr(CFG, "reg_only_on_memory_batches", True)):
                    bn = getattr(mixer, "last_name", "")
                    apply_reg = any(tk in bn for tk in ("copy","repeat","nback"))
                if apply_reg:
                    w_means = [m.get("write_gate_mean") for m in aux["blocks"] if isinstance(m, dict)]
                    w_means = [float(x) for x in w_means if isinstance(x, (float,int))]
                    if w_means: loss = loss + lam_w * (sum(w_means)/len(w_means))

        if amp_dtype in (torch.float16, torch.bfloat16):
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), CFG.grad_clip)
            scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), CFG.grad_clip)
            optim.step(); optim.zero_grad(set_to_none=True)

        scheduler.step()

        if step % log_every == 0 and tb and tb.writer:
            tb.writer.add_scalar("train/loss", float(loss.item()), step)
            tb.writer.add_scalar("train/lr", float(scheduler.get_last_lr()[0]), step)
            if isinstance(gates, (list, tuple)):
                for bi, g in enumerate(gates):
                    m, f, e = gate_metrics(g)
                    tb.writer.add_scalar(f"gates/block_{bi}_mean", m, step)
                    tb.writer.add_scalar(f"gates/block_{bi}_frac>0.5", f, step)
                    tb.writer.add_scalar(f"gates/block_{bi}_entropy", e, step)
                mix_name = getattr(mixer, "last_name", None) or "unknown"
                tb.writer.add_scalar(f"loss_by_task/{mix_name}", float(loss.item()), step)
                for bi, g in enumerate(gates):
                    m, f, _ = gate_metrics(g)
                    tb.writer.add_scalar(f"gates_by_task/block_{bi}_mean/{mix_name}", m, step)
                    tb.writer.add_scalar(f"gates_by_task/block_{bi}_frac>0.5/{mix_name}", f, step)

                if len(gates) > 0:
                    g0 = reduce_gate_tensor(gates[0].detach()); T = g0.size(1)
                    q = max(1, T // 4)
                    for qi, (s, e) in enumerate([(0,q),(q,2*q),(2*q,3*q),(3*q,T)], start=1):
                        tb.writer.add_scalar(f"gates/block0_q{qi}_mean/{mix_name}", float(g0[:, s:e].mean().item()), step)

                # Expert diagnostics
                for bi, b in enumerate(aux.get("blocks", [])):
                    if isinstance(b, dict) and "experts_pi_mean" in b:
                        for j, v in enumerate(b["experts_pi_mean"]):
                            tb.writer.add_scalar(f"experts/block_{bi}/pi_mean_{j}", float(v), step)
                    if isinstance(b, dict) and "experts_pi_entropy" in b:
                        tb.writer.add_scalar(f"experts/block_{bi}/pi_entropy", float(b["experts_pi_entropy"]), step)
                    if isinstance(b, dict) and "fusion_delta_norm" in b:
                        tb.writer.add_scalar(f"fusion/block_{bi}/delta_norm", float(b["fusion_delta_norm"]), step)
                    if isinstance(b, dict) and "write_gate_mean" in b:
                        tb.writer.add_scalar(f"reg/block_{bi}/write_gate_mean", float(b["write_gate_mean"]), step)

            tb.flush()

        if step % log_every == 0:
            gmeans = [float(reduce_gate_tensor(g).mean().item()) for g in gates] if isinstance(gates, (list, tuple)) else []
            print(f"step {step} | loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e} | gates={gmeans} | mix={getattr(mixer,'last_name','?')}")

    if tb and tb.writer: tb.flush()
    return head, tok
