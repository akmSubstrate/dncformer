# dncformer/model/blocks.py
from __future__ import annotations
from typing import Optional, List
import torch, torch.nn as nn
import torch.nn.functional as F
from .memory import DNCformerBlock
from ..config import CFG
from ..utils.helpers import causal_mask, mean_safely

class VanillaTransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float=0.1, ffn_mult: float=4.0):
        super().__init__()
        self.collect_metrics = False
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, int(d_model*ffn_mult)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model*ffn_mult), d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        h = x + self.dropout(a)
        z = self.ln2(h)
        z2 = self.ff(z)
        return h + self.dropout(z2)

class ParallelEnrichmentBlock(nn.Module):
    def __init__(self, d_model: int, d_in: int, R: int, W: int, N: int,
                 heads: int = 4, dropout: float = 0.1, ffn_mult: float = 4.0,
                 block_index: int = 0, gate_bias_init: float = -1.0):
        super().__init__()
        # … existing init …
        self.router_noise_std = float(getattr(CFG, "router_noise_std", 0.0))
        self.expert_dropout_p = float(getattr(CFG, "expert_dropout_p", 0.0))
        self._lb_loss = None  # A1: populated per forward when enabled

    def forward(self, x: torch.Tensor, dnc_state=None, gate_override: float = None):
        B, T, D = x.shape
        mask = causal_mask(T, device=x.device)

        # 1) vanilla path
        vt = self.vanilla(x, attn_mask=mask)

        # 2) memory experts
        states_in = dnc_state if isinstance(dnc_state, (list, tuple)) else [dnc_state]*self.mem_experts
        dts, states_out, per_mem_metrics = [], [], []
        last_read_feat = None
        for m, st in zip(self.dncblocks, states_in):
            dt, st2 = m(x, state=st)
            dts.append(dt); states_out.append(st2)
            per_mem_metrics.append(getattr(m, "last_metrics", {}) or {})
            if last_read_feat is None:
                last_read_feat = getattr(m, "last_read_feat", None)

        # 3) optional fusion
        if self.fusion_enable:
            r_feat = last_read_feat
            if r_feat is None:
                r_feat = torch.zeros(B, T, self.dncblocks[0].W, device=x.device, dtype=vt.dtype)
            fuse_in = torch.cat([x, r_feat.to(vt.dtype)], dim=-1)
            delta = self.fuse_mlp(self.fuse_ln(fuse_in))
            vt = vt + delta
            self.last_metrics["fusion_delta_norm"] = float(delta.norm().detach().item() / max(1, delta.numel()))

        # 4) gate mixture
        paths = [vt] + dts
        z = self.pre_gate_ln(torch.cat(paths, dim=-1))
        logits = self.gate(z)

        # ---- B2 (block-local) & A2 & expert-dropout on logits BEFORE softmax ----
        if self.training:
            # A2: small Gaussian noise on memory logits only (keeps vanilla stable)
            if self.router_noise_std > 0.0:
                noise = torch.randn_like(logits[..., 1:]) * float(self.router_noise_std)
                logits = torch.cat([logits[..., :1], logits[..., 1:] + noise], dim=-1)
                self.last_metrics["router_noise_std"] = float(self.router_noise_std)
            # Expert dropout (memory experts only); ensure not all dropped
            if self.expert_dropout_p > 0.0 and self.mem_experts > 1:
                drop = (torch.rand(B, T, self.mem_experts, device=logits.device) < float(self.expert_dropout_p))
                # Ensure at least one expert remains per token
                all_dropped = drop.all(dim=-1, keepdim=True)
                if all_dropped.any():
                    # randomly keep one expert
                    keep_idx = torch.randint(0, self.mem_experts, (B, T, 1), device=logits.device)
                    drop = drop.scatter(-1, keep_idx, False)
                mem_logits = logits[..., 1:].masked_fill(drop, -1e9)
                logits = torch.cat([logits[..., :1], mem_logits], dim=-1)
                self.last_metrics["experts_drop_frac"] = float(drop.float().mean().item())

        temp = max(1e-6, float(getattr(CFG, "expert_gate_temp", getattr(self, "gate_temp", 1.0))))
        logits = logits / temp

        # Optional global override (force memory fraction)
        if gate_override is not None:
            g_mem = float(gate_override)
            pi = torch.zeros_like(logits)            # (B,T,K+1)
            pi[...,0] = 1.0 - g_mem                  # vanilla
            if self.mem_experts > 0:
                pi[...,1:] = g_mem / float(self.mem_experts)
        else:
            pi = torch.softmax(logits, dim=-1)

        g_mem_synth = pi[...,1:].sum(dim=-1, keepdim=True)
        out = sum(pi[...,i:i+1] * p for i,p in enumerate(paths))
        out = self.dropout(out)

        # ---- A1 (soft balance) raw loss (differentiable) ----
        self._lb_loss = None
        lam = float(getattr(CFG, "router_balance_lambda", 0.0))
        if self.training and lam > 0.0 and self.mem_experts > 1:
            mu = pi[..., 1:].mean(dim=(0,1))               # (K,)
            u = 1.0 / float(self.mem_experts)
            lb = ((mu - u)**2).mean()                      # scalar tensor
            self._lb_loss = lb                              # weight is applied in train loop

        with torch.no_grad():
            pi_mean = pi.mean(dim=(0,1))
            H = - (pi * (pi.clamp_min(1e-9).log())).sum(dim=-1).mean().item()
            self.last_metrics.update({
                "experts_pi_mean": [float(v) for v in pi_mean.detach().cpu()],
                "experts_pi_entropy": float(H),
                "write_gate_mean": float(mean_safely([m.get("write_gate_mean", float("nan")) for m in per_mem_metrics])),
            })

        return out, (states_out if self.mem_experts>1 else states_out[0]), g_mem_synth
