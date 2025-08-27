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
        self.d_model = d_model
        self.block_index = block_index

        # 0) Vanilla Transformer sub-block
        self.vanilla = VanillaTransformerBlock(
            d_model, heads=heads, dropout=dropout, ffn_mult=ffn_mult
        )

        # 1) Per-block overrides
        N_, W_, R_ = N, W, R
        self.gate_temp = float(getattr(CFG, "gate_temp", 1.0))
        fbias = 0.0
        pbc = getattr(CFG, "per_block_cfg", None)
        if isinstance(pbc, (list, tuple)) and self.block_index < len(pbc) and pbc[self.block_index]:
            blk = pbc[self.block_index]
            N_ = int(blk.get("N", N_)); W_ = int(blk.get("W", W_)); R_ = int(blk.get("R", R_))
            self.gate_temp = float(blk.get("gate_temp", self.gate_temp))
            fbias = float(blk.get("free_bias", 0.0))
        pbf = getattr(CFG, "per_block_free_bias", None)
        if pbf and self.block_index < len(pbf):
            fbias = float(pbf[self.block_index])

        # 2) Memory experts — shared IO signature, K experts per block
        K = int(getattr(CFG, "mem_experts", 1))
        self.mem_experts = K

        # robust expert_N: use per-block N_ if missing/short/invalid
        ns_cfg = getattr(CFG, "expert_N", None)
        if isinstance(ns_cfg, (list, tuple)) and len(ns_cfg) >= K:
            try:
                Ns_list = [int(ns_cfg[i]) for i in range(K)]
            except Exception:
                Ns_list = [int(N_)] * K
        else:
            Ns_list = [int(N_)] * K

        # robust expert_W: fall back to per-block W_ when None/missing
        w_cfg = getattr(CFG, "expert_W", None)
        Wexp = int(W_) if (w_cfg is None) else int(w_cfg)

        if K == 1:
            self.dncblocks = nn.ModuleList([
                DNCformerBlock(d_in=d_in, d_model=d_model, R=R_, W=W_, N=N_,
                               heads=heads, dropout=dropout, ffn_mult=ffn_mult, free_bias=fbias)
            ])
        else:
            self.dncblocks = nn.ModuleList([
                DNCformerBlock(d_in=d_in, d_model=d_model, R=R_, W=Wexp, N=Ns_list[i],
                               heads=heads, dropout=dropout, ffn_mult=ffn_mult, free_bias=fbias)
                for i in range(K)
            ])

        # 3) Gating head (vanilla + K experts)
        self.gate = nn.Linear((K + 1) * d_model, (K + 1))
        nn.init.constant_(self.gate.bias, float(gate_bias_init))

        # 4) Optional fusion - |||DANGER: HEREIN LIE MEMORY LEAKS|||
        self.fusion_enable = bool(getattr(CFG, "fusion_enable", False))
        if self.fusion_enable:
            fuse_in = d_model + W_   # concat [x, pooled-read-feat]
            hidden = int(getattr(CFG, "fusion_hidden_mult", 2.0) * d_model)
            self.fuse_ln = nn.LayerNorm(fuse_in)
            self.fuse_mlp = nn.Sequential(
                nn.Linear(fuse_in, hidden),
                nn.GELU(),
                nn.Dropout(getattr(CFG, "fusion_drop", 0.0)),
                nn.Linear(hidden, d_model),
            )

        # 5) Collapse guards (A2/B2)
        self.router_noise_std = float(getattr(CFG, "router_noise_std", 0.0))   # small Gaussian on logits
        self.expert_dropout_p = float(getattr(CFG, "expert_dropout_p", 0.0))   # small prob to drop experts per batch

        # 6) Housekeeping
        self.dropout = nn.Dropout(dropout)
        self.pre_gate_ln = nn.LayerNorm((K + 1) * d_model)
        self.last_metrics = {}

    def forward(self, x: torch.Tensor, dnc_state=None, gate_override: float = None):
        """
        x: (B,T,D)
        Returns:
          - out: (B,T,D)
          - new_state: state or list-of-states (len=K) if mem_experts>1
          - g_mem_synth: (B,T,1) = sum of expert probs (for back-compat logging)
        """
        B, T, D = x.shape
        mask = causal_mask(T, device=x.device)

        # 1) Vanilla path (dtype-safe: drive inputs by LN param dtype)
        dtype_v = self.vanilla.ln1.weight.dtype
        x_v = x.to(dtype_v) if x.dtype != dtype_v else x
        vt = self.vanilla(x_v, attn_mask=mask)  # (B,T,D), dtype=dtype_v

        # 2) Memory experts (run in their native dtype, cast to dtype_v for fusion/gate)
        K = int(getattr(self, "mem_experts", 1))
        states_in = dnc_state if isinstance(dnc_state, (list, tuple)) else [dnc_state] * K
        dts, states_out, per_mem_metrics = [], [], []
        last_read_feat = None  # (B,T,W) pooled read vectors if provided by block

        for m, st in zip(self.dncblocks, states_in):
            dt, st2 = m(x, state=st)               # use original x (keeps DNC path native)
            if dt.dtype != dtype_v: dt = dt.to(dtype_v)
            dts.append(dt); states_out.append(st2)
            pm = getattr(m, "last_metrics", {}) or {}
            per_mem_metrics.append(pm)
            if last_read_feat is None:
                lf = getattr(m, "last_read_feat", None)
                if lf is not None:
                    last_read_feat = lf.to(dtype_v)

        # 3) Optional fusion (read-to-attention hint)
        if self.fusion_enable:
            if last_read_feat is None:
                W = getattr(self.dncblocks[0], "W", self.d_model)
                last_read_feat = torch.zeros(B, T, W, device=x.device, dtype=dtype_v)
            dtype_f = self.fuse_ln.weight.dtype
            fuse_in = torch.cat([x_v, last_read_feat], dim=-1)
            fuse_in = fuse_in.to(dtype_f) if fuse_in.dtype != dtype_f else fuse_in
            delta = self.fuse_mlp(self.fuse_ln(fuse_in))
            if delta.dtype != vt.dtype: delta = delta.to(vt.dtype)
            vt = vt + delta
            self.last_metrics["fusion_delta_norm"] = float(delta.norm().detach().item() / max(1, delta.numel()))

        # 4) Router (gate over vanilla + K experts)
        paths = [vt] + dts
        z = self.pre_gate_ln(torch.cat(paths, dim=-1))
        # compute logits in gate's dtype
        dtype_g = self.gate.weight.dtype
        z_g = z.to(dtype_g) if z.dtype != dtype_g else z
        temp = float(getattr(CFG, "expert_gate_temp", getattr(self, "gate_temp", 1.0)))
        logits = self.gate(z_g) / max(1e-6, temp)

        # 4.a) Override (used by tests & ablations) — bypass noise/dropout
        if gate_override is not None:
            g_mem = float(gate_override)
            pi = torch.zeros_like(logits)
            pi[..., 0] = 1.0 - g_mem               # vanilla
            if K > 0: pi[..., 1:] = g_mem / float(K)
        else:
            # 4.b) B2: expert dropout (per forward; keep at least one expert)
            if self.training and (self.expert_dropout_p > 0.0) and K > 0:
                drop = torch.rand(K, device=x.device) < self.expert_dropout_p
                if bool(drop.all()):
                    drop[torch.randint(0, K, (1,), device=x.device)] = False
                if bool(drop.any()):
                    mask = drop.view(1, 1, K).expand(B, T, K)  # (B,T,K)
                    logits[..., 1:] = logits[..., 1:].masked_fill(mask, -1e4)

            # 4.c) A2: tiny Gaussian noise on router logits (training only)
            if self.training and (self.router_noise_std > 0.0):
                logits = logits + torch.randn_like(logits) * self.router_noise_std

            pi = torch.softmax(logits, dim=-1)

        # Synthesize a single "memory prob" for legacy metrics and weighted sum
        pi_v = pi.to(dtype_v) if pi.dtype != dtype_v else pi
        g_mem_synth = pi_v[..., 1:].sum(dim=-1, keepdim=True)  # (B,T,1)
        out = sum(pi_v[..., i:i+1] * p for i, p in enumerate(paths))
        out = self.dropout(out)

        # 5) Metrics for TB
        with torch.no_grad():
            pi_mean = pi.mean(dim=(0, 1))     # (K+1,)
            H = - (pi * (pi.clamp_min(1e-9).log())).sum(dim=-1).mean().item()
            # NOTE: If your helpers export `_mean_safely` under a different name, keep the one you use elsewhere.
            try:
                wg_mean = mean_safely([m.get("write_gate_mean", float("nan")) for m in per_mem_metrics])
            except NameError:
                # fallback if your helpers expose `mean_safely` instead
                from ..utils.helpers import mean_safely as _mean_safely  # type: ignore
                wg_mean = _mean_safely([m.get("write_gate_mean", float("nan")) for m in per_mem_metrics])

            self.last_metrics.update({
                "experts_pi_mean": [float(v) for v in pi_mean.detach().cpu()],
                "experts_pi_entropy": float(H),
                "write_gate_mean": float(wg_mean),
            })

        return out, (states_out if K > 1 else states_out[0]), g_mem_synth
