# dncformer/model/blocks.py
from __future__ import annotations
from typing import Optional, List
import torch, torch.nn as nn
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

        self.vanilla = VanillaTransformerBlock(d_model, heads=heads, dropout=dropout, ffn_mult=ffn_mult)

        # per-block overrides
        N_, W_, R_ = N, W, R
        self.gate_temp = float(getattr(CFG, "gate_temp", 1.0))
        fbias = 0.0
        if isinstance(CFG.per_block_cfg, (list, tuple)) and self.block_index < len(CFG.per_block_cfg) and CFG.per_block_cfg[self.block_index]:
            blk = CFG.per_block_cfg[self.block_index]
            N_ = int(blk.get("N", N_)); W_ = int(blk.get("W", W_)); R_ = int(blk.get("R", R_))
            self.gate_temp = float(blk.get("gate_temp", self.gate_temp))
            fbias = float(blk.get("free_bias", 0.0))
        if CFG.per_block_free_bias and self.block_index < len(CFG.per_block_free_bias):
            fbias = float(CFG.per_block_free_bias[self.block_index])

        K = int(getattr(CFG, "mem_experts", 1))
        self.mem_experts = K
        if K == 1:
            self.dncblocks = nn.ModuleList([
                DNCformerBlock(d_in=d_in, d_model=d_model, R=R_, W=W_, N=N_,
                               heads=heads, dropout=dropout, ffn_mult=ffn_mult, free_bias=fbias)
            ])
        else:
            Ns = getattr(CFG, "expert_N", None)
            if not isinstance(Ns, (list, tuple)) or len(Ns) != K:
                Ns = [N_] * K

            Wexp = getattr(CFG, "expert_W", None)
            if not isinstance(Wexp, int) or Wexp <= 0:
                Wexp = W_
            self.dncblocks = nn.ModuleList([
                DNCformerBlock(d_in=d_in, d_model=d_model, R=R_, W=Wexp, N=Ns[i], # raises error
                               heads=heads, dropout=dropout, ffn_mult=ffn_mult, free_bias=fbias)
                for i in range(K)
            ])

        self.gate = nn.Linear((K+1) * d_model, (K+1))
        nn.init.constant_(self.gate.bias, float(gate_bias_init))

        self.fusion_enable = bool(getattr(CFG, "fusion_enable", False))
        if self.fusion_enable:
            fuse_in = d_model + W_
            hidden = int(CFG.fusion_hidden_mult * d_model)
            self.fuse_ln = nn.LayerNorm(fuse_in)
            self.fuse_mlp = nn.Sequential(
                nn.Linear(fuse_in, hidden),
                nn.GELU(),
                nn.Dropout(getattr(CFG, "fusion_drop", 0.0)),
                nn.Linear(hidden, d_model),
            )
        self.dropout = nn.Dropout(dropout)
        self.pre_gate_ln = nn.LayerNorm((K+1) * d_model)
        self.last_metrics = {}

    def forward(self, x: torch.Tensor, dnc_state=None, gate_override: float = None):
        B, T, D = x.shape
        mask = causal_mask(T, device=x.device)

        # vanilla path in its own dtype
        dtype_v = self.vanilla.ln1.weight.dtype
        x_v = x.to(dtype_v) if x.dtype != dtype_v else x
        vt = self.vanilla(x_v, attn_mask=mask)

        # memory experts
        states_in = dnc_state if isinstance(dnc_state, (list, tuple)) else [dnc_state]*self.mem_experts
        dts, states_out, per_mem_metrics = [], [], []
        last_read_feat = None
        for m, st in zip(self.dncblocks, states_in):
            dt, st2 = m(x, state=st)
            if dt.dtype != dtype_v: dt = dt.to(dtype_v)
            dts.append(dt); states_out.append(st2)
            pm = getattr(m, "last_metrics", {}) or {}
            per_mem_metrics.append(pm)
            if last_read_feat is None:
                lf = getattr(m, "last_read_feat", None)
                if lf is not None: last_read_feat = lf

        if self.fusion_enable:
            if last_read_feat is None:
                W = getattr(self.dncblocks[0], "W", self.d_model)
                last_read_feat = torch.zeros(B, T, W, device=x.device, dtype=dtype_v)
            else:
                last_read_feat = last_read_feat.to(dtype_v)
            dtype_f = self.fuse_ln.weight.dtype
            fuse_in = torch.cat([x_v, last_read_feat], dim=-1).to(dtype_f)
            delta = self.fuse_mlp(self.fuse_ln(fuse_in))
            if delta.dtype != vt.dtype: delta = delta.to(vt.dtype)
            vt = vt + delta
            self.last_metrics["fusion_delta_norm"] = float(delta.norm().detach().item() / max(1, delta.numel()))

        paths = [vt] + dts
        z = self.pre_gate_ln(torch.cat(paths, dim=-1))
        temp = float(getattr(CFG, "expert_gate_temp", getattr(self, "gate_temp", 1.0)))
        logits = self.gate(z.to(self.gate.weight.dtype)) / max(1e-6, temp)

        if gate_override is not None:
            g_mem = float(gate_override)
            pi = torch.zeros_like(logits); pi[...,0] = 1.0 - g_mem
            if self.mem_experts > 0: pi[...,1:] = g_mem/float(self.mem_experts)
        else:
            pi = torch.softmax(logits, dim=-1)

        pi_v = pi.to(vt.dtype)
        g_mem_synth = pi_v[...,1:].sum(dim=-1, keepdim=True)
        out = sum(pi_v[...,i:i+1]*p for i,p in enumerate(paths))
        out = self.dropout(out)

        with torch.no_grad():
            pi_mean = pi.mean(dim=(0,1))
            H = - (pi * (pi.clamp_min(1e-9).log())).sum(dim=-1).mean().item()
            self.last_metrics.update({
                "experts_pi_mean": [float(v) for v in pi_mean.detach().cpu()],
                "experts_pi_entropy": float(H),
                "write_gate_mean": float(mean_safely([m.get("write_gate_mean") for m in per_mem_metrics])),
            })
        return out, (states_out if self.mem_experts>1 else states_out[0]), g_mem_synth
