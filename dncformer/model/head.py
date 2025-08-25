# dncformer/model/head.py
from __future__ import annotations
from typing import Optional, List
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from .blocks import ParallelEnrichmentBlock
from ..config import CFG
from ..utils.env import sdpa_ctx
from ..utils.helpers import gate_metrics

class DNCFormerHead(nn.Module):
    def __init__(self, base: AutoModelForCausalLM, cfg):
        super().__init__()
        self.base = base
        d_model = base.config.hidden_size if cfg.d_model is None else cfg.d_model
        self.d_model = d_model
        self.blocks = nn.ModuleList([
            ParallelEnrichmentBlock(
                d_model=d_model, d_in=d_model,
                R=cfg.dnc_read_heads, W=cfg.dnc_cell_size, N=cfg.dnc_nr_cells,
                heads=cfg.attn_heads, dropout=cfg.attn_dropout,
                ffn_mult=cfg.ffn_mult, gate_bias_init=cfg.gate_bias_init,
                block_index=i,
            ) for i in range(cfg.n_blocks)
        ])
        self.proj_out = nn.Identity()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None):
        with torch.no_grad():
            with sdpa_ctx():
                out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1]
        dnc_states = [None] * len(self.blocks)
        gates = []
        for i, blk in enumerate(self.blocks):
            st_in = dnc_states[i]
            K = int(getattr(blk, "mem_experts", 1))
            if K > 1 and not isinstance(st_in, (list, tuple)):
                st_in = [st_in] * K
            h, dnc_states[i], g = blk(h, dnc_state=st_in)
            gates.append(g.detach())
        logits = self.base.lm_head(self.proj_out(h).to(self.base.lm_head.weight.dtype))
        return logits, gates

    def forward_with_metrics(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                             gate_override: Optional[float] = None):
        with torch.no_grad():
            with sdpa_ctx():
                out = self.base(input_ids=input_ids, attention_mask=attention_mask,
                                output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1].to(next(self.parameters()).device)
        dnc_states = [None] * len(self.blocks)
        gates_det, gates_raw, per_block = [], [], []

        for blk in self.blocks:
            if hasattr(blk, "collect_metrics"): blk.collect_metrics = True
        for i, blk in enumerate(self.blocks):
            st_in = dnc_states[i]
            K = int(getattr(blk, "mem_experts", 1))
            if K > 1 and not isinstance(st_in, (list, tuple)): st_in = [st_in] * K
            h, dnc_states[i], g = blk(h, dnc_state=st_in, gate_override=gate_override)
            gates_raw.append(g); gates_det.append(g.detach())
            per_block.append(getattr(blk, "last_metrics", {}) or {})
            if hasattr(blk, "collect_metrics"): blk.collect_metrics = False

        lm_dev = self.base.lm_head.weight.device
        y = self.proj_out(h).to(lm_dev, dtype=self.base.lm_head.weight.dtype)
        logits = self.base.lm_head(y).to(next(self.parameters()).device)

        aux = {"per_block": per_block, "blocks": per_block, "gates_raw": gates_raw, "gates_detached": gates_det}
        aux["g_entropy_block"] = []
        for g in gates_det:
            _, _, ent = gate_metrics(g)
            aux["g_entropy_block"].append(ent)
        return logits, gates_det, aux
