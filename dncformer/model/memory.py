# dncformer/model/memory.py
from __future__ import annotations
from typing import Optional, Dict
import torch, torch.nn as nn, torch.nn.functional as F
from ..utils.helpers import causal_mask
from ..utils.helpers import mean_safely

class TransformerController(nn.Module):
    def __init__(self, d_in: int, d_model: int, heads: int, dropout: float=0.1, ffn_mult: float=4.0):
        super().__init__()
        self.proj_in = nn.Linear(d_in, d_model)
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

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        h = self.proj_in(x)
        h = self.ln1(h)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        h = h + self.dropout(a)
        h = self.ln2(h)
        z2 = self.ff(h)
        return h + self.dropout(z2)

class DNCInterfaceHead(nn.Module):
    def __init__(self, d_model: int, R: int, W: int):
        super().__init__()
        self.R, self.W = R, W
        out = R*W + R + W + 1 + W + W + R + 1 + 1 + R*3
        self.proj = nn.Linear(d_model, out)

    def forward(self, h: torch.Tensor):
        B, T, D = h.shape; v = self.proj(h); idx = 0
        def take(sz):
            nonlocal idx
            part = v[..., idx:idx+sz]; idx += sz; return part
        R, W = self.R, self.W
        k_read = take(R*W).view(B,T,R,W)
        beta_read = F.softplus(take(R)).view(B,T,R,1)
        k_write = take(W).view(B,T,W)
        beta_write = F.softplus(take(1)).view(B,T,1)
        erase = torch.sigmoid(take(W)).view(B,T,W)
        write_vec = take(W).view(B,T,W)
        free_gates = torch.sigmoid(take(R)).view(B,T,R,1)
        alloc_gate = torch.sigmoid(take(1)).view(B,T,1)
        write_gate = torch.sigmoid(take(1)).view(B,T,1)
        read_mode = take(R*3).view(B,T,R,3)
        return {"k_read": k_read, "beta_read": beta_read,
                "k_write": k_write, "beta_write": beta_write,
                "erase": erase, "write_vec": write_vec,
                "free_gates": free_gates, "alloc_gate": alloc_gate,
                "write_gate": write_gate, "read_mode": read_mode}

class DNCMemory(nn.Module):
    def __init__(self, nr_cells: int, cell_size: int, read_heads: int):
        super().__init__()
        self.N, self.W, self.R = nr_cells, cell_size, read_heads
        self.free_bias = 0.0
        self.probe = None
        self._last_metrics = {}

    def reset(self, B: int, device=None):
        device = device or "cpu"
        M = torch.zeros(B, self.N, self.W, device=device)
        u = torch.zeros(B, self.N, device=device)
        L = torch.zeros(B, self.N, self.N, device=device)
        p = torch.zeros(B, self.N, device=device)
        rw = F.one_hot(torch.zeros(B, self.R, dtype=torch.long, device=device), num_classes=self.N).float()
        r = torch.zeros(B, self.R, self.W, device=device)
        return {"M": M, "u": u, "L": L, "p": p, "rw": rw, "r": r}

    @staticmethod
    def _cosine_sim(M: torch.Tensor, k: torch.Tensor):
        if k.dim() == 2: k = k.unsqueeze(1)
        Mnorm = F.normalize(M, p=2, dim=-1); kn = F.normalize(k, p=2, dim=-1)
        return torch.einsum("bnw,brw->brn", Mnorm, kn)  # (B,R,N)

    def _allocation(self, u: torch.Tensor):
        δ = 1e-6; u = δ + (1 - δ) * u
        B, N = u.shape
        sorted_u, phi = torch.sort(u, dim=-1, descending=False)
        ones = torch.ones(B, 1, device=u.device, dtype=u.dtype)
        prod_excl = torch.cumprod(torch.cat([ones, sorted_u], dim=1), dim=1)[:, :-1]
        a_sorted = (1 - sorted_u) * prod_excl
        inv_phi = torch.argsort(phi, dim=-1)
        return a_sorted.gather(1, inv_phi)

    def forward(self, x_if: dict, state: dict):
        M, u, L, p, rw, r = state["M"], state["u"], state["L"], state["p"], state["rw"], state["r"]

        ww_prev = state.get("ww", torch.zeros(M.size(0), self.N, device=M.device, dtype=M.dtype))
        u = u + (1 - u) * (1 - ww_prev).clamp(min=0.0, max=1.0)

        free_g = x_if["free_gates"]
        if getattr(self, "free_bias", 0.0) != 0.0:
            free_g = (free_g + self.free_bias).clamp(0.0, 0.1)
        psi = torch.prod(1 - free_g * rw, dim=1)
        u = torch.clamp(u * psi, 0, 1)

        sim_w = self._cosine_sim(M, x_if["k_write"])  # (B,R,N) or (B,1,N)
        beta_w = x_if["beta_write"]
        if beta_w.dim() == 2: beta_w = beta_w.unsqueeze(-1)
        cw = torch.softmax(sim_w * beta_w, dim=-1)    # (B,R,N)
        if cw.size(1) > 1: cw = cw.mean(dim=1)       # (B,N)
        else: cw = cw.squeeze(1)                      # (B,N)

        a = self._allocation(u)                       # (B,N)
        alloc = x_if["alloc_gate"]                   # (B,1)
        write_gate = x_if["write_gate"]              # (B,1)
        ww = write_gate * (alloc * a + (1.0 - alloc) * cw)
        state["ww"] = ww

        erase = x_if["erase"].unsqueeze(1)
        write_vec = x_if["write_vec"].unsqueeze(1)
        M = M * (1 - ww.unsqueeze(-1) * erase) + ww.unsqueeze(-1) * write_vec

        prev_p = p
        p = (1 - ww.sum(dim=-1, keepdim=True)) * p + ww
        L = (1 - ww.unsqueeze(2) - ww.unsqueeze(1)) * L + torch.einsum("bn,bm->bnm", prev_p, ww)
        L = L * (1 - torch.eye(self.N, device=M.device).unsqueeze(0))

        cr = torch.softmax(self._cosine_sim(M, x_if["k_read"]) * x_if["beta_read"], dim=-1)
        fwd = torch.einsum("brn,bnm->brm", rw, L)
        bwd = torch.einsum("brn,bmn->brm", rw, L)
        read_mode = torch.softmax(x_if["read_mode"], dim=-1)
        rw = read_mode[:,:,0:1]*bwd + read_mode[:,:,1:2]*cr + read_mode[:,:,2:3]*fwd
        r = torch.einsum("brn,bnw->brw", rw, M)

        state = {"M": M, "u": u, "L": L, "p": p, "rw": rw, "r": r}

        try:
            wg = x_if.get("write_gate", None)
            write_gate_mean = float(wg.mean().detach().item()) if wg is not None else float("nan")
            read_norm = float(r.norm().detach().item()/max(1, r.numel()))
            self._last_metrics = {"write_gate_mean": write_gate_mean, "read_vec_norm": read_norm}
        except Exception:
            self._last_metrics = {}

        if self.probe is not None:
            try:
                self.probe({
                    "u": u.detach().float().cpu(),
                    "ww": ww.detach().float().cpu(),
                    "rw": rw.detach().float().cpu(),
                    "M_norm": M.detach().float().cpu().norm(dim=-1),
                    "L_diag_mean": torch.diagonal(L, dim1=-2, dim2=-1).mean(dim=-1),
                })
            except Exception:
                pass
        return r, state

class DNCformerBlock(nn.Module):
    def __init__(self, d_in: int, d_model: int, R: int, W: int, N: int,
                 heads: int, dropout: float, ffn_mult: float, free_bias: float = 0.0):
        super().__init__()
        self.R, self.W, self.N = R, W, N
        self.ctrl = TransformerController(d_in + R*W, d_model, heads=heads, dropout=dropout, ffn_mult=ffn_mult)
        self.if_head = DNCInterfaceHead(d_model, R=R, W=W)
        self.mem = DNCMemory(nr_cells=N, cell_size=W, read_heads=R)
        self.mem.free_bias = float(free_bias)
        self.out_proj = nn.Linear(d_model + R*W, d_model)
        self.last_metrics = {}
        self.last_read_feat = None

    def forward(self, x: torch.Tensor, state: Optional[dict]=None):
        B, T, D = x.shape
        if state is None: state = self.mem.reset(B, device=x.device)
        reads = state["r"].reshape(B, self.R*self.W)
        reads_seq = reads.unsqueeze(1).expand(B, T, self.R*self.W)
        ctrl_in = torch.cat([x, reads_seq], dim=-1)
        h = self.ctrl(ctrl_in, attn_mask=causal_mask(T, device=x.device))

        r_list, new_state = [], state
        iface = self.if_head(h)
        for t in range(T):
            x_if = {k: v[:,t] for k, v in iface.items()}
            r_t, new_state = self.mem(x_if, new_state)
            r_list.append(r_t)
        Rseq = torch.stack(r_list, dim=1)  # (B,T,R,W)
        reads_flat = Rseq.reshape(B,T,self.R*self.W)
        y = self.out_proj(torch.cat([h, reads_flat], dim=-1))

        try:
            self.last_read_feat = Rseq.mean(dim=2).detach()
            rnorm = float(self.last_read_feat.norm().item() / max(1, self.last_read_feat.numel()))
        except Exception:
            self.last_read_feat = None; rnorm = float("nan")

        lm = getattr(self.mem, "_last_metrics", None)
        wg_mean = float(lm.get("write_gate_mean", float("nan"))) if isinstance(lm, dict) else float("nan")
        self.last_metrics = {"read_vec_norm": rnorm, "write_gate_mean": wg_mean}
        return y, new_state
