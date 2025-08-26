# tests/test_router_balance_and_dropout.py
import torch
from dncformer.config import CFG
from dncformer.model.blocks import ParallelEnrichmentBlock

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def test_router_noise_and_expert_dropout_smoke():
    B,T,D = 2, 12, 32
    R,W,N = 1, 8, 32
    x = torch.randn(B,T,D, device=device)

    # enable A2 + expert dropout
    CFG.mem_experts = 2
    CFG.router_noise_std = 0.05
    CFG.expert_dropout_p = 0.25
    CFG.per_block_cfg = None

    blk = ParallelEnrichmentBlock(d_model=D, d_in=D, R=R, W=W, N=N,
                                  heads=2, dropout=0.0, ffn_mult=2.0).to(device)
    blk.train()
    y, st, g = blk(x, dnc_state=None)
    assert y.shape == (B,T,D)
    assert g.shape == (B,T,1)
    assert torch.isfinite(g).all()

@torch.no_grad()
def test_memory_block_memdrop_smoke():
    B,T,D = 2, 8, 32
    R,W,N = 1, 8, 32
    x = torch.randn(B,T,D, device=device)

    # force full memory dropout inside the block to verify gating→vanilla
    CFG.mem_experts = 1
    CFG.router_noise_std = 0.0
    CFG.expert_dropout_p = 0.0
    CFG.per_block_cfg = None

    blk = ParallelEnrichmentBlock(d_model=D, d_in=D, R=R, W=W, N=N,
                                  heads=2, dropout=0.0, ffn_mult=2.0).to(device)
    blk.train()

    # Simulate block-local dropout by passing gate_override=0.0
    y, st, g = blk(x, dnc_state=None, gate_override=0.0)
    assert (g < 1e-6).all(), "memory fraction should be ~0 when forced drop is active"
    assert y.shape == (B,T,D)

if __name__ == "__main__":
    test_router_noise_and_expert_dropout_smoke()
    test_memory_block_memdrop_smoke()
    print("router balance/dropout smoke: OK")
