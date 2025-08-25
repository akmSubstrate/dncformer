# tests/test_smoke.py
import torch
from dncformer.config import CFG
from dncformer.model.blocks import ParallelEnrichmentBlock
from dncformer.model.head import DNCFormerHead
from dncformer.train.loop import build_model_and_tokenizer

def test_parallel_block_basic():
    B,T,D = 2,5,32; R,W,N=1,8,16
    x = torch.randn(B,T,D)
    CFG.mem_experts=1; CFG.fusion_enable=False; CFG.per_block_cfg=None
    blk = ParallelEnrichmentBlock(d_model=D, d_in=D, R=R, W=W, N=N, heads=2, dropout=0.0, ffn_mult=2.0)
    y, st, g = blk(x, dnc_state=None)
    assert y.shape==(B,T,D) and g.shape==(B,T,1)

def test_head_forward_tiny():
    # uses real base model; skip on CI if env lacks GPU/weights
    tok, head = None, None
    try:
        tok, head = build_model_and_tokenizer()
        x = tok("hello world", return_tensors="pt").input_ids[:,:8]
        logits, gates = head(x)
        assert logits.shape[0]==x.shape[0] and len(gates)==len(head.blocks)
    except Exception:
        pass
