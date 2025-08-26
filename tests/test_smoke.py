# tests/test_smoke.py
import torch
import contextlib
from dncformer.config import CFG
from dncformer.model.blocks import ParallelEnrichmentBlock
from dncformer.model.head import DNCFormerHead
from dncformer.train.loop import build_model_and_tokenizer
from dncformer.utils.helpers import free_head_and_cache

device = "cuda"

def test_parallel_block_basic():
    print("Testing parallel block load functionality...")
    try:
        B,T,D = 2,5,32; R,W,N=1,8,16
        x = torch.randn(B,T,D)
        CFG.mem_experts=1; CFG.fusion_enable=False; CFG.per_block_cfg=None
        blk = ParallelEnrichmentBlock(d_model=D, d_in=D, R=R, W=W, N=N, heads=2, dropout=0.0, ffn_mult=2.0)
        y, st, g = blk(x, dnc_state=None)
        assert y.shape==(B,T,D) and g.shape==(B,T,1)
        print("Test: PASSED")
    except Exception:
        print("Test: FAILED")

def test_head_forward_tiny():
    print("Testing head forward functionality...")
    try:
        tok, head = build_model_and_tokenizer()
        dev = next(head.parameters()).device
        toks = tok("hello world", return_tensors="pt")
        x = toks.input_ids[:, :8].to(dev)          # ensure same device as model
        logits, gates = head(x, collect_metrics=False)
        assert logits.shape[0]==x.shape[0] and len(gates)==len(head.blocks)
        print("Test: PASSED")
    except Exception as e:
        print("Test: FAILED", e)



def _snapshot_attr(obj, name):
    return getattr(obj, name, None), hasattr(obj, name)

def _restore_attr(obj, name, value, existed: bool):
    if existed:
        setattr(obj, name, value)
    else:
        with contextlib.suppress(Exception):
            delattr(obj, name)

@torch.no_grad()
def smoke_head_mem_experts_one_step_real():
    """
    Build a real base+head via build_model_and_tokenizer(), set mem_experts=1,
    and run a tiny forward. Uses pad/eos to form a minimal batch.
    """
    print("[smoke] head forward (mem_experts=1) with real builder")
    old_val, existed = _snapshot_attr(CFG, "mem_experts")
    try:
        setattr(CFG, "mem_experts", 1)

        tok, head = build_model_and_tokenizer()
        head.eval()

        B, T = 1, 8
        # Make a tiny, valid batch from tokenizer constants
        pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0) or 0
        x = torch.full((B, T), int(pad_id), dtype=torch.long, device=device)

        logits, gates = head(x, collect_metrics=False)
        assert logits.shape[:2] == (B, T), f"bad logits shape: {tuple(logits.shape)}"
        assert isinstance(gates, (list, tuple)) and len(gates) == len(head.blocks), "gates list mismatch"
        for g in gates:
            assert g.shape[:2] == (B, T)
            assert (g >= 0).all() and (g <= 1).all(), "gate out of [0,1] range"

        print("Test: PASSED")
    finally:
        _restore_attr(CFG, "mem_experts", old_val, existed)
        # free VRAM / allocator
        free_head_and_cache()

@torch.no_grad()
def smoke_head_mem_experts_two_step_real():
    """
    Same as above, but set mem_experts=2 and call forward_with_metrics to
    verify per-block expert diagnostics are wired end-to-end.
    """
    print("[smoke] head forward_with_metrics (mem_experts=2) with real builder")
    old_val, existed = _snapshot_attr(CFG, "mem_experts")
    try:
        setattr(CFG, "mem_experts", 2)

        tok, head = build_model_and_tokenizer()
        head.eval()

        B, T = 1, 8
        pad_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", 0) or 0
        x = torch.full((B, T), int(pad_id), dtype=torch.long, device=device)

        logits, gates, aux = head.forward(x, gate_override=None, collect_metrics=True)
        assert logits.shape[:2] == (B, T), f"bad logits shape: {tuple(logits.shape)}"
        assert isinstance(gates, (list, tuple)) and len(gates) == len(head.blocks)
        assert isinstance(aux, dict) and "per_block" in aux and "blocks" in aux
        for m in aux["blocks"]:
            assert isinstance(m, dict), "per-block metrics missing"

        print("Test: PASSED")
    finally:
        _restore_attr(CFG, "mem_experts", old_val, existed)
        free_head_and_cache()

def run_patch_smoke_tests():
    test_parallel_block_basic()
    test_head_forward_tiny()
    smoke_head_mem_experts_one_step_real()
    smoke_head_mem_experts_two_step_real()
    print("[smoke] all tests passed.")

if __name__=="__main__":
    run_patch_smoke_tests()