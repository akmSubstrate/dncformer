# tests/test_optim_defaults.py
from dncformer.train.loop import build_model_and_tokenizer
from dncformer.train.optim import make_optimizer
from dncformer.config import CFG

def test_make_optimizer_defaults():
    tok, head = build_model_and_tokenizer()
    # Count trainables before constructing the optimizer
    expected = sum(p.numel() for p in head.parameters() if p.requires_grad)
    opt = make_optimizer(head)  # no args: pulls lr/weight_decay from CFG
    got = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert got == expected, f"optimizer did not capture all trainables: {got} vs {expected}"
