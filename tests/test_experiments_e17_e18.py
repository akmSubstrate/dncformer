import torch
from dncformer.config import CFG
from dncformer.experiments.experiments import run_e17, run_e18
from dncformer.utils.helpers import free_head_and_cache
from dncformer.train.loop import train_experiment, start_tb_run

device = "cuda" if torch.cuda.is_available() else "cpu"

def test_e17_preflight():
    # synthetic only, 6 steps to exercise schedules a bit
    CFG.train_steps = 6
    head, tok = run_e17(steps=6, mixture=(0.0, 0.34, 0.33, 0.33), hf_dataset=None, hf_max_items=0)
    assert hasattr(head, "blocks") and len(head.blocks) == 1
    free_head_and_cache()

def test_e18_preflight():
    CFG.train_steps = 6
    head, tok = run_e18(steps=6, mixture=(0.0, 0.34, 0.33, 0.33), hf_dataset=None, hf_max_items=0)
    assert hasattr(head, "blocks") and len(head.blocks) == 2
    free_head_and_cache()


def test_TBfix():
    start_tb_run("TBfix-smoke")
    CFG.train_steps = 2
    head, tok = train_experiment(steps=2, batch_size=1, log_every=1,
                                 mixture_weights=(0.0,0.34,0.33,0.33),
                                 hf_dataset=None, hf_max_items=0)

