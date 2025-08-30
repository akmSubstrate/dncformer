import os, torch
from dncformer.config import CFG
from dncformer.train.runner import train_runner

def test_runner_synthetic_only_tiny():
    CFG.seed = 1337
    CFG.base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    CFG.lora_enable = False  # keep CI light; enable in real runs
    head, tok = train_runner(
        steps=4, batch_size=2, mixture=(0.0, 0.34, 0.33, 0.33),
        warmup_steps=2, hf_dataset=None, hf_max_items=0, chunk_len=2, log_every=2,
        label="smoke_runner_synth"
    )
    assert hasattr(head, "blocks") and len(head.blocks) >= 1
