# tests/test_sticky_chunks.py
import torch, random
from dncformer.config import CFG
from dncformer.train.loop import build_model_and_tokenizer
from dncformer.data.mix import build_mixer

def test_sticky_chunks_obvious():
    random.seed(0); torch.manual_seed(0)
    CFG.mixture_chunk_steps = 25  # small chunk for the test
    tok, _ = build_model_and_tokenizer()

    # synthetic-only, equal weights among 3 tasks
    mixer = build_mixer(tok, (0.0, 1.0, 1.0, 1.0), hf_dataset=None, hf_max_items=0, sticky_chunk_steps=25)

    names = []
    for step in range(1, 101):
        _ = mixer(batch=2)
        names.append(mixer.last_name)
    # assert there are at most 4 chunks in 100 steps (25-step chunks)
    boundaries = [i for i in range(1, 100) if names[i] != names[i-1]]
    assert len(boundaries) <= 4, f"Too many switches: {len(boundaries)} at {boundaries}"

