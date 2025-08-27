import torch
from dncformer.train.loop import train_experiment
from dncformer.config import CFG

def test_train_experiment_ministeps():
    # keep this tiny; just checks the loop runs end-to-end
    head, tok = train_experiment(steps=2, batch_size=max(1, CFG.batch_size//2), log_every=1,
                                 mixture_weights=(0.0, 0.34, 0.33, 0.33),  # memory-only warm start is fine here
                                 hf_dataset=None, hf_max_items=0)
    assert head is not None and tok is not None

if __name__=="__main__":
    test_train_experiment_ministeps()
