# dncformer/train/experiments.py
from datetime import timedelta
import time, json, torch, random, numpy as np
from dncformer.config import CFG
from dncformer.train.loop import train_experiment, start_tb_run
from dncformer.utils.helpers import free_head_and_cache
from dncformer.utils.env import report_cuda
from copy import deepcopy

def _cfg_snapshot() -> dict:
    # shallow is sufficient for scalar CFG fields
    return deepcopy({k: getattr(CFG, k) for k in dir(CFG)
                     if not k.startswith("_") and not callable(getattr(CFG, k))})

def _cfg_restore(snap: dict):
    # restore keys we saved
    for k, v in snap.items():
        setattr(CFG, k, v)

def set_e14_curriculum(S=200):
    CFG.mixture_schedule = [(S, (0.0, 0.34, 0.33, 0.33)), (None, (0.4, 0.2, 0.2, 0.2))]
    CFG.gate_temp_schedule = [(S, 0.9), (None, 1.0)]

def set_e17_parallel():
    """Parallel multi-memory: 1 block with K=2 experts, anti-collapse A1+A2, LoRA, warm-start."""
    CFG.n_blocks = 1
    CFG.mem_experts = 2
    CFG.per_block_cfg = None
    # anti-collapse
    CFG.router_balance_lambda = 1e-3   # A1 (soft)
    CFG.router_noise_std = 0.03        # A2
    CFG.expert_dropout_p = 0.02        # tiny expert-drop
    CFG.block_memdrop_prob = 0.0       # head-level dropout not needed for single block
    CFG.cross_block_balance_lambda = 0.0
    # LoRA + curriculum
    CFG.lora_enable = True
    set_e14_curriculum(S= max(50, int(CFG.train_steps//5)))

def set_e18_sequential():
    """Sequential multi-memory: 2 blocks, (many/shallow) -> (few/deep), anti-collapse B1+B2, LoRA, warm-start."""
    CFG.n_blocks = 2
    CFG.mem_experts = 1
    CFG.per_block_cfg = [
        {"N": 192, "W": 32, "R": 1, "gate_temp": 1.0, "free_bias": +0.20},  # block 0 = many addresses, shallow
        {"N":  64, "W": 64, "R": 1, "gate_temp": 0.9, "free_bias": -0.20},  # block 1 = fewer addresses, deeper cells
    ]
    # anti-collapse
    CFG.router_balance_lambda = 0.0    # no parallel experts here
    CFG.router_noise_std = 0.02
    CFG.expert_dropout_p = 0.0
    CFG.block_memdrop_prob = 0.02      # B2: rare memory block dropout
    CFG.cross_block_balance_lambda = 1e-3  # B1: gentle equalization across the two blocks
    # LoRA + curriculum
    CFG.lora_enable = True
    set_e14_curriculum(S= max(50, int(CFG.train_steps//5)))

def run_e17(label="E17_parallel",
            steps=1000, seed=1337,
            mixture=(0.4,0.2,0.2,0.2),
            hf_dataset: str | None = "tatsu-lab/alpaca",
            hf_max_items: int = 5000):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    snap = _cfg_snapshot()
    try:
        set_e17_parallel()
        start_tb_run(f"{label}-s{seed}")
        free_head_and_cache(); report_cuda(f"before {label}-s{seed}")

        head, tok = train_experiment(
            steps=steps, warmup_steps=max(10, steps//20),
            mixture_weights=mixture,
            mixture_schedule=CFG.mixture_schedule,
            gate_temp_schedule=CFG.gate_temp_schedule,
            hf_dataset=hf_dataset, hf_max_items=hf_max_items,
        )
        report_cuda(f"after  {label}-s{seed}"); free_head_and_cache()
        return head, tok
    finally:
        _cfg_restore(snap)


def run_e18(label="E18_sequential",
            steps=1000, seed=1337,
            mixture=(0.4,0.2,0.2,0.2),
            hf_dataset: str | None = "tatsu-lab/alpaca",
            hf_max_items: int = 5000):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    snap = _cfg_snapshot()
    try:
        set_e18_sequential()
        start_tb_run(f"{label}-s{seed}")
        free_head_and_cache(); report_cuda(f"before {label}-s{seed}")

        head, tok = train_experiment(
            steps=steps, warmup_steps=max(10, steps//20),
            mixture_weights=mixture,
            mixture_schedule=CFG.mixture_schedule,
            gate_temp_schedule=CFG.gate_temp_schedule,
            hf_dataset=hf_dataset, hf_max_items=hf_max_items,
        )
        report_cuda(f"after  {label}-s{seed}"); free_head_and_cache()
        return head, tok
    finally:
        _cfg_restore(snap)



if __name__ == "__main__":  # change to "__main__" if you’ll run as a script
    # Conservative, VRAM‑friendly runtime knobs for 24GB cards with LoRA:
    CFG.precision     = "bf16"            # keeps matmul fast & stable on recent GPUs
    CFG.max_seq_len   = 256               # HF windows kept short; synthetics already ≤128
    CFG.batch_size    = max(4, getattr(CFG, "batch_size", 8))  # adjust if you see OOM
    CFG.lr            = getattr(CFG, "lr", 2e-4)
    CFG.weight_decay  = getattr(CFG, "weight_decay", 0.01)
    # If your LoRA patch exposes these, they’re safe defaults:
    if not hasattr(CFG, "lora_rank"):   CFG.lora_rank = 16
    if not hasattr(CFG, "lora_alpha"):  CFG.lora_alpha = 32
    if not hasattr(CFG, "lora_dropout"): CFG.lora_dropout = 0.05

    # Memory‑first curriculum in both experiments; HF resumes after S≈steps/5
    mixture      = (0.0, 0.34, 0.33, 0.33)   # (hf, copy, repeat, nback)
    hf_dataset   = "tatsu-lab/alpaca"
    hf_max_items = 8000
    #steps        = 1000
    steps        = 100
    #seeds        = (1337, 2027, 4242)
    seeds        = (1337,2027)

    for s in seeds:
        start = time.time()
        run_e17(label="E17_parallel",
                steps=steps, seed=s,
                mixture=mixture,
                hf_dataset=hf_dataset, hf_max_items=hf_max_items)
        print(f"[ Run time: {timedelta(seconds = time.time()-start)} ]")
        start = time.time()
        # tiny spacer avoids TB dir collisions on fast filesystems
        time.sleep(1.0)
        run_e18(label="E18_sequential",
                steps=steps, seed=s,
                mixture=mixture,
                hf_dataset=hf_dataset, hf_max_items=hf_max_items)
        print(f"[ run time: {timedelta(seconds = time.time()-start)} ]")