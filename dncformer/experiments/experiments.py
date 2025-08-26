# dncformer/train/experiments.py
import time, json, torch, random, numpy as np
from ..config import CFG
from ..train.loop import train_experiment, start_tb_run
from ..utils.helpers import free_head_and_cache
from ..utils.env import report_cuda

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

def run_e17(label="E17_parallel", steps=1000, seed=1337, mixture=(0.4,0.2,0.2,0.2)):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    set_e17_parallel()
    start_tb_run(f"{label}-s{seed}")
    free_head_and_cache(); report_cuda(f"before {label}-s{seed}")
    head, tok = train_experiment(steps=steps, warmup_steps=max(10, steps//20),
                                 mixture_weights=mixture,
                                 mixture_schedule=CFG.mixture_schedule,
                                 gate_temp_schedule=CFG.gate_temp_schedule)
    report_cuda(f"after  {label}-s{seed}"); free_head_and_cache()
    return head, tok

def run_e18(label="E18_sequential", steps=1000, seed=1337, mixture=(0.4,0.2,0.2,0.2)):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    set_e18_sequential()
    start_tb_run(f"{label}-s{seed}")
    free_head_and_cache(); report_cuda(f"before {label}-s{seed}")
    head, tok = train_experiment(steps=steps, warmup_steps=max(10, steps//20),
                                 mixture_weights=mixture,
                                 mixture_schedule=CFG.mixture_schedule,
                                 gate_temp_schedule=CFG.gate_temp_schedule)
    report_cuda(f"after  {label}-s{seed}"); free_head_and_cache()
    return head, tok
