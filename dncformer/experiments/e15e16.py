# dncformer/experiments/e15e16.py
from __future__ import annotations
import time, json, random
import numpy as np
import torch
from ..config import CFG
from ..train.loop import train_experiment, evaluate_haystack
from ..log.tb import start_tb_run, TB_AVAILABLE, tb
from ..utils.helpers import free_head_and_cache

def set_e11b_baseline():
    CFG.per_block_cfg = [
        {"N": 64,  "W": 32, "R": 1, "gate_temp": 1.0, "free_bias": +0.30},
        {"N": 128, "W": 64, "R": 2, "gate_temp": 0.9, "free_bias": -0.20},
    ]
    CFG.mem_experts = 1
    CFG.fusion_enable = False
    CFG.gate_reg_lambda = getattr(CFG, "gate_reg_lambda", 2e-4)
    CFG.write_reg_lambda = 0.0
    CFG.key_overlap_lambda = getattr(CFG, "key_overlap_lambda", 0.0)
    for k in ("expert_N", "expert_W", "expert_gate_temp"):
        if hasattr(CFG, k): delattr(CFG, k)
    for k in ("force_g","mixture_schedule","gate_temp_schedule","gate_reg_schedule"):
        if hasattr(CFG, k): delattr(CFG, k)

def set_e15a_write_sparse_light(lambda_write: float = 5e-5):
    set_e11b_baseline(); CFG.write_reg_lambda = float(lambda_write)

def set_e15b_two_experts_smallW(K: int = 2, expert_W: int = 32, expert_gate_temp: float = 1.0):
    set_e11b_baseline()
    CFG.mem_experts = int(K); CFG.expert_W = int(expert_W); CFG.expert_gate_temp = float(expert_gate_temp)
    CFG.write_reg_lambda = 0.0

def set_e16a(steps:int, K:int=2, expert_N_each:int=64, expert_W:int=None, expert_R:int=1,
             gate_temp:float=0.9, diversity_lambda:float=1e-3, warm_stage:int=None):
    W_default = getattr(CFG, "dnc_cell_size", 64)
    expert_W  = int(expert_W or W_default)
    warm_S    = int(warm_stage if warm_stage is not None else min(steps // 4, 250))
    CFG.mem_experts = int(K); CFG.expert_N = [int(expert_N_each)]*K; CFG.expert_W = int(expert_W)
    CFG.expert_R = int(expert_R); CFG.expert_gate_temp = float(gate_temp)
    CFG.expert_diversity_lambda = float(diversity_lambda)
    CFG.per_block_cfg = [
        {"N":128, "W":32, "R":1, "gate_temp":1.0, "free_bias": +0.30},
        {"N": 64, "W":64, "R":1, "gate_temp":0.8, "free_bias": -0.20},
    ]
    CFG.mixture_schedule   = [(warm_S, (0.0, 0.34, 0.33, 0.33)), (None, (0.4, 0.2, 0.2, 0.2))]
    CFG.gate_temp_schedule = [(warm_S, 0.8), (None, 1.0)]
    CFG.fusion_enable = False

def set_e16b(steps:int, K:int=2, expert_N_each:int=64, expert_W:int=None, expert_R:int=1,
             gate_temp:float=0.8, diversity_lambda:float=1e-3, warm_stage:int=None):
    W_default = getattr(CFG, "dnc_cell_size", 64)
    expert_W  = int(expert_W or W_default)
    warm_S    = int(warm_stage if warm_stage is not None else min(steps // 4, 250))
    CFG.mem_experts = int(K); CFG.expert_N = [int(expert_N_each)]*K; CFG.expert_W = int(expert_W)
    CFG.expert_R = int(expert_R); CFG.expert_gate_temp = float(gate_temp)
    CFG.expert_diversity_lambda = float(diversity_lambda)
    CFG.per_block_cfg = [
        {"N": 64, "W":64, "R":1, "gate_temp":1.0, "free_bias": -0.20},
        {"N":128, "W":32, "R":1, "gate_temp":0.9, "free_bias": +0.30},
    ]
    CFG.mixture_schedule   = [(warm_S, (0.0, 0.34, 0.33, 0.33)), (None, (0.4, 0.2, 0.2, 0.2))]
    CFG.gate_temp_schedule = [(warm_S, 0.8), (None, 1.0)]
    CFG.fusion_enable = False

def run_e16_once(label:str, steps:int, mode:str="a", seed:int=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if mode.lower().startswith("a"): set_e16a(steps=steps)
    else: set_e16b(steps=steps)

    free_head_and_cache()
    time.sleep(1.2)
    start_tb_run(f"{label}-s{seed}")
    if TB_AVAILABLE and tb and tb.writer:
        tb.add_text("run/config/E16", json.dumps({
            "label": label, "mode": mode, "steps": steps, "seed": seed,
            "mem_experts": getattr(CFG, "mem_experts", 1),
            "per_block_cfg": getattr(CFG, "per_block_cfg", None),
            "mixture_schedule": getattr(CFG, "mixture_schedule", None),
            "gate_temp_schedule": getattr(CFG, "gate_temp_schedule", None),
            "diversity_lambda": getattr(CFG, "expert_diversity_lambda", 0.0),
        }, indent=2), 0)

    head, tok = train_experiment(
        steps=steps, warmup_steps=max(10, steps//20), mixture_weights=(0.4,0.2,0.2,0.2),
        mixture_schedule=getattr(CFG, "mixture_schedule", None),
        gate_temp_schedule=getattr(CFG, "gate_temp_schedule", None),
        gate_reg_schedule=getattr(CFG, "gate_reg_schedule", None),
        viz_memory_after=False,
    )
    free_head_and_cache()
    return head, tok

def run_e16_sweep(steps:int=1000, seeds=(1337, 2027, 4242)):
    results = {}
    for s in seeds:
        results[("E16a", s)] = run_e16_once("E16a", steps, "a", s)
    for s in seeds:
        results[("E16b", s)] = run_e16_once("E16b", steps, "b", s)
    return results
