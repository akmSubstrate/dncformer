# dncformer/train/experiments.py
from copy import deepcopy
from datetime import timedelta
import time, json, torch, random, numpy as np
from dncformer.config import CFG
from dncformer.train.loop import train_experiment, start_tb_run
from dncformer.utils.helpers import free_head_and_cache
from dncformer.utils.env import report_cuda
from dncformer.train.scheduler import make_chunked_mixture_schedule

def _cfg_snapshot() -> dict:
    # shallow is sufficient for scalar CFG fields
    return deepcopy({k: getattr(CFG, k) for k in dir(CFG)
                     if not k.startswith("_") and not callable(getattr(CFG, k))})

def _cfg_restore(snap: dict):
    # restore saved keys
    for k, v in snap.items():
        setattr(CFG, k, v)

def build_chunked_schedules(total_steps: int,
                            chunks: list[dict],
                            default_mix=(0.4, 0.2, 0.2, 0.2),
                            default_temp: float = 1.0):
    """
    Minimal 'chunked schedule' helper.

    chunks = [
      {"len": 0.4, "mix": (0.0,0.34,0.33,0.33), "temp": 0.9},   # 40% of steps
      {"len": 0.4, "mix": (0.2,0.26,0.27,0.27), "temp": 0.95}, # 40%
      {"len": None, "mix": (0.4,0.2,0.2,0.2), "temp": 1.0},    # remainder
    ]
    - "len" can be a float fraction (0–1) or an int absolute steps.
    - The final chunk should usually have len=None to “hold” the last setting.
    Returns (mixture_schedule, gate_temp_schedule) as lists of (until, value).
    """
    assert total_steps > 0
    mix_sched, temp_sched = [], []
    used = 0
    for i, ch in enumerate(chunks):
        L = ch.get("len", None)
        mix = tuple(ch.get("mix", default_mix))
        temp = float(ch.get("temp", default_temp))
        if L is None:
            until = None
            mix_sched.append((until, mix))
            temp_sched.append((until, temp))
            break
        # normalize len: fraction -> steps
        if isinstance(L, float):
            steps = int(round(total_steps * max(0.0, min(1.0, L))))
        else:
            steps = int(L)
        used += steps
        until = min(total_steps, used)
        mix_sched.append((until, mix))
        temp_sched.append((until, temp))
        if used >= total_steps:
            # no need to add a trailing None; last 'until' holds to the end
            break
    if used < total_steps and (len(temp_sched) == 0 or temp_sched[-1][0] is not None):
        # add a trailing 'hold' with defaults if caller forgot a None chunk
        mix_sched.append((None, default_mix))
        temp_sched.append((None, default_temp))
    return mix_sched, temp_sched

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


# ----------------------------- E17b (parallel) -------------------------------
def set_e17b_parallel_stronger():
    """
    Parallel multi-memory: 1 block with K=2 experts,
    stronger anti-collapse (A1+A2), LoRA on, no block-level dropout needed.
    """
    CFG.n_blocks = 1
    CFG.mem_experts = 2
    CFG.per_block_cfg = None

    # Anti-collapse (stronger than E17)
    CFG.router_balance_lambda = 3e-3   # A1: stronger soft balance across experts
    CFG.router_noise_std      = 0.05   # A2: tiny Gaussian router noise
    CFG.expert_dropout_p      = 0.05   # small expert-drop in training
    CFG.block_memdrop_prob    = 0.0    # only one memory block here
    CFG.cross_block_balance_lambda = 0.0

    # LoRA + other toggles
    CFG.lora_enable = True
    set_e14_curriculum(S=max(50, int(CFG.train_steps // 5)))
    CFG.mixture_chunk_steps = int(getattr(CFG, "mixture_chunk_steps", 0) or 200)

def run_e17_b(label="E17_parallel_b",
              steps=1000, seed=1337, chunk_len=200,
              order=("copy","repeat","nback"),
              hf_dataset: str | None = None, hf_max_items: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    snap = _cfg_snapshot()
    try:
        set_e17b_parallel_stronger()
        # chunked, one-hot schedule (no HF slot since hf_dataset=None by default)
        sched = make_chunked_mixture_schedule(
            total_steps=steps, chunk_len=chunk_len, order=order, include_hf=bool(hf_dataset)
        )
        start_tb_run(f"{label}-s{seed}")
        free_head_and_cache(); report_cuda(f"before {label}-s{seed}")

        head, tok = train_experiment(
            steps=steps, warmup_steps=max(10, steps//20),
            mixture_weights=(0.0, 1.0, 0.0, 0.0) if hf_dataset else (1.0, 0.0, 0.0),  # ignored after first step; schedule will take over
            mixture_schedule=sched,
            hf_dataset=hf_dataset, hf_max_items=hf_max_items,
        )
        report_cuda(f"after  {label}-s{seed}"); free_head_and_cache()
        return head, tok
    finally:
        _cfg_restore(snap)

# ---------------------------- E18b (sequential) ------------------------------
def set_e18b_sequential_stronger():
    """
    Sequential: 2 blocks with deliberate role asymmetry:
      - block 0: many addresses, shallow cells (short-horizon, high capacity)
      - block 1: fewer addresses, deeper cells (longer-horizon, lower capacity)
    Stronger B1+B2 collapse guards.
    """
    CFG.n_blocks = 2
    CFG.mem_experts = 1
    CFG.per_block_cfg = [
        {"N": 192, "W": 32, "R": 1, "gate_temp": 1.0, "free_bias": +0.20},  # early, shallow/many
        {"N":  64, "W": 64, "R": 1, "gate_temp": 0.9, "free_bias": -0.20},  # late, deep/few
    ]

    # Anti-collapse for sequential:
    CFG.router_balance_lambda = 0.0     # no parallel experts to balance
    CFG.router_noise_std      = 0.02
    CFG.expert_dropout_p      = 0.00
    CFG.block_memdrop_prob    = 0.03    # B2: occasional memory-block dropout
    CFG.cross_block_balance_lambda = 2e-3  # B1: encourage both blocks to be used

    CFG.lora_enable = True
    set_e14_curriculum(S=max(50, int(CFG.train_steps // 5)))
    CFG.mixture_chunk_steps = int(getattr(CFG, "mixture_chunk_steps", 0) or 200)

def run_e18_b(label="E18_sequential_b",
              steps=1000, seed=1337, chunk_len=200,
              order=("copy","repeat","nback"),
              hf_dataset: str | None = None, hf_max_items: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    snap = _cfg_snapshot()
    try:
        set_e18b_sequential_stronger()
        sched = make_chunked_mixture_schedule(
            total_steps=steps, chunk_len=chunk_len, order=order, include_hf=bool(hf_dataset)
        )
        start_tb_run(f"{label}-s{seed}")
        free_head_and_cache(); report_cuda(f"before {label}-s{seed}")

        head, tok = train_experiment(
            steps=steps, warmup_steps=max(10, steps//20),
            mixture_weights=(0.0, 1.0, 0.0, 0.0) if hf_dataset else (1.0, 0.0, 0.0),
            mixture_schedule=sched,
            hf_dataset=hf_dataset, hf_max_items=hf_max_items,
        )
        report_cuda(f"after  {label}-s{seed}"); free_head_and_cache()
        return head, tok
    finally:
        _cfg_restore(snap)

# Optional convenience sweep (single-GPU sequential; multi-GPU note below)
def run_e17_18b_sweep(steps=3000, seeds=(1337,2027,4242,1561,6969,3030), mixture=(0.4,0.2,0.2,0.2),
                      hf_dataset="tatsu-lab/alpaca", hf_max_items=8000):
    results = {}
    for s in seeds:
        results[(f"E17b", s)] = run_e17_b(steps=steps, seed=s, mixture=mixture,
                                         hf_dataset=hf_dataset, hf_max_items=hf_max_items)
    for s in seeds:
        results[(f"E18b", s)] = run_e18_b(steps=steps, seed=s, mixture=mixture,
                                         hf_dataset=hf_dataset, hf_max_items=hf_max_items)
    return results

def _parse_seeds_arg(s: str) -> tuple[int, ...]:
    s = (s or "").strip()
    if not s:
        return (1337,)
        out = []
    for tok in s.replace(" ", "").split(","):
        if not tok:
            continue
        try:
            out.append(int(tok))
        except Exception:
            pass
        return tuple(out) or (1337,)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="DNCFormer E17/E18 experiment runner")

    ap.add_argument("--mode", choices=["e17", "e18", "sweep"], default="sweep",
                    help = "Run a single E17, a single E18, or the E17/E18-b sweep (default).")
    ap.add_argument("--steps", type=int, default=1000, help="Total training steps.")
    ap.add_argument("--seeds", type=str, default="1337,2027,4242",
                    help = "Comma-separated seed list for sweep mode.")
    ap.add_argument("--hf-dataset", type=str, default=None,
                    help = 'HF dataset id; use "none" to disable and run synthetic-only.')
    ap.add_argument("--hf-max-items", type=int, default=0,
                    help = "Cap on HF samples; 0 means small/light or synthetic-only when dataset is none.")
    ap.add_argument("--label-prefix", type=str, default="E17E18b",
                    help = "Prefix added to TB run names during sweep.")
    ap.add_argument("--chunk-len", type=int, default=50,
                    help = "StickyMixtureSampler chunk length for task segments.")
    args = ap.parse_args()

  # Normalize HF dataset sentinel
    hf_ds = args.hf_dataset

    if isinstance(hf_ds, str) and hf_ds.lower() in ("none", ""):
        hf_ds = None


    if args.mode == "e17":
    # Single E17 (parallel) run—use the first seed parsed
        seed0 = _parse_seeds_arg(args.seeds)[0]
        run_e17(
            steps = args.steps,
            seed = seed0,
            hf_dataset = hf_ds,
            hf_max_items = args.hf_max_items,
        )
    elif args.mode == "e18":
    # Single E18 (sequential) run—use the first seed parsed
        seed0 = _parse_seeds_arg(args.seeds)[0]
        run_e18(
            steps = args.steps,
            seed = seed0,
            hf_dataset = hf_ds,
            hf_max_items = args.hf_max_items,
        )
    else:
    # Sweep: E17 and E18 across multiple seeds with chunked schedule
        run_e17_18b_sweep(
            steps = args.steps,
            seeds = _parse_seeds_arg(args.seeds),
            hf_dataset = hf_ds,
            hf_max_items = args.hf_max_items,
            label_prefix = args.label_prefix,
            chunk_len = args.chunk_len,
        )

# # Conservative, VRAM‑friendly runtime knobs for 24GB cards with LoRA:
    # CFG.precision     = "bf16"            # keeps matmul fast & stable on recent GPUs
    # CFG.max_seq_len   = 256               # HF windows kept short; synthetics already ≤128
    # CFG.batch_size    = max(4, getattr(CFG, "batch_size", 8))  # adjust if you see OOM
    # CFG.lr            = getattr(CFG, "lr", 2e-4)
    # CFG.weight_decay  = getattr(CFG, "weight_decay", 0.01)
    # # If your LoRA patch exposes these, they’re safe defaults:
    # if not hasattr(CFG, "lora_rank"):   CFG.lora_rank = 16
    # if not hasattr(CFG, "lora_alpha"):  CFG.lora_alpha = 32
    # if not hasattr(CFG, "lora_dropout"): CFG.lora_dropout = 0.05
    #
    # # Memory‑first curriculum in both experiments; HF resumes after S≈steps/5
    # mixture      = (0.0, 0.34, 0.33, 0.33)   # (hf, copy, repeat, nback)
    # hf_dataset   = "tatsu-lab/alpaca"
    # hf_max_items = 8000
    # steps        = 1000
    # #seeds        = (1337, 2027, 4242)
    # seeds        = (1337,)
    #
    # sweep_start = time.time()
    # for s in seeds:
    #     start = time.time()
    #     run_e17(label="E17_parallel",
    #             steps=steps, seed=s,
    #             mixture=mixture,
    #             hf_dataset=hf_dataset, hf_max_items=hf_max_items)
    #     print(f"[ Run time: {timedelta(seconds = time.time()-start)} ]")
    #     start = time.time()
    #     # tiny spacer avoids TB dir collisions on fast filesystems
    #     time.sleep(1.0)
    #     run_e18(label="E18_sequential",
    #             steps=steps, seed=s,
    #             mixture=mixture,
    #             hf_dataset=hf_dataset, hf_max_items=hf_max_items)
    #     print(f"[ Run time: {timedelta(seconds = time.time()-start)} ]")
    # print(f"[ Sweep time: {timedelta(seconds= time.time()-sweep_start)} for {int(len(seeds)*2)} runs total")