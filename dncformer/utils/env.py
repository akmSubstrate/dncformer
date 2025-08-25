# dncformer/utils/env.py
from __future__ import annotations
import os, contextlib, random
import numpy as np
import torch

def choose_amp_dtype(precision: str) -> torch.dtype:
    if precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if precision == "fp16" and torch.cuda.is_available():
        return torch.float16
    return torch.float32

def sdpa_ctx():
    try:
        from torch.backends.cuda import sdp_kernel
        return sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True)
    except Exception:
        return contextlib.nullcontext()

def set_determinism(seed: int = 42, deterministic: bool = True, cudnn_benchmark: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True, warn_only=True)

def report_cuda(tag=""):
    if not torch.cuda.is_available():
        print("[cuda] not available"); return
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated(); reserv = torch.cuda.memory_reserved()
    print(f"[{tag}] alloc={alloc/1e9:.2f} GB | reserved={reserv/1e9:.2f} GB | free={free/1e9:.2f} GB | total={total/1e9:.2f} GB")
