# dncformer/config.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
import yaml

@dataclass
class DNCFormerConfig:
    # Base / model IO
    base_model_id: str = "microsoft/Phi-3-mini-4k-instruct"
    d_model: Optional[int] = None
    n_blocks: int = 2

    # Controller / enrichment
    attn_heads: int = 8
    attn_dropout: float = 0.1
    ffn_mult: float = 4.0

    # DNC memory (global defaults)
    dnc_read_heads: int = 2
    dnc_cell_size: int = 64
    dnc_nr_cells: int = 256
    gate_bias_init: float = -1.0

    # Train
    lr: float = 2e-4
    weight_decay: float = 0.01
    max_seq_len: int = 1024
    train_steps: int = 200
    warmup_steps: int = 20
    grad_clip: float = 1.0
    precision: str = "bf16"     # "bf16"|"fp16"|"fp32"
    use_torch_compile: bool = False
    device: str = "cuda"
    log_every: int = 10
    batch_size: int = 8
    seed: int = 42

    # Schedules / gating (E series)
    gate_temp: float = 1.0
    gate_reg_lambda: float = 0.0
    force_g: Optional[float] = None
    mixture_schedule: Optional[List[Tuple[Optional[int], Tuple[float,float,float,float]]]] = None
    gate_temp_schedule: Optional[List[Tuple[Optional[int], float]]] = None
    gate_reg_schedule: Optional[List[Tuple[Optional[int], float]]] = None

    # E10 experts
    mem_experts: int = 1
    expert_N: Optional[List[int]] = None
    expert_W: Optional[int] = None
    expert_R: Optional[int] = None
    expert_gate_temp: float = 1.0
    expert_diversity_lambda: float = 0.0

    # E11 per-block overrides & free bias
    per_block_cfg: Optional[List[Optional[Dict[str, Any]]]] = None
    per_block_free_bias: Optional[List[float]] = None

    # E12 fusion (off by default)
    fusion_enable: bool = False
    fusion_hidden_mult: float = 2.0
    fusion_drop: float = 0.0
    fusion_bias_queries: bool = False

    # E13 regs
    write_reg_lambda: float = 0.0
    key_overlap_lambda: float = 0.0
    key_overlap_window: int = 1
    reg_only_on_memory_batches: bool = True

def load_config_yaml(path: str) -> DNCFormerConfig:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    cfg = DNCFormerConfig()
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg

def cfg_to_dict(cfg: DNCFormerConfig) -> dict:
    return asdict(cfg)

# Global config instance (preserve legacy usage)
CFG = DNCFormerConfig()
