# dncformer/config.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Sequence
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
    gate_fp32: bool = True # toggle to compute router LN/Linear in float 32 for stability

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
    lr_min_ratio: float = 0.10
    lr_cawr_T0: int = 200
    lr_cawr_Tmult: int = 2

    scheduler_type: str = "cosine"  # ["cosine","one_cycle","plateau"]
    min_lr_ratio: float = 0.1  # for cosine/one-cycle
    plateau_factor: float = 0.5
    plateau_patience: int = 200  # steps of no improvement
    early_stop_patience: int = 800  # steps
    grad_accum_steps: int = 1  # >1 enables accumulation

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


    # LoRA settings
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2", "wq", "wk", "wv", "wo", "gate_proj", "up_proj",
        "down_proj"
    )
    lora_last_n_layers: int = 2  # restrict to last N base model transformer blocks

    # Multi-block configuration
    block_roles: Optional[List[str]] = None

    # Gate anti-collapse functionality (default inactive)
    router_noise_std: float = 0.0  # A2: σ for Gaussian noise on memory logits (training only)
    router_balance_lambda: float = 0.0  # A1: weight for per-block soft balance across memory experts
    expert_dropout_p: float = 0.0  # tiny per-token expert-dropout within a block (memory experts only)
    block_memdrop_prob: float = 0.0  # B2: chance to drop a block’s memory path (training only, head-level)
    cross_block_balance_lambda: float = 0.0  # B1: weight for cross-block memory-usage balance


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
