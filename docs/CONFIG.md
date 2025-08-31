# CONFIG — knobs & context (v0.3.1)

This summarizes the most important config fields. The canonical defaults live in `dncformer/config.py`

---

## Model & training

- `base_model_id`: HF model id (e.g., TinyLlama…)
- `precision`: `bf16 | fp16 | fp32` (autocast handled automatically)
- `batch_size`, `train_steps`, `warmup_steps`, `lr`, `weight_decay`
- `grad_clip`: gradient clipping (default 1.0)
- `log_every`: print/TB cadence
- `device`: usually `"cuda"`

## Head / PEB

- `n_blocks`: number of blocks in the DNCFormer head
- `attn_heads`, `attn_dropout`, `ffn_mult`: vanilla sub-block sizes
- `d_model`: if `None`, inferred from base hidden size

## Memory defaults (per block unless overridden)

- `dnc_read_heads` (`R`), `dnc_cell_size` (`W`), `dnc_nr_cells` (`N`)
- `gate_bias_init`: initial bias favoring vanilla at start
- `gate_fp32`: compute router in float32 for stability (if enabled)

## Per-block overrides

- `per_block_cfg`: list of dicts per block:
  - `N`, `W`, `R`, `gate_temp`, `free_bias`
- `per_block_free_bias`: per-block write-free bias list if set globally elsewhere

## Experts & fusion

- `mem_experts` (`K`), `expert_N` (list), `expert_W`, `expert_R`
- `expert_gate_temp`, `expert_diversity_lambda`
- `fusion_enable`, `fusion_hidden_mult`, `fusion_drop`, `fusion_bias_queries` (optional fusion path)

## Anti-collapse & regs

- **Parallel experts**: `router_balance_lambda`, `router_noise_std`, `expert_dropout_p`
- **Sequential blocks**: `cross_block_balance_lambda`, `block_memdrop_prob`
- **General**: `gate_temp`, `gate_reg_lambda`, `write_reg_lambda`, `reg_only_on_memory_batches`

## Data (YAML)

Under `data:`:
- `sticky_mix`: steps to hold the same task before re-sampling (chunk length)
- `tasks`: list of `{name, type, weight, …}` entries
  - `type: hf`: `dataset`, `max_items`, optional `text_key`
  - `type: synth`: `kind` or `name` ∈ `{copy, repeat, nback}`, with `params`

The sampler is built from YAML by `dncformer/data/registry.py`. No legacy fallback in 0.3.x; missing/empty tasks cause a loud error.【:contentReference[oaicite:22]{index=22}】

---

## Logger

- TB run setup/writer lifecycle: `dncformer/log/tb.py` (`start_tb_run`, `TBLogger`).【:contentReference[oaicite:23]{index=23}】

---
