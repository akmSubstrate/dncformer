# CLI — Training DNCFormer (v0.3.x)

The CLI expects a **YAML config** describing model/training and **data tasks**. It launches the new runner (not the legacy 0.2.x loop).

## Quick start

Run a command like: `python -m cli.train -c dncformer/configs/multi_hf_synth.yaml --steps 1000 --label my_run`

This will:
1. Load base_model_id + freeze weights (+ optional LoRA),
2. Build the sampler from data.tasks (HF + synthetic),
3. Train for steps with warmup+cosine LR, logging to ./runs/<timestamp>-<label>.
> The runner writes scalars to TB via the module logger set up in dncformer/log/tb.py (start_tb_run).
See also the main training loop in dncformer/train/runner.py.

## CLI flags

-c, --config      Path to YAML (required)
--steps           Override train_steps in YAML
--batch-size      Override batch_size
--lr              Override lr
--chunk-len       Override data.sticky_mix (hold task this many steps)
--base-model-id   Override base_model_id (HF id)
--hf-dataset      Legacy HF shortcut (prefer data.tasks)
--hf-max-items    Legacy HF shortcut (prefer data.tasks[*].max_items)
--label           TB run label suffix

## YAML essentials
seed: 1337
label: example
base_model_id: TinyLlama/TinyLlama-1.1B-Chat-v1.0
precision: bf16

lora_enable: true
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

batch_size: 8
train_steps: 1000
warmup_steps: 50
lr: 2.0e-4
weight_decay: 0.01

data:
  sticky_mix: 10
  tasks:
    - { name: alpaca, type: hf, dataset: tatsu-lab/alpaca, weight: 0.25, max_items: 4000 }
    - { name: tinystories, type: hf, dataset: roneneldan/TinyStories, weight: 0.25, max_items: 4000 }
    - { name: copy, type: synth, weight: 0.20, params: { T: 128, vocab: 200 } }
    - { name: nback, type: synth, weight: 0.30, params: { T: 128, n: 5, vocab: 100 } }

## Task types

- HF: dataset (HF id), max_items, optional text_key.
- Synthetic: kind (defaults to name) in {copy, repeat, nback}, with params.

## Logging (TensorBoard)

- Scalars: train/loss, train/lr, loss_by_task/<task>
- Gates: gates/block_i_*, gates_by_task/block_i_*/*
- Experts: experts/block_i/pi_mean_j, experts/block_i/pi_entropy