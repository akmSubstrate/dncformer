from __future__ import annotations
import argparse, json, torch, random, numpy as np
from dncformer.config import CFG
from dncformer.utils.yaml_utils import load_yaml_cfg
from dncformer.train.runner import train_runner

def _apply_overrides_from_cli(args):
    if args.lr is not None:            CFG.lr = float(args.lr)
    if args.batch_size is not None:    CFG.batch_size = int(args.batch_size)
    if args.steps is not None:         CFG.train_steps = int(args.steps)
    if args.chunk_len is not None:     CFG.sticky_mix = int(args.chunk_len)
    if args.base_model_id is not None: CFG.base_model_id = str(args.base_model_id)
    if args.hf_dataset is not None:    CFG.hf_dataset = (None if args.hf_dataset.lower()=="none" else args.hf_dataset)
    if args.hf_max_items is not None:  CFG.hf_max_items = int(args.hf_max_items)

def main():
    p = argparse.ArgumentParser("DNCFormer trainer (0.3.0)")
    p.add_argument("-c","--config", type=str, required=True, help="Experiment YAML path")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--chunk-len", type=int, default=None)
    p.add_argument("--base-model-id", type=str, default=None)
    p.add_argument("--hf-dataset", type=str, default=None)
    p.add_argument("--hf-max-items", type=int, default=None)
    p.add_argument("--label", type=str, default=None)
    args = p.parse_args()

    cfg = load_yaml_cfg(args.config)
    _apply_overrides_from_cli(args)

    # seeding
    seed = int(getattr(CFG, "seed", 1337))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    # mixture default aligns with legacy order (hf, copy, repeat, nback)
    mixture = tuple(getattr(CFG, "mixture", (0.4,0.2,0.2,0.2)))
    steps   = int(args.steps or getattr(CFG, "train_steps", 1000))
    batch   = int(args.batch_size or getattr(CFG, "batch_size", 8))
    warmup  = int(getattr(CFG, "warmup_steps", max(10, steps//20)))
    chunk   = int(args.chunk_len or getattr(CFG, "sticky_mix", 0))
    label   = args.label or getattr(CFG, "label", None)
    hf_ds   = getattr(CFG, "hf_dataset", "tatsu-lab/alpaca")
    hf_N    = int(getattr(CFG, "hf_max_items", 5000))

    train_runner(steps=steps, batch_size=batch, mixture=mixture,
                 warmup_steps=warmup, min_lr_ratio=float(getattr(CFG,"lr_min_ratio",0.1)),
                 hf_dataset=hf_ds, hf_max_items=hf_N, chunk_len=chunk, label=label)

if __name__ == "__main__":
    main()
