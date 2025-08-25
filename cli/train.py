# cli/train.py
from __future__ import annotations
import argparse, json
from dncformer.config import CFG, load_config_yaml, DNCFormerConfig, cfg_to_dict
from dncformer.log.tb import start_tb_run, TB_AVAILABLE, tb
from dncformer.train.loop import train_experiment

def main():
    ap = argparse.ArgumentParser("DNCFormer training")
    ap.add_argument("--config", type=str, default=None, help="YAML config file")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--label", type=str, default=None)
    ap.add_argument("--mix", type=float, nargs=4, default=(0.4,0.2,0.2,0.2))
    args = ap.parse_args()

    if args.config:
        cfg = load_config_yaml(args.config)
        # copy loaded into global CFG
        for k, v in cfg.__dict__.items(): setattr(CFG, k, v)

    if args.label and TB_AVAILABLE:
        start_tb_run(args.label)
        if tb and tb.writer:
            tb.add_text("run/cli", json.dumps({"label": args.label, "steps": args.steps, "mix": list(args.mix)}, indent=2), 0)

    train_experiment(steps=args.steps, mixture_weights=tuple(args.mix))
    print("done")

if __name__ == "__main__":
    main()
