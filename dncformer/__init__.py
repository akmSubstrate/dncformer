from .config import CFG
from .utils.yaml_utils import load_yaml_cfg

# Training (runner-based, replaces train_experiment)
from .train.runner import train_runner, build_model_and_tokenizer

# Logging
from .log.tb import start_tb_run, TBLogger, tb

# Data entry points (compose samplers from YAML; also export a few built-in generators)
from .data.registry import build_sampler_from_cfg
from .data.synthetic import make_copy_task, make_repeat_copy, make_n_back

__all__ = [
    "CFG",
    "load_yaml_cfg",
    "train_runner",
    "build_model_and_tokenizer",
    "start_tb_run",
    "TBLogger",
    "tb",
    "build_sampler_from_cfg",
    "make_copy_task",
    "make_repeat_copy",
    "make_n_back",
]

# Optional: loud deprecation helpers (uncomment if you want explicit runtime errors on old imports)
# def train_experiment(*args, **kwargs):
#     raise RuntimeError(
#         "train_experiment has been removed in v0.3.x. "
#         "Use `dncformer.train.runner.train_runner` and a YAML config instead."
#     )
