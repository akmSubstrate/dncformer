# dncformer/__init__.py
from .config import CFG, DNCFormerConfig, load_config_yaml, cfg_to_dict
from .train.loop import (
    train_experiment, evaluate_haystack, build_model_and_tokenizer,
    load_base_model, lm_shift_labels
)
from .log.tb import (
    TB_AVAILABLE, tb_available, tb_analyzer_available,
    start_tb_run, TBLogger, tb  # tb is module-level global writer (optional)
)

# Common data helpers
from .data.mix import MixtureSampler, build_mixer
from .data.synthetic import make_copy_task, make_repeat_copy, make_n_back, make_haystack_batch
from .data.hf import hf_instruction_loader, make_hf_batch, format_instruction

# Experiments (optional surface)
from .experiments.e15e16 import (
    set_e11b_baseline,
    set_e15a_write_sparse_light, set_e15b_two_experts_smallW,
    set_e16a, set_e16b, run_e16_once, run_e16_sweep
)

__all__ = [
    "CFG", "DNCFormerConfig", "load_config_yaml", "cfg_to_dict",
    "train_experiment", "evaluate_haystack", "build_model_and_tokenizer",
    "load_base_model", "lm_shift_labels",
    "TB_AVAILABLE", "tb_available", "tb_analyzer_available",
    "start_tb_run", "TBLogger", "tb",
    "MixtureSampler", "build_mixer",
    "make_copy_task", "make_repeat_copy", "make_n_back", "make_haystack_batch",
    "hf_instruction_loader", "make_hf_batch", "format_instruction",
    "set_e11b_baseline",
    "set_e15a_write_sparse_light", "set_e15b_two_experts_smallW",
    "set_e16a", "set_e16b", "run_e16_once", "run_e16_sweep",
]
