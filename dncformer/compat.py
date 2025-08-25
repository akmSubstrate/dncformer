# dncformer/compat.py
from .config import CFG, DNCFormerConfig, load_config_yaml, cfg_to_dict
from .train.loop import (
    train_experiment, evaluate_haystack, build_model_and_tokenizer,
    load_base_model, lm_shift_labels
)
from .log.tb import TB_AVAILABLE, tb_available, tb_analyzer_available, start_tb_run, TBLogger, tb
from .data.mix import MixtureSampler, build_mixer as _build_mixer

# Legacy alias (some notebooks used _build_mixer)
def build_mixer(*a, **k): return _build_mixer(*a, **k)
