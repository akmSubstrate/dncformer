# dncformer/log/tb.py
from __future__ import annotations
import os, re, time, contextlib, json
from typing import Optional
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa
    TB_AVAILABLE = True
except Exception:
    SummaryWriter = None  # type: ignore
    TB_AVAILABLE = False

try:
    import tensorboard as _tb  # noqa
    TENSORBOARD_VERSION = getattr(_tb, "__version__", "unknown")
except Exception:
    TENSORBOARD_VERSION = "unavailable"

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa
    from tensorboard.backend.event_processing import event_accumulator as ea_mod  # noqa
except Exception:
    EventAccumulator, ea_mod = None, None

def tb_available() -> bool:
    return bool(TB_AVAILABLE and SummaryWriter is not None)

def tb_analyzer_available() -> bool:
    return bool(EventAccumulator is not None and ea_mod is not None)

class TBLogger:
    def __init__(self, logdir: Optional[str] = None, run_name: Optional[str] = None):
        self.enabled = tb_available()
        self.writer = None
        if not self.enabled: return
        logdir = logdir or "./runs"
        run_name = run_name or time.strftime("dncformer-%Y%m%d-%H%M%S")
        self.path = os.path.join(logdir, run_name)
        os.makedirs(self.path, exist_ok=True)
        self.writer = SummaryWriter(self.path)

    def add_scalar(self, *a, **kw):
        if self.writer: self.writer.add_scalar(*a, **kw)
    def add_text(self, *a, **kw):
        if self.writer: self.writer.add_text(*a, **kw)
    def add_image(self, *a, **kw):
        if self.writer: self.writer.add_image(*a, **kw)
    def add_histogram(self, *a, **kw):
        if self.writer: self.writer.add_histogram(*a, **kw)
    def flush(self):
        if self.writer: self.writer.flush()
    def close(self):
        if self.writer: self.writer.close()

# module‑level logger (optional, mirrors notebook usage)
tb: Optional[TBLogger] = None

def start_tb_run(label: str | None = None, logdir: str = "./runs") -> bool:
    global tb
    if not tb_available():
        print("TensorBoard not available; skipping start_tb_run.")
        return False
    try:
        if isinstance(tb, TBLogger) and tb.writer:
            tb.flush(); tb.close()
    except Exception:
        pass
    ts = time.strftime("dncformer-%Y%m%d-%H%M%S")
    run_name = f"{ts}-{re.sub(r'[^A-Za-z0-9_.-]+','_',str(label))}" if label else ts
    tb = TBLogger(logdir=logdir, run_name=run_name)
    tb.add_text("run/label", str(label or "unlabeled"), 0)
    print("TB run started:", getattr(tb, "path", None))
    return True
