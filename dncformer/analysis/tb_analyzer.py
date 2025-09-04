from __future__ import annotations

import os, re, glob, json, argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Prefer the project’s TB shim (gives EventAccumulator & version-safe helpers)
try:
    # If your repo path is dncformer/log/tb.py this will succeed:
    from dncformer.log.tb import (
        tb_available as _tb_available,
        tb_analyzer_available as _tb_ana_available,
        EventAccumulator,
        ea_mod,
    )
    _TB_OK = bool(_tb_available() and _tb_ana_available())
except Exception:
    # Fallback: import directly from tensorboard
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from tensorboard.backend.event_processing import event_accumulator as ea_mod
        _TB_OK = True
    except Exception:
        EventAccumulator, ea_mod = None, None
        _TB_OK = False


# version-agnostic size guidance
def _size_guidance_version_safe() -> Dict:
    """
    Build size_guidance dict usable across TB versions (module constants or lowercase fallbacks).
    """
    sg = {}
    keys = ["SCALARS", "HISTOGRAMS", "IMAGES", "COMPRESSED_HISTOGRAMS", "AUDIO", "TENSORS"]
    for k in keys:
        v = getattr(ea_mod, k, None) if ea_mod is not None else None
        if v is not None:
            sg[v] = 0
        else:
            sg[k.lower()] = 0
    return sg


# low-level readers
def _load_scalars_from_event_file(ev_path: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load all scalar series from a single event file into: tag -> list[(step, value)].
    """
    acc = EventAccumulator(ev_path, size_guidance=_size_guidance_version_safe())
    acc.Reload()
    tags = acc.Tags().get("scalars", []) or []
    out: Dict[str, List[Tuple[int, float]]] = {}
    for tag in tags:
        vals = acc.Scalars(tag)
        out[tag] = [(int(ev.step), float(ev.value)) for ev in vals]
    return out


def _merge_scalar_dicts(list_of_scalar_dicts: List[Dict[str, List[Tuple[int, float]]]]
                        ) -> Dict[str, List[Tuple[int, float]]]:
    """
    Merge multiple files for the same run. Keep last value per step, then sort by step.
    """
    merged = defaultdict(dict)  # tag -> {step: value}
    for scal in list_of_scalar_dicts:
        for tag, series in scal.items():
            d = merged[tag]
            for step, val in series:
                d[step] = val
    out = {}
    for tag, d in merged.items():
        steps_sorted = sorted(d.keys())
        out[tag] = [(s, d[s]) for s in steps_sorted]
    return out


def _detect_tasks_and_blocks(scalars: Dict[str, List[Tuple[int, float]]]) -> Tuple[List[str], List[int]]:
    """
    Detect task names and block ids from present tags.
    Falls back to {hf,copy,repeat,nback} and blocks [0,1].
    """
    tasks, blocks = set(), set()

    # tasks from loss_by_task/<task>
    for tag in scalars.keys():
        if tag.startswith("loss_by_task/"):
            tasks.add(tag.split("/", 1)[1])

    # blocks from gates_by_task/block_<b>_* or gates/block_<b>_mean
    for tag in scalars.keys():
        m = re.match(r"gates_by_task/block_(\d+)_", tag)
        if m:
            blocks.add(int(m.group(1)))
        m2 = re.match(r"gates/block_(\d+)_mean$", tag)
        if m2:
            blocks.add(int(m2.group(1)))

    if not tasks:
        tasks = {"hf", "copy", "repeat", "nback"}
    if not blocks:
        blocks = {0, 1}
    return sorted(tasks), sorted(blocks)


#  small reducers
def _s_last(vals: List[Tuple[int, float]], k: Optional[int] = None) -> float:
    if not vals: return float("nan")
    arr = np.array([v for _, v in vals], dtype=float)
    if k is None or k >= len(arr): return float(arr[-1])
    return float(np.nanmean(arr[-k:]))

def _s_first(vals: List[Tuple[int, float]], k: Optional[int] = None) -> float:
    if not vals: return float("nan")
    arr = np.array([v for _, v in vals], dtype=float)
    if k is None or k >= len(arr): return float(arr[0])
    return float(np.nanmean(arr[:k]))

def _s_mean(vals: List[Tuple[int, float]]) -> float:
    if not vals: return float("nan")
    return float(np.nanmean([v for _, v in vals]))

def _s_count(vals: List[Tuple[int, float]]) -> int:
    return len(vals) if vals else 0


# label inference
def _decode_text_tensor(tensor_proto) -> Optional[str]:
    """
    Decode TB text payload from TensorEvent.tensor_proto. Works across TB builds.
    """
    try:
        # tensorboard.util.tensor_util.make_ndarray may exist on some versions:
        from tensorboard.util import tensor_util as _tb_tensor_util  # type: ignore
        arr = _tb_tensor_util.make_ndarray(tensor_proto)
    except Exception:
        try:
            # legacy location (rare)
            from tensorboard.compat.proto.tensor_util import MakeNdarray as _tb_make_ndarray  # type: ignore
            arr = _tb_make_ndarray(tensor_proto)
        except Exception:
            return None

    try:
        val = arr.item() if getattr(arr, "size", 0) == 1 else arr
        if isinstance(val, bytes):
            return val.decode("utf-8", "replace")
        if isinstance(val, str):
            return val
        if hasattr(val, "dtype") and str(val.dtype).startswith("|S"):
            return val.tobytes().decode("utf-8", "replace")
    except Exception:
        pass
    return None


def _infer_label_from_text_tags(run_dir: Path) -> Optional[str]:
    """
    Prefer 'run/label' then JSON in 'run/meta' with a 'label' field.
    """
    ev_files = sorted(glob.glob(str(run_dir / "events.out.tfevents.*")))
    for ev in reversed(ev_files):
        try:
            acc = EventAccumulator(ev, size_guidance=_size_guidance_version_safe())
            acc.Reload()
            text_tags = acc.Tags().get("tensors", []) or []
            # 1) explicit label
            if "run/label" in text_tags:
                tens = acc.Tensors("run/label")
                for e in reversed(tens):
                    txt = _decode_text_tensor(e.tensor_proto)
                    if txt: return txt.strip()
            # 2) meta JSON with label
            if "run/meta" in text_tags:
                tens = acc.Tensors("run/meta")
                for e in reversed(tens):
                    txt = _decode_text_tensor(e.tensor_proto)
                    if txt:
                        try:
                            obj = json.loads(txt)
                            lab = obj.get("label", None)
                            if isinstance(lab, str) and lab.strip():
                                return lab.strip()
                        except Exception:
                            pass
        except Exception:
            continue
    return None


def _infer_label_from_dirname(run_dir: Path) -> str:
    """
    If dir is 'dncformer-YYYYMMDD-HHMMSS-LABEL', return LABEL; else the dirname.
    """
    name = run_dir.name
    m = re.match(r".*-\d{8}-\d{6}-(.+)$", name)
    return m.group(1) if m else name


# per-run summarizer
def summarize_run(run_dir: Path) -> Tuple[Dict, Dict[str, List[Tuple[int, float]]]]:
    """
    Build a single row summary for a run directory and return (summary, merged_scalars).
    """
    ev_files = sorted(glob.glob(str(run_dir / "events.out.tfevents.*")))
    scalars_all = [_load_scalars_from_event_file(p) for p in ev_files]
    scal = _merge_scalar_dicts(scalars_all)

    # Labels / meta
    label = _infer_label_from_text_tags(run_dir) or _infer_label_from_dirname(run_dir)
    steps_logged = max([s for s, _ in scal.get("train/loss", [])], default=np.nan)
    lr_last = _s_last(scal.get("train/lr", []), k=1)

    # Globals
    loss0  = _s_first(scal.get("train/loss", []), k=5)
    lossT  = _s_last  (scal.get("train/loss", []), k=10)
    delta  = (loss0 - lossT) if not any(map(np.isnan, [loss0, lossT])) else np.nan

    # Gates (global) — your loop writes these tags each log interval. :contentReference[oaicite:3]{index=3}
    tasks, blocks = _detect_tasks_and_blocks(scal)
    g_means = {b: _s_last(scal.get(f"gates/block_{b}_mean", []), k=10) for b in blocks}
    g_ents  = {b: _s_last(scal.get(f"gates/block_{b}_entropy", []), k=10) for b in blocks}
    # Average frac>0.5 over tasks (if available)
    g_frac_avg = {}
    for b in blocks:
        vals = []
        for t in tasks:
            tag = f"gates_by_task/block_{b}_frac>0.5/{t}"
            if tag in scal:
                vals.append(_s_last(scal[tag], k=10))
        g_frac_avg[b] = float(np.nanmean(vals)) if vals else np.nan

    # Task losses — mean of last quarter of points
    task_last = {}
    for t in tasks:
        ts = scal.get(f"loss_by_task/{t}", [])
        if ts:
            k = max(1, len(ts) // 4)
            task_last[t] = _s_last(ts, k=k)
        else:
            task_last[t] = float("nan")

    # Expert diagnostics (if present) — written by your loop for E17/E18. :contentReference[oaicite:4]{index=4}
    # We summarize last entropy per block, and store last per-path mean as JSON-ish str.
    experts_entropy = {}
    experts_pi_last = {}
    for b in blocks:
        ent_tag = f"experts/block_{b}/pi_entropy"
        ents = scal.get(ent_tag, [])
        if ents: experts_entropy[b] = _s_last(ents, k=5)
        # gather pi_mean_* for this block if present
        j = 0
        means = []
        while True:
            tag = f"experts/block_{b}/pi_mean_{j}"
            if tag not in scal: break
            means.append(_s_last(scal[tag], k=5))
            j += 1
        if means:
            experts_pi_last[b] = means  # vanilla + K experts

    # Quartiles (block 0), averaged across tasks if present
    q_means = {}
    for qi in (1, 2, 3, 4):
        vals = []
        for t in tasks:
            tagt = f"gates/block0_q{qi}_mean/{t}"
            if tagt in scal:
                vals.append(_s_last(scal[tagt], k=10))
        q_means[qi] = float(np.nanmean(vals)) if vals else np.nan

    # Assemble row
    row = {
        "label": label,
        "run_dir": str(run_dir),
        "steps_logged": steps_logged,
        "loss_start~5": loss0,
        "loss_end~10": lossT,
        "loss_delta": delta,
        "lr_last": lr_last,
    }
    for b in blocks:
        row[f"g_mean_b{b}"] = g_means.get(b, float("nan"))
        row[f"g_entropy_b{b}"] = g_ents.get(b, float("nan"))
        row[f"g_frac>0.5_avg_b{b}"] = g_frac_avg.get(b, float("nan"))
        if b in experts_entropy: row[f"experts_pi_entropy_b{b}"] = experts_entropy[b]
        if b in experts_pi_last: row[f"experts_pi_mean_b{b}"] = json.dumps(experts_pi_last[b])

    for t in tasks: row[f"loss_{t}_last"] = task_last[t]
    for qi in (1, 2, 3, 4): row[f"g_b0_Q{qi}_mean"] = q_means[qi]

    return row, scal


def _collect_run_dirs(root: Path, glob_pat: Optional[str]) -> List[Path]:
    """
    Discover run directories under root; optionally filter by a glob on dirname.
    """
    cand = []
    for p in root.glob("**/*"):
        if p.is_dir():
            if list(p.glob("events.out.tfevents.*")):
                cand.append(p)
    if glob_pat:
        cand = [p for p in cand if p.match(glob_pat)]
    # Sort by mtime for nicer order
    cand.sort(key=lambda p: p.stat().st_mtime)
    return cand


def analyze_runs(log_root: str, out_dir: str, run_glob: Optional[str] = None, verbose: bool = True
                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry: analyze all runs under log_root and write CSVs to out_dir.
    Returns (df_runs, df_task_ts).
    """
    if not _TB_OK or EventAccumulator is None:
        raise RuntimeError("TensorBoard EventAccumulator is unavailable; install `tensorboard` first.")

    root = Path(log_root).resolve()
    outp = Path(out_dir).resolve()
    outp.mkdir(parents=True, exist_ok=True)

    run_dirs = _collect_run_dirs(root, run_glob)
    if verbose:
        print(f"[tb_analyzer] discovered {len(run_dirs)} run dirs under {root}")

    summaries = []
    all_scalars_by_run = {}

    for rd in run_dirs:
        try:
            s, scal = summarize_run(rd)
            s["tasks_detected"] = ",".join(sorted({t for t in _detect_tasks_and_blocks(scal)[0]}))
            s["blocks_detected"] = ",".join(map(str, _detect_tasks_and_blocks(scal)[1]))
            summaries.append(s)
            all_scalars_by_run[rd.name] = (rd, scal)
        except Exception as e:
            summaries.append({"label": _infer_label_from_dirname(rd), "run_dir": str(rd), "error": str(e)})

    df_runs = pd.DataFrame(summaries)

    # Reorder columns (most relevant first)
    front = [
        "label", "run_dir", "steps_logged",
        "loss_start~5", "loss_end~10", "loss_delta", "lr_last",
        # gates
        "g_mean_b0", "g_entropy_b0", "g_frac>0.5_avg_b0",
        "g_mean_b1", "g_entropy_b1", "g_frac>0.5_avg_b1",
        # experts
        "experts_pi_entropy_b0", "experts_pi_mean_b0",
        "experts_pi_entropy_b1", "experts_pi_mean_b1",
        # tasks
        "loss_hf_last", "loss_copy_last", "loss_repeat_last", "loss_nback_last",
        # quartiles
        "g_b0_Q1_mean", "g_b0_Q2_mean", "g_b0_Q3_mean", "g_b0_Q4_mean",
        "tasks_detected", "blocks_detected", "error",
    ]
    cols = [c for c in front if c in df_runs.columns] + [c for c in df_runs.columns if c not in front]
    df_runs = df_runs[cols]

    # per‑task granular time series
    rows_ts: List[Dict] = []
    for run_name, (rd, scal) in all_scalars_by_run.items():
        label = _infer_label_from_text_tags(rd) or _infer_label_from_dirname(rd)
        tasks, blocks = _detect_tasks_and_blocks(scal)

        # step->lr map (optional)
        lr_map = {int(s): float(v) for s, v in (scal.get("train/lr", []) or [])}

        for t in tasks:
            loss_series = scal.get(f"loss_by_task/{t}", [])
            if not loss_series: continue

            gmean_dict = {
                b: {int(s): float(v) for s, v in scal.get(f"gates_by_task/block_{b}_mean/{t}", [])}
                for b in blocks
            }
            gfrac_dict = {
                b: {int(s): float(v) for s, v in scal.get(f"gates_by_task/block_{b}_frac>0.5/{t}", [])}
                for b in blocks
            }
            qdict = {
                qi: {int(s): float(v) for s, v in scal.get(f"gates/block0_q{qi}_mean/{t}", [])}
                for qi in (1, 2, 3, 4)
            }

            for step, loss_val in loss_series:
                step = int(step); loss_val = float(loss_val)
                for b in blocks:
                    row = {
                        "label": label,
                        "run_dir": str(rd),
                        "task": t,
                        "block": b,
                        "step": step,
                        "loss": loss_val,
                        "g_mean": gmean_dict[b].get(step, float("nan")),
                        "g_frac>0.5": gfrac_dict[b].get(step, float("nan")),
                        "lr": lr_map.get(step, float("nan")),
                        "g_b0_Q1": qdict[1].get(step, float("nan")) if b == 0 else float("nan"),
                        "g_b0_Q2": qdict[2].get(step, float("nan")) if b == 0 else float("nan"),
                        "g_b0_Q3": qdict[3].get(step, float("nan")) if b == 0 else float("nan"),
                        "g_b0_Q4": qdict[4].get(step, float("nan")) if b == 0 else float("nan"),
                    }
                    rows_ts.append(row)

    df_task_ts = pd.DataFrame(rows_ts)
    if not df_task_ts.empty:
        df_task_ts = df_task_ts.sort_values(["label", "task", "step", "block"], ignore_index=True)

    # Write CSVs
    out_runs = outp / "run_level_summary.csv"
    out_tasks = outp / "per_task_metrics.csv"
    df_runs.to_csv(out_runs, index=False)
    df_task_ts.to_csv(out_tasks, index=False)

    print(f"[tb_analyzer] wrote: {out_runs}")
    print(f"[tb_analyzer] wrote: {out_tasks}")
    return df_runs, df_task_ts


# CLI
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze DNCFormer TensorBoard runs → CSV.")
    p.add_argument("--runs", default="./runs", help="Root directory containing TB run subdirs.")
    p.add_argument("--out",  default="./analysis", help="Output directory for CSVs.")
    p.add_argument("--glob", default=None, help="Optional glob on run dir names (e.g. 'dncformer-*-E*').")
    p.add_argument("--quiet", action="store_true", help="Reduce console output.")
    return p


def main():
    args = _build_argparser().parse_args()
    if not _TB_OK or EventAccumulator is None:
        raise SystemExit("TensorBoard not available; install with `pip install tensorboard`.")
    analyze_runs(args.runs, args.out, run_glob=args.glob, verbose=not args.quiet)


if __name__ == "__main__":
    main()
