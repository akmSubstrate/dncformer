"""
Export the current conda env + pip packages into versioned files under ./envs/.
- Conda YAML   -> envs/dncformer-conda-env.yml
- pip freeze   -> envs/dncformer-pip-freeze.txt
- Full report  -> envs/dncformer-env-report.txt
"""
from __future__ import annotations
import os, subprocess, sys, datetime, shutil

def _run(cmd: list[str]) -> tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    outdir = os.path.join(root, "envs")
    os.makedirs(outdir, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    yml_path = os.path.join(outdir, "dncformer-conda-env.yml")
    pip_path = os.path.join(outdir, "dncformer-pip-freeze.txt")
    rpt_path = os.path.join(outdir, "dncformer-env-report.txt")

    # conda export (no builds to ease cross‑platform)
    code, out, err = _run(["conda", "env", "export", "--no-builds"])
    if code != 0:
        print("[env] conda export failed; is conda available?", err.strip(), file=sys.stderr)
    else:
        with open(yml_path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"[env] wrote {yml_path}")

    # pip freeze (even inside conda this captures pip‑installed extras)
    code, out, err = _run([sys.executable, "-m", "pip", "freeze"])
    if code != 0:
        print("[env] pip freeze failed:", err.strip(), file=sys.stderr)
    else:
        with open(pip_path, "w", encoding="utf-8") as f:
            f.write(out)
        print(f"[env] wrote {pip_path}")

    # report: python, torch, cuda etc.
    pyver = sys.version.replace("\n", " ")
    try:
        import torch
        torch_report = f"torch={torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}, devices={torch.cuda.device_count()}"
    except Exception as e:
        torch_report = f"(torch not importable: {type(e).__name__}: {e})"
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write(f"# DNCFormer environment report ({ts})\n")
        f.write(f"python: {pyver}\n")
        f.write(f"{torch_report}\n")
        f.write(f"conda_yaml: {yml_path if os.path.exists(yml_path) else '(not generated)'}\n")
        f.write(f"pip_freeze: {pip_path if os.path.exists(pip_path) else '(not generated)'}\n")
    print(f"[env] wrote {rpt_path}")

if __name__ == "__main__":
    main()
