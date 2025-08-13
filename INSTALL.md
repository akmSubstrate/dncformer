# DNCFormer Environment Setup

This repo was developed on Windows with an RTX 3090 (24GB). We recommend using **mamba/conda** on Windows.

## 1) Create env (CUDA 12.x toolchain)
```bash
mamba create -n dncformer python=3.10 -y
mamba activate dncformer
# Core GPU stack (choose CUDA build matching your driver)
mamba install -c conda-forge pytorch=2.5.* torchvision=0.20.* torchaudio=2.5.* pytorch-cuda=12.6 -y
# Research libs
mamba install -c conda-forge numpy scipy sentencepiece tqdm ipywidgets jupyterlab -y
# Hugging Face stack
mamba install -c conda-forge transformers>=4.44 tokenizers>=0.19 datasets>=2.20 accelerate>=0.33 -y
# TensorBoard
mamba install -c conda-forge tensorboard -y
```

If you prefer `pip` for HF:
```bash
pip install -U transformers tokenizers datasets accelerate tensorboard
```

> We intentionally **do not** install `flash-attn`; the notebook uses **PyTorch SDPA**.

## 2) Export your exact environment (lockfiles)

After everything works in Jupyter:
```bash
conda env export --from-history > environment.yml     # minimal spec (portable)
conda env export > environment.lock.yml               # full, exact versions
pip list --format=freeze > requirements-pip.txt       # pip extras if any
```

Commit those files so others can reproduce.

## 3) Jupyter tips (Windows + PyCharm)
- Set the project interpreter and Jupyter server to the **dncformer** env.
- If ipywidgets warnings appear: `mamba install -c conda-forge ipywidgets`
- If you see `DynamicCache` errors with Transformers, upgrade: `mamba install -c conda-forge "transformers>=4.44" "accelerate>=0.33"`

## 4) Running the notebook
- Open `dncformer_notebook.ipynb`
- Run top-to-bottom. Training logs stream to **TensorBoard**:
```bash
tensorboard --logdir ./runs
```