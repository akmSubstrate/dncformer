# DNCFormer: Differentiable Neural Computer‑Transformer Hybrid

DNCFormer is an experimental research project that extends a frozen base large language model (LLM) with Parallel 
Enrichment Blocks (PEBs) – lightweight modules that blend a vanilla transformer sub‑block with differentiable 
neural computer (DNC) memory experts. Each PEB reintroduces an external memory interface without the heavy cost of
scaling the base LLM. Here follows a summary of the core concepts, a brief theoretical overview of the architecture,
and links to more in‑depth documentation.

## Why DNCFormer?

Transformers struggle with **algorithmic** and **long-dependency** tasks without brute-force scaling of context 
length/parameters, or external tools for context manipulation (RAG systems, auxillary models, etc.).
Classic memory-augmented models (NTM/DNC) offer *explicit* read/write semantics, but have historically proved too
unstable to scale well, or be applied to NL data.

DNCFormer’s PEBs fold **external memory** into a more modern DL stack: each block computes a vanilla path *and* 
memory-expert path(s), then learns a **gate** that chooses/blends their outputs. The framework was designed so users can
stack blocks in series, attach multiple memory experts to the same block in parallel, and tailor memory capacity within
blocks for behavioral tuning

## Theoretical background

Traditional recurrent neural networks (RNNs) can learn short sequential dependencies but struggle with long‑range
reasoning. To bridge this gap, researchers proposed memory‑augmented neural networks such as the 
differentiable neural computer [(Neural Turing Machines, 2014)](https://arxiv.org/abs/1410.5401). A DNC consists
of a controller (typically an RNN) connected to an external memory matrix. The controller can read and write using
content‑based addressing, freeing it from the limits of its hidden state. DNCs have been shown to solve algorithmic
tasks such as copying, repeating, and sequence completion, where the controller must store and retrieve information 
across very long time horizons.

The DNCFormer adapts this idea to modern language models by inserting an external memory path in parallel with a
'vanilla' transformer backbone. At each token position, the block computes both the vanilla path
(attention + feedforward) and one or more memory expert paths. The outputs are then combined with a learned gate, 
allowing the model to learn when to route information into external memory and when to rely on local context. 
Multiple memory experts can be attached either in parallel (sharing a controller but storing independent memories) 
or sequentially across layers. The design is intended encourages the network to learn algorithmic behaviours 
and make use of long‑horizon storage for tasks that require those capabilities, while preserving a base model’s 
language understanding.

## Documentation overview

- **Launch training with CLI** [`CLI.md`](docs/CLI.md): How to train DNCFormer. Describes the
command‑line interface, required YAML configuration, and key
training flags. Shows examples of synthetic‑only and mixed
HF‑dataset runs.

- **Add custom tasks/datasets** [`HOWTO_data.md`](docs/HOWTO_data.md): Adding custom tasks via
the data registry. Explains how to register new synthetic
generators or hook additional HuggingFace datasets into the sampler.
Provides examples for algorithmic probes like balanced parentheses
and long addition.

- **Model/block details** [`MODEL.md`](docs/MODEL.md): A textual diagram of the Parallel
Enrichment Block. Explains how gating mixes vanilla and memory
paths, how multiple experts work in parallel or in sequential layers,
and what metrics are recorded for interpretability.

- **Config surface** [`CONFIG.md`](docs/CONFIG.md): Reference to all configurable fields. Covers model‑related
(memory size, number of experts, gating bias), LoRA settings, training hyperparameters, 
anti‑collapse penalties and data sampler options.

## Core components / Useful entry points

- [`cli/train.py`](cli/train.py): Entry point for training. Reads YAML
config, builds model and data pipeline, runs the training loop and
logs to TensorBoard

- [`dncformer/train/runner.py`](dncformer/train/runner.py): Training loop. Uses
the registry to build the task sampler, handles AMP + gradient
scaling, warmup + LR scheduling, and logs gate metrics

- [`dncformer/data/registry.py`](dncformer/data/registry.py): Builds a MixtureSampler from YAML specified config.
Supports arbitrary numbers of HF datasets and synthetic tasks. Provides optional
'sticky' chunked sampling, so different task modalities persist for multiple steps

- [`dncformer/model/head.py`](dncformer/model/head.py): Defines the DNCFormer head and Parallel Enrichment Blocks. See 
MODEL.md for more detailed overview

- [`dncformer/log/tb.py`](dncformer/log/tb.py): Simple TensorBoard logger and run initialiser

## Getting started

1. Create a YAML config describing your run, including the base model,
training hyperparameters and a `data.tasks` section. See the
included configs in [`dncformer/configs/`](dncformer/configs) for examples

2. Launch training via the CLI: `python -m cli.train -c your_config.yaml --label my_run`

3. The training script writes logs to ./runs/<timestamp>-<label>. Use
tensorboard --logdir ./runs to visualise training loss, gate
statistics, and per‑task performance

4. Review the other docs to learn how to customise tasks, experiment with multiple memory experts and adjust
regularisation knobs

DNCFormer is a research project. Expect to tune regularisation parameters (e.g., gate balance, expert diversity)
and design tasks carefully to encourage the network to use its external memory

Contributions and feedback welcome as the project evolves!