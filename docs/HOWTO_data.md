# HOWTO: Add custom tasks (v0.3.x)

This shows how to register **synthetic** generators and wire **HF** datasets into the YAML-driven sampler.

The registry assembles a `MixtureSampler` (optionally `StickyMixtureSampler`) from `data.tasks` in your YAML and returns a callable `sampler(batch)` that yields `LongTensor[B, T]`. See `dncformer/data/registry.py`.【:contentReference[oaicite:13]{index=13}】

---

## 1) Add a synthetic task

Implement a generator that returns `[B, T]` token IDs and then expose it in the registry. Example: `data/synthetic.py` already provides `copy`, `repeat`, `nback`.【:contentReference[oaicite:14]{index=14}】

### Example: “zigzag” pattern (toy)

```python
# dncformer/data/synthetic_custom.py
import torch, random

def make_zigzag(batch: int, T: int, vocab: int = 100):
    x = torch.randint(1, vocab, (batch, T), dtype=torch.long)
    # enforce alternating up/down steps every 4
    for i in range(0, T-1, 4):
        if i+1 < T: x[:, i+1] = x[:, i] + 1
        if i+2 < T: x[:, i+2] = x[:, i+1] - 2
        if i+3 < T: x[:, i+3] = x[:, i+2] + 3
    return x
```

### Expose in registry (either import and route via a `newtask_zigzag` branch, or register in a small map):

```python
# dncformer/data/registry.py  (inside build_sampler_from_cfg loop, type=='synth'):
elif kind in ("zigzag",):
    T = int(params.get("T", min(mx, 128))); vocab = int(params.get("vocab", 100))
    from .synthetic_custom import make_zigzag
    def gen_zig(b, _T=T, _V=vocab): return make_zigzag(b, T=_T, vocab=_V)
    gens.append(gen_zig); weights.append(weight); names.append(name)
```

### YAML description:
```yaml
data:
  sticky_mix: 20
  tasks:
    - { name: zig, type: synth, weight: 0.5, params: { T: 128, vocab: 200, kind: zigzag } }
    - { name: nback, type: synth, weight: 0.5, params: { T: 128, n: 5, vocab: 100 } }
```

## 2) Add a new LLM dataset from HugginFace
The registry can build two HF task flavors:
- **Alpaca-style** instruction/output via hf_instruction_loader (pairs) → make_hf_batch
- **Generic corpora** (e.g., TinyStories): tokenize and cache up to max_items, then sample windows on demand. See 
  _build_hf_gen for details.

## 3) 'Sticky' task sampling
Due to the nature of DNCs, mixing distinct memory/algorithmic task types alongside LLM-style data is essentail for
activating memory gating, especially in initial runs. To prevent thrashing, bundling different task types into larger
chunks is advisable.

Chunking is configured via:
- `data.sticky_mix` in YAML training config file or
- `--chunk-len` CLI arg override

Under the hood, we wrap the data blend in the `StickyMixtureSampler` so task modalities/datasets persist for N steps 
before re-sampling and switching to a new task modality (see `dncformer/data/mix.py` for more detail)

## 4) Sanity test: verify the sampler picks up new task
on startup, runner will print a task summary to the console, e.g.
`[data] sampler tasks=['alpaca','tinystories','copy','nback'] weights=[0.25,0.25,0.2,0.3] sticky_mix=10
`
If the new task name doesn't appear, check your YAML file to ensure the new modality is specified correctly

