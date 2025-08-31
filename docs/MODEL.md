# MODEL — Parallel Enrichment Blocks (PEB)

DNCFormer inserts **Parallel Enrichment Blocks** in front of the base LM’s output head. Each block computes:

1. **Vanilla** path (LayerNorm → MHA → residual → FFN → residual).
2. **Memory experts**: one or more `DNCformerBlock`s, each with a compact **DNC-style** read/write interface and internal temporal links.
3. A **gate** mixing `[vanilla, expert_1, …, expert_K]` per token.

A block produces `(B,T,D)` features and a per-token **memory usage** gate (for logging).

---

## Block layout

X (B,T,D)
│
├─ VanillaTransformerBlock ───────► v_t (B,T,D)
│
├─ DNCformerBlock #1 ──────► m1_t (B,T,D)
│
├─ DNCformerBlock #K ──────► mK_t (B,T,D)
│
└─ Concat [v_t, m1_t, …, mK_t] ──► LN ─► Linear((K+1)D→K+1) ─► softmax
│
└─ gated sum over {v_t, m_i_t} → y_t (B,T,D)

- **Memory**: each DNCformerBlock carries `(M, u, L, p, rw, r)` state and exposes minimal statistics like `write_gate_mean` for regularizers/logging. See in-code comments for usage update, allocation, temporal links, and read modes.  
- **Gate**: a low-dim linear + softmax over `[vanilla + experts]`. We track global means, frac>0.5, entropy; per-task versions of these; and (if K>0) expert path distributions and entropies. (See TB logging in the runner for exact tag names.)【:contentReference[oaicite:19]{index=19}】

---

## Multiple experts & topologies

- **Parallel experts (K>1)**  
  Gate mixes among `vanilla` and `K` memory paths. Optional anti-collapse losses:
  - **A1:** soft expert balance (spread expert usage).  
  - **A2:** tiny Gaussian noise on router logits; **expert_dropout_p**.  
- **Sequential blocks**  
  Stacked PEBs (e.g., block 0 = small/short horizon; block 1 = large/long horizon). Optional **cross-block usage balance** pushes both blocks to contribute, not just one.

You can mix these: e.g., 2 blocks, first with 2 experts (K=2), second with K=1 but larger N/W.

---

## Read-hint fusion

A light **read-hint fusion** adds an MLP-projected read vector (pooled across heads) back to the vanilla path as a residual. Helpful when memory should bias attention features directly; keep off by default if VRAM is tight.
This feature is currently very experimental and may lead to OOM errors - **use with caution**

---

## Metrics that matter

- **Global gates:** `gates/block_i_mean`, `gates/block_i_frac>0.5`, `gates/block_i_entropy`
- **Per-task gates/loss:** `gates_by_task/block_i_*(/<task>)`, `loss_by_task/<task>`
- **Experts:** `experts/block_i/pi_mean_j`, `experts/block_i/pi_entropy`
- **Regularization signals:** `reg/block_i/write_gate_mean`
- **Memory visuals (optional):** usage, L, read weights snapshots.

All of these are written from the 0.3.x runner via TB.【:contentReference[oaicite:20]{index=20}】

---
