from __future__ import annotations
from typing import Callable, List, Tuple, Optional, Iterable
import random, torch
from datasets import load_dataset

def _pad_or_trim(ids, T, pad_id):
    n = ids.numel()
    if n == T: return ids
    if n > T:  return ids[:T]
    return torch.cat([ids, torch.full((T-n,), pad_id, dtype=torch.long)], dim=0)

def _batchize(tok, texts, B, max_len, pad_id):
    outs=[]
    for _ in range(B):
        s = random.choice(texts)
        ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        outs.append(_pad_or_trim(ids, max_len, pad_id).unsqueeze(0))
    return torch.cat(outs, dim=0)

def _pack_to_len(tok, text: str, max_len: int, pad_id: int) -> torch.Tensor:
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
    n = ids.numel()
    if n >= max_len:
        return ids[:max_len]
    pad = torch.full((max_len - n,), int(pad_id), dtype=torch.long)
    return torch.cat([ids, pad], dim=0)

def make_clutrr_gen(tok, max_len: int, pad_id: int, split="train", max_items: int = 20000) -> Callable[[int], torch.Tensor]:
    """
    CLUTRR: story with relations and a query about relation between two entities.
    """
    # Common mirrors: "clutrr", "facebook/clutrr"
    ds = None; last_err=None
    for name in ("clutrr","facebook/clutrr"):
        try:
            ds = load_dataset(name, split=split, streaming=True)
            break
        except Exception as e:
            last_err = e
    if ds is None:
        raise RuntimeError(f"[clutrr] load failed: {last_err}")

    buf=[]
    for ex in ds:
        story = ex.get("story") or ex.get("context") or ""
        q = ex.get("query") or ex.get("question") or ""
        if isinstance(story, list): story = " ".join(story)
        if not story or not q: continue
        prompt = f"### Story:\n{story.strip()}\n### Query:\n{q.strip()}\n### Answer:\n"
        buf.append(prompt)
        if len(buf) >= int(max_items): break
    if not buf: raise RuntimeError("[clutrr] 0 samples found")
    def gen(B: int) -> torch.Tensor:
        return _batchize(tok, buf, B, max_len, pad_id)
    return gen


"""
    WARNING: the generators below are unstable/bugged - avoid for now
"""
# LOGIQA: logic MCQ
def make_logiqa_gen(tok, max_len: int, pad_id: int, split: str = "train",
                    max_items: int = 20000) -> Callable[[int], torch.Tensor]:
    """
    Loads LogiQA (logic RC MCQ: passage, question, 4 options, 1 label).
    We format a simple SFT prompt and teach the model to emit the correct letter.
    """
    ds = None
    last_err: Optional[Exception] = None

    # Try a few common HF entries. If your fork uses a different repo, add it here.
    candidates = [
        # (dataset_id, config)
        ("logiqa", None),
        ("logiqa", "logiqa2"),  # some mirrors publish LogiQA 2.0 under a config
        ("json", None),         # final fallback requires user-provided data_files via registry
    ]
    for ds_id, conf in candidates:
        try:
            if ds_id == "json":
                raise RuntimeError("LogiQA JSON path not configured in registry.")
            if conf:
                ds = load_dataset(ds_id, conf, split=split, streaming=True, trust_remote_code=True)
            else:
                ds = load_dataset(ds_id, split=split, streaming=True, trust_remote_code=True)
            break
        except Exception as e:
            last_err = e
            ds = None

    if ds is None:
        raise RuntimeError(f"[LogiQA] load failed: {last_err}")

    # Normalize a small iterator of examples up to max_items
    def _iter_rows(stream: Iterable):
        count = 0
        for ex in stream:
            # Common field names seen across mirrors/papers
            passage = ex.get("passage") or ex.get("context") or ""
            question = ex.get("question") or ex.get("query") or ""
            options = ex.get("options") or ex.get("choices") or []
            label   = ex.get("label") or ex.get("answer") or ex.get("gold")  # could be idx or letter
            if not (passage and question and options and label is not None and len(options) >= 4):
                continue
            # normalize label to letter A/B/C/D
            if isinstance(label, int):
                gold_letter = "ABCD"[label]
            else:
                s = str(label).strip().upper()
                gold_letter = s[0] if s and s[0] in "ABCD" else "A"

            prompt = (
                "### Passage:\n" + passage.strip() + "\n\n"
                "### Question:\n" + question.strip() + "\n\n"
                "### Options:\n"
                "(A) " + str(options[0]) + "\n"
                "(B) " + str(options[1]) + "\n"
                "(C) " + str(options[2]) + "\n"
                "(D) " + str(options[3]) + "\n\n"
                "### Answer:\n"
            )
            text = prompt + gold_letter
            yield text
            count += 1
            if count >= int(max_items):
                break

    cache: list[torch.Tensor] = []
    for t in _iter_rows(ds):
        cache.append(_pack_to_len(tok, t, max_len, pad_id))
        if len(cache) >= max_items:
            break
    if not cache:
        raise RuntimeError("[LogiQA] 0 usable samples after normalization.")

    def gen(b: int) -> torch.Tensor:
        out = [random.choice(cache).unsqueeze(0) for _ in range(b)]
        return torch.cat(out, dim=0)

    return gen

# WIKIHOP: multi-hop QA
def make_wikihop_gen(tok, max_len: int, pad_id: int, split: str = "train",
                     max_items: int = 20000, filtered: bool = True) -> Callable[[int], torch.Tensor]:
    """
    Loads WikiHop (QAngaroo): query, candidates, answer, supports (multi-doc).
    Format: join supports with separators, list candidates, teach model to emit the answer string.
    """
    ds = None
    last_err: Optional[Exception] = None

    # Try a few common HF surfaces / configs
    # Some hubs expose 'qangaroo' builder with config 'wikihop' or 'wikihop_original'
    candidates = [
        ("qangaroo", "wikihop"),
        ("qangaroo", "wikihop_original"),
        ("qangaroo/wikihop", None),  # direct path
    ]
    for ds_id, conf in candidates:
        try:
            if conf:
                ds = load_dataset(ds_id, conf, split=split, streaming=True, trust_remote_code=True)
            else:
                ds = load_dataset(ds_id, split=split, streaming=True, trust_remote_code=True)
            break
        except Exception as e:
            last_err = e
            ds = None

    if ds is None:
        # Provide an actionable error; no silent fallback
        raise RuntimeError(
            f"[WikiHop] load failed: {last_err}\n"
            "If this mirror isn't available in your region, try another HF mirror or "
            "download the JSON from the QAngaroo site and expose it via a custom dataset script."
        )

    def _iter_rows(stream: Iterable):
        count = 0
        for ex in stream:
            q        = ex.get("query") or ex.get("question") or ""
            cands    = ex.get("candidates") or []
            answer   = ex.get("answer") or ex.get("gold") or ""
            supports = ex.get("supports") or ex.get("documents") or []
            if not (q and cands and answer and supports):
                continue
            # Join multiple docs with separators; keep it compact to fit max_len
            docs = "\n\n".join([str(s) for s in supports[:10]])  # truncate extreme cases
            cand_str = " | ".join([str(c) for c in cands[:10]])
            prompt = (
                "### Multi-hop supports:\n" + docs + "\n\n"
                "### Query:\n" + str(q).strip() + "\n\n"
                "### Candidates:\n" + cand_str + "\n\n"
                "### Answer:\n"
            )
            text = prompt + str(answer).strip()
            yield text
            count += 1
            if count >= int(max_items):
                break

    cache: list[torch.Tensor] = []
    for t in _iter_rows(ds):
        cache.append(_pack_to_len(tok, t, max_len, pad_id))
        if len(cache) >= max_items:
            break
    if not cache:
        raise RuntimeError("[WikiHop] 0 usable samples after normalization.")

    def gen(b: int) -> torch.Tensor:
        out = [random.choice(cache).unsqueeze(0) for _ in range(b)]
        return torch.cat(out, dim=0)

    return gen

def _fmt_mc_prompt(question: str, options: List[str], labels: List[str], answer_key: str) -> str:
    """
    Standard, LM-friendly multiple-choice prompt.
    """
    lines = [f"Question: {question}", "Choices:"]
    for lab, opt in zip(labels, options):
        lines.append(f"{lab}) {opt}")
    lines.append("Answer:")
    # For LM training we let the model learn to emit the label after 'Answer:'.
    # (Our standard LM loss will supervise the next tokens.)
    return "\n".join(lines) + " "

def make_qasc_gen(tok, max_len: int, pad_id: int, max_items: int = 2000, split: str = "train") -> Callable[[int], torch.Tensor]:
    """
    QASC (AllenAI) loader → simple MC prompt builder.

    HF schema (see dataset card): question: str, choices: {text: [..], label: [..]}, answerKey: str.
    We keep up to 4 options: if >4, downsample while preserving the correct answer.
    """
    try:
        ds = load_dataset("allenai/qasc", split=split, streaming=True)
    except Exception as e:
        raise RuntimeError(f"[QASC] load failed: {type(e).__name__}: {e}")

    # Pre-tokenize a bounded number of items for speed.
    samples: List[torch.Tensor] = []
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    it = iter(ds)
    while len(samples) < int(max_items):
        try:
            ex = next(it)
        except StopIteration:
            break
        except Exception:
            continue

        q = ex.get("question", None)
        ch = ex.get("choices", {})
        ans = ex.get("answerKey", None)
        if not q or not ans:
            continue

        opt_texts = ch.get("text", None) if isinstance(ch, dict) else None
        opt_labels = ch.get("label", None) if isinstance(ch, dict) else None
        if not opt_texts:
            continue

        # Normalize labels; if absent, synthesize A,B,C,...
        if not opt_labels or len(opt_labels) != len(opt_texts):
            opt_labels = letters[: len(opt_texts)]

        # Downselect to 4 options max while keeping the correct answer
        if len(opt_texts) > 4:
            # find index of correct answer
            try:
                idx_correct = opt_labels.index(ans)
            except ValueError:
                # If answerKey not found in provided labels, skip example
                continue
            # keep the correct one and sample three others
            others = [i for i in range(len(opt_texts)) if i != idx_correct]
            keep = [idx_correct] + random.sample(others, k=3 if len(others) >= 3 else len(others))
            keep.sort()
            opt_texts = [opt_texts[i] for i in keep]
            opt_labels = [opt_labels[i] for i in keep]
            # re-map labels to A-D and adjust answerKey
            opt_labels = list("ABCD")[: len(opt_texts)]
            ans = opt_labels[0] if keep[0] == idx_correct else opt_labels[keep.index(idx_correct)]

        prompt = _fmt_mc_prompt(q, opt_texts, opt_labels, ans) + ans  # append the gold label as target token(s)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len, padding="max_length").input_ids.squeeze(0)
        # Ensure there is at least something besides padding
        if int((ids != pad_id).sum()) < 5:
            continue
        samples.append(ids.cpu())

    if not samples:
        raise RuntimeError("[QASC] no usable examples produced (empty samples).")

    def gen(b: int) -> torch.Tensor:
        out = []
        for _ in range(b):
            seq = random.choice(samples)
            out.append(seq.unsqueeze(0))
        return torch.cat(out, dim=0)

    return gen