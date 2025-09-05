# dncformer/data/tasks_algo.py
from __future__ import annotations
from typing import Callable, List
import random, string, torch

def _pad_or_trim(ids, T, pad_id):
    n = ids.numel()
    if n == T: return ids
    if n > T:  return ids[:T]
    return torch.cat([ids, torch.full((T-n,), pad_id, dtype=torch.long)], dim=0)

def _batchize(tok, prompts: List[str], B: int, max_len: int, pad_id: int) -> torch.Tensor:
    outs=[]
    for _ in range(B):
        s = random.choice(prompts)
        ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        outs.append(_pad_or_trim(ids, max_len, pad_id).unsqueeze(0))
    return torch.cat(outs, dim=0)

def _rand_tokens(alpha: str, n: int) -> List[str]:
    return [random.choice(alpha) for _ in range(n)]

def make_dyck_gen(tok, max_len: int, pad_id: int, depth: int = 4, T: int = 80):
    """
    Generate Dyck-like parenthesis strings with maximum nesting bounded by `depth`
    Occasionally flip a single token to create a near-miss error
    """
    assert depth >= 1, "depth must be >= 1"

    def _make_sample():
        s = []
        d = 0  # current stack depth
        steps = max(1, T // 2)

        for _ in range(steps):
            if d == 0:
                # must open to start
                s.append("("); d += 1
            else:
                # If we haven't reached the max depth, we can still open.
                # Bias toward opening a bit, but never exceed `depth`.
                can_open = (d < depth)
                open_bias = 0.6
                if can_open and random.random() < open_bias:
                    s.append("("); d += 1
                else:
                    s.append(")"); d = max(0, d - 1)

        # close any remaining opens
        s += [")"] * d
        text = "".join(s)

        # With 30% chance, inject a single flip error to create a "fix or classify" variant
        if random.random() < 0.3 and len(text) > 2:
            i = random.randrange(1, len(text) - 1)
            text = text[:i] + (")" if text[i] == "(" else "(") + text[i + 1:]
            q = (
                f"### Sequence:\n{text}\n"
                f"### Task:\nFix the sequence to be well-formed.\n"
                f"### Fixed:\n"
            )
        else:
            q = (
                f"### Sequence:\n{text}\n"
                f"### Task:\nIs it well-formed? Answer Yes/No and explain briefly.\n"
                f"### Answer:\n"
            )
        return q

    prompts = [_make_sample() for _ in range(256)]

    def gen(B: int) -> torch.Tensor:
        return _batchize(tok, prompts, B, max_len, pad_id)

    return gen

def make_stack_ops_gen(tok, max_len: int, pad_id: int, ops: int = 24) -> Callable[[int], torch.Tensor]:
    """
    Natural-language push/pop program; ask final top-of-stack and full content.
    """
    vals = list(string.ascii_uppercase[:16])

    def _one():
        st = []
        stack=[]
        for _ in range(ops):
            if not stack or random.random() < 0.6:
                v = random.choice(vals); stack.append(v)
                st.append(f"push {v}")
            else:
                stack.pop()
                st.append("pop")
        seq = "; ".join(st)
        tgt = "".join(stack) if stack else "(empty)"
        return f"### Program:\n{seq}.\n### Task:\nWhat is the final top item and full stack state?\n### Answer:\n"

    prompts = [_one() for _ in range(256)]
    def gen(B: int) -> torch.Tensor:
        return _batchize(tok, prompts, B, max_len, pad_id)
    return gen

def make_sort_list_gen(tok, max_len: int, pad_id: int, length: int = 12, vocab: int = 100) -> Callable[[int], torch.Tensor]:
    """
    Ask model to sort a list; lists fit entirely into the prompt window (linguistic).
    """
    def _one():
        xs = [random.randint(0, vocab-1) for _ in range(length)]
        xs_txt = ", ".join(map(str, xs))
        return f"### Numbers:\n{xs_txt}\n### Task:\nSort ascending. Provide the list only.\n### Answer:\n"

    prompts = [_one() for _ in range(256)]
    def gen(B: int) -> torch.Tensor:
        return _batchize(tok, prompts, B, max_len, pad_id)
    return gen
