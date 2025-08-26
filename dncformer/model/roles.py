# dncformer/model/roles.py
from __future__ import annotations
import torch

VALID_ROLES = {"generic", "read_only", "write_only", "vanilla_only"}

def set_block_roles(head, roles, prefer_vanilla_bias: float = 2.5):
    """
    roles: list[str] of length len(head.blocks), each in:
           {"read_write","read_only","write_only","vanilla_only"}.
    - For read/write roles we set `DNCformerBlock.io_role`.
    - For vanilla_only we strongly bias the gate towards vanilla path.
    """
    assert len(roles) == len(head.blocks), "roles must match number of blocks"
    for i, role in enumerate(roles):
        assert role in VALID_ROLES, f"invalid role[{i}]: {role}"
        peb = head.blocks[i]

        # Apply to all memory experts in this block
        for m in getattr(peb, "dncblocks", []):
            if role in ("read_write", "read_only", "write_only"):
                setattr(m, "io_role", role)

        # For vanilla_only, (a) return memory to generic internally and (b) bias gate
        if role == "vanilla_only":
            for m in getattr(peb, "dncblocks", []):
                setattr(m, "io_role", "read_write")
            with torch.no_grad():
                if hasattr(peb, "gate") and getattr(peb.gate, "bias", None) is not None:
                    # gate is (K+1): index 0 is vanilla, 1..K are memory experts
                    bias = peb.gate.bias
                    bias.zero_()
                    bias[0] = float(abs(prefer_vanilla_bias))
                    if getattr(peb, "mem_experts", 1) > 0:
                        bias[1:] = float(-abs(prefer_vanilla_bias))
