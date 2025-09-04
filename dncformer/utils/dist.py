from __future__ import annotations
import os, socket, contextlib
import torch
import torch.distributed as dist

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def get_local_rank() -> int:
    # torchrun sets LOCAL_RANK
    try:
        return int(os.environ.get("LOCAL_RANK", "0"))
    except Exception:
        return 0

def is_main_process() -> bool:
    return get_rank() == 0

def _pick_free_port() -> int:
    # only used on non-torchrun manual init
    print("[context] non-torchrun manual init - did you mean to do this?")
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def init_distributed(backend: str = "nccl") -> tuple[int, int]:
    """
    Initialize torch.distributed if torchrun passed env; return (local_rank, world_size).
    If not under torchrun, returns (0,1) and does nothing.
    """
    if is_distributed():
        return get_local_rank(), get_world_size()

    # Detect torchrun
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 1

    rank = int(os.environ.get("RANK", "0"))
    local_rank = get_local_rank()
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", str(_pick_free_port()))

    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    return local_rank, world

def barrier():
    if is_distributed():
        dist.barrier()

def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()
