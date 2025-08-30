from __future__ import annotations
from typing import Any, Dict
import yaml
from dncformer import CFG

def load_yaml_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Flatten shallow dict → CFG attributes (nested groups optional)
    def apply(d: Dict[str, Any]):
        for k, v in d.items():
            if isinstance(v, dict):
                # lift nested dictionaries one level into CFG with prefix if desired
                for kk, vv in v.items():
                    setattr(CFG, kk, vv)
            else:
                setattr(CFG, k, v)
    apply(data)
    return data
