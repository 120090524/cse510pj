from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def wrap_monitor(env, log_dir: str | Path):
    log_dir = ensure_dir(log_dir)
    return Monitor(env, filename=str(log_dir / "monitor.csv"))



def save_metadata(output_dir: str | Path, metadata: dict[str, Any]) -> None:
    output_dir = ensure_dir(output_dir)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
