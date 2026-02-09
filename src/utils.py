import os
import random
import numpy as np
import torch

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pick_device(device: str = "cpu") -> torch.device:
    """
    device:
      - "cpu"  (default)
      - "cuda"
      - "mps"
      - "auto"  (prefer cuda, then mps, else cpu)
    """
    device = device.lower()
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unknown device='{device}'. Use cpu|cuda|mps|auto.")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
