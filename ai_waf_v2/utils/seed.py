"""
ai_waf_v2.utils.seed
----------------
Set all random seeds for reproducible experiments.

Usage
-----
    from ai_waf_v2.utils.seed import seed_everything
    seed_everything(42)        # call once at the top of every stage script
"""

from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    Seed Python, NumPy, PyTorch CPU, PyTorch CUDA, and (optionally)
    enable cuDNN determinism.

    Parameters
    ----------
    seed : int
        The seed value to use. Default: 42.
    deterministic_cudnn : bool
        If True, sets torch.backends.cudnn.deterministic = True and
        torch.backends.cudnn.benchmark = False. This guarantees reproducibility
        at the cost of ~5–10% throughput on convolution-heavy ops.
        For transformer attention (no convolutions), the overhead is negligible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)   # multi-GPU
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # PyTorch 1.11+: warn when a non-deterministic op is used
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass