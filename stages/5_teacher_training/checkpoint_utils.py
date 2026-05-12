"""
stages/5_training/checkpoint_utils.py
--------------------------------------
Unified checkpoint management for all Stage 5 training scripts.

Resolves the "Unified Checkpoint Management" gap from the critique:
Stage 6 (Inference/Deployment) can always reference a static symlink
path like `models/track_b/latest` regardless of training timestamp
or step count.

Public API
----------
  save_checkpoint(model, meta, ckpt_dir)  — atomically save a checkpoint
  load_checkpoint(path, model)            — restore weights + return metadata
  CheckpointTracker                       — tracks best metric, calls save + symlink
  resolve_checkpoint(experiment_type, cfg)— returns canonical checkpoint path
                                            for use in scripts 04 and 05
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ai_waf_v2.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps experiment_type → relative directory under cfg.paths.models
EXPERIMENT_DIRS: dict[str, str] = {
    "track_a_large": "track_a/large",
    "track_a_small": "track_a/small",
    "track_b_99m":   "track_b/99m",
}

_WEIGHTS_FILE  = "model_weights.pt"
_METADATA_FILE = "checkpoint_meta.json"
_SYMLINK_NAME  = "latest"


# ---------------------------------------------------------------------------
# Low-level save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:    nn.Module,
    meta:     dict[str, Any],
    ckpt_dir: Path,
) -> Path:
    """
    Atomically save model weights and metadata to *ckpt_dir*.

    The write goes to a temporary sibling directory and is renamed into
    place, so a failed save never leaves a corrupt checkpoint.

    Args:
        model:    The full model (backbone + head).
        meta:     Arbitrary JSON-serialisable metadata (epoch, metrics, …).
        ckpt_dir: Target directory — created if absent.

    Returns:
        The resolved path of the saved checkpoint directory.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(dir=ckpt_dir.parent, prefix=".tmp_ckpt_"))
    try:
        torch.save(model.state_dict(), tmp_dir / _WEIGHTS_FILE)
        (tmp_dir / _METADATA_FILE).write_text(json.dumps(meta, indent=2))

        # Atomic rename — on POSIX this is guaranteed; on Windows it's best-effort
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        tmp_dir.rename(ckpt_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    log.info(f"Checkpoint saved → {ckpt_dir}")
    return ckpt_dir.resolve()


def load_checkpoint(
    path:  Path | str,
    model: nn.Module,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Load weights from *path* into *model* in-place.

    *path* may point to a checkpoint directory (containing model_weights.pt)
    or directly to a .pt file.

    Returns:
        Metadata dict (empty dict if no metadata file exists).
    """
    path = Path(path)

    # Resolve symlinks (e.g. models/track_b/99m/latest)
    if path.is_symlink():
        path = path.resolve()

    weights_path = path / _WEIGHTS_FILE if path.is_dir() else path
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")

    map_location = device or torch.device("cpu")
    state_dict   = torch.load(weights_path, map_location=map_location)
    model.load_state_dict(state_dict, strict=True)
    log.info(f"Weights loaded from {weights_path}")

    meta_path = weights_path.parent / _METADATA_FILE
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    return meta


# ---------------------------------------------------------------------------
# Symlink management
# ---------------------------------------------------------------------------

def update_best_symlink(
    ckpt_dir:        Path,
    experiment_type: str,
    models_root:     Path,
) -> Path:
    """
    Create or update a `latest` symlink inside the experiment model directory.

    Layout:
        models/
          track_b/
            99m/
              latest  →  <ckpt_dir>   (symlink)
              step_1200/
              step_2400/

    Stage 6 always loads `models/<track>/latest` and is therefore
    independent of training timestamps or step counts.

    Returns:
        The path of the updated symlink.
    """
    rel_dir  = EXPERIMENT_DIRS.get(experiment_type)
    if rel_dir is None:
        raise ValueError(
            f"Unknown experiment_type '{experiment_type}'. "
            f"Valid values: {list(EXPERIMENT_DIRS)}"
        )

    link_parent = models_root / rel_dir
    link_parent.mkdir(parents=True, exist_ok=True)
    link_path = link_parent / _SYMLINK_NAME

    # Remove stale symlink or empty placeholder
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()

    link_path.symlink_to(ckpt_dir.resolve())
    log.info(f"Best model symlink updated: {link_path} → {ckpt_dir.resolve()}")
    return link_path


# ---------------------------------------------------------------------------
# CheckpointTracker
# ---------------------------------------------------------------------------

@dataclass
class CheckpointTracker:
    """
    Tracks the best validation metric across training steps and manages
    saving + symlink updates when a new best is found.

    Args:
        experiment_type: One of the keys in EXPERIMENT_DIRS.
        models_root:     Root directory for model artefacts.
        metric:          Name of the metric to track (must be in eval dict).
        mode:            'max' (e.g. macro_f1) or 'min' (e.g. loss).
        patience:        Early-stopping patience (epochs without improvement).
    """

    experiment_type: str
    models_root:     Path
    metric:          str  = "macro_f1"
    mode:            Literal["max", "min"] = "max"
    patience:        int  = 5

    best_value:   float = field(init=False)
    best_ckpt:    Path  = field(init=False, default=None)  # type: ignore[assignment]
    no_improve:   int   = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.best_value = float("-inf") if self.mode == "max" else float("inf")

    def _is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value
        return value < self.best_value

    def step(
        self,
        model:     nn.Module,
        eval_metrics: dict[str, Any],
        meta:      dict[str, Any],
        step:      int,
    ) -> bool:
        """
        Call at the end of each evaluation step.

        Saves a checkpoint unconditionally (under step_{step}) and updates
        the `latest` symlink only when a new best is reached.

        Args:
            model:        Model to checkpoint.
            eval_metrics: Output of evaluate().
            meta:         Supplementary metadata to embed in the checkpoint.
            step:         Current global step or epoch number.

        Returns:
            True if this is a new best checkpoint.
        """
        value = float(eval_metrics.get(self.metric, 0.0))
        rel   = EXPERIMENT_DIRS[self.experiment_type]
        ckpt_dir = self.models_root / rel / f"step_{step:06d}"

        full_meta = {**meta, **eval_metrics, "step": step, "experiment_type": self.experiment_type}
        save_checkpoint(model, full_meta, ckpt_dir)

        if self._is_better(value):
            self.best_value = value
            self.best_ckpt  = ckpt_dir
            self.no_improve = 0
            update_best_symlink(ckpt_dir, self.experiment_type, self.models_root)
            log.info(
                f"New best {self.metric}={value:.5f} at step {step} — symlink updated."
            )
            return True

        self.no_improve += 1
        log.info(
            f"{self.metric}={value:.5f} (best={self.best_value:.5f}, "
            f"no_improve={self.no_improve}/{self.patience})"
        )
        return False

    @property
    def should_stop(self) -> bool:
        """True when patience has been exhausted."""
        return self.no_improve >= self.patience


# ---------------------------------------------------------------------------
# Checkpoint resolution for eval scripts (04, 05)
# ---------------------------------------------------------------------------

def resolve_checkpoint(experiment_type: str, cfg: Any) -> Path:
    """
    Return the canonical checkpoint path for a given experiment type.

    Checks (in order):
      1. The `latest` symlink under models_root / <rel_dir>
      2. cfg.model.<experiment_type>.checkpoint_dir (explicit override)

    Raises FileNotFoundError if neither exists.
    """
    rel = EXPERIMENT_DIRS.get(experiment_type)
    if rel is None:
        raise ValueError(
            f"Unknown experiment_type '{experiment_type}'. "
            f"Valid values: {list(EXPERIMENT_DIRS)}"
        )

    models_root = Path(cfg.paths.models)
    symlink     = models_root / rel / _SYMLINK_NAME

    if symlink.exists():
        resolved = symlink.resolve()
        log.info(f"Resolved '{experiment_type}' checkpoint via symlink: {resolved}")
        return resolved

    # Fallback: explicit config key
    try:
        model_cfg  = getattr(cfg.model, experiment_type)
        explicit   = Path(model_cfg.checkpoint_dir)
        if explicit.exists():
            log.info(f"Resolved '{experiment_type}' checkpoint from config: {explicit}")
            return explicit
    except AttributeError:
        pass

    raise FileNotFoundError(
        f"No checkpoint found for experiment_type='{experiment_type}'. "
        f"Expected symlink at {symlink} or config key cfg.model.{experiment_type}.checkpoint_dir."
    )