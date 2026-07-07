"""
ai_waf_v2.utils.mlflow_utils
------------------------
Thin wrappers around MLflow to keep stage scripts clean.

Usage
-----
    from ai_waf_v2.utils.mlflow_utils import init_experiment, mlflow_run

    init_experiment(cfg)

    with mlflow_run(cfg, run_name="track_b_99m_20250505") as run:
        mlflow.log_param("d_model", 768)
        mlflow.log_metric("val_auc_pr", 0.97, step=1000)
"""

from __future__ import annotations

import contextlib
from collections.abc import Generator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_waf_v2.utils.config import PipelineConfig


def init_experiment(cfg: PipelineConfig) -> str:
    """
    Set the MLflow tracking URI and create (or retrieve) the experiment.

    Returns
    -------
    str
        The experiment ID.
    """
    import mlflow

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment = mlflow.set_experiment(cfg.mlflow.experiment_name)
    return experiment.experiment_id


@contextlib.contextmanager
def mlflow_run(
    cfg: PipelineConfig,
    run_name: str,
    tags: dict[str, str] | None = None,
    nested: bool = False,
) -> Generator:
    """
    Context manager that starts an MLflow run, logs config-level tags,
    and ends the run cleanly on exit (even on exception).

    Parameters
    ----------
    cfg : PipelineConfig
    run_name : str
        Human-readable run name shown in the MLflow UI.
    tags : dict[str, str] | None
        Additional tags to attach to the run.
    nested : bool
        If True, allows nesting inside an active run.

    Yields
    ------
    mlflow.ActiveRun
    """
    import mlflow

    init_experiment(cfg)

    merged_tags = {**cfg.mlflow.tags, **(tags or {})}
    merged_tags["run_name"] = run_name

    with mlflow.start_run(run_name=run_name, tags=merged_tags, nested=nested) as run:
        # Log the full config as params (flattened with dot-notation keys)
        flat = _flatten(cfg.model_dump())
        # MLflow has a 250-char limit on param values; truncate safely
        mlflow.log_params({k: str(v)[:250] for k, v in flat.items()})
        yield run


def log_metrics_dict(
    metrics: dict[str, float],
    step: int | None = None,
    prefix: str = "",
) -> None:
    """
    Log a dictionary of metrics to the active MLflow run.

    Parameters
    ----------
    metrics : dict[str, float]
    step : int | None
    prefix : str
        Optional prefix added to every key (e.g. "val/").
    """
    import math

    import mlflow

    clean = {
        f"{prefix}{k}": v
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and math.isfinite(v)
    }
    if clean:
        mlflow.log_metrics(clean, step=step)


def log_artifact_path(path: str) -> None:
    """Log a local file or directory as an MLflow artifact."""
    import mlflow
    mlflow.log_artifact(path)


# ─────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────

def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, object]:
    """Flatten a nested dict into dot-notation keys."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)