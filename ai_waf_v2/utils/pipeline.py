"""
ai_waf_v2/utils/pipeline.py
---------------------------
Shared guards for pipeline stage scripts.

Every stage script should call:
  1. require_inputs()  — fail fast if prerequisite outputs are missing
  2. check_output()    — skip if output already exists, unless --force

Usage pattern
-------------
    from ai_waf_v2.utils.pipeline import require_inputs, check_output

    def parse_args():
        p = argparse.ArgumentParser()
        p.add_argument("--config", default="config/pipeline.yaml")
        p.add_argument("--force",  action="store_true",
                       help="Re-run even if outputs already exist")
        return p.parse_args()

    def run(args):
        require_inputs({
            "data/normalized/deduped.parquet": "make data_collect",
        })
        if check_output(Path("reports/metrics/corpus_report.json"), args.force):
            return
        # ... do work ...
"""

from __future__ import annotations

import sys
from pathlib import Path

from ai_waf_v2.utils.logging import get_logger

log = get_logger(__name__)


def require_inputs(inputs: dict[str | Path, str]) -> None:
    """
    Verify that all prerequisite files exist before starting work.

    Parameters
    ----------
    inputs : {path: make_target_or_hint}
        Keys are file paths that must exist; values are the make target or
        script name the user should run to produce them.

    Exits with code 1 on the first missing file.
    """
    missing = False
    for path, hint in inputs.items():
        p = Path(path)
        if not p.exists():
            log.error(f"Required input missing: {p}  →  run: {hint}")
            missing = True
    if missing:
        sys.exit(1)


def check_output(output: Path, force: bool, label: str = "") -> bool:
    """
    Return True (and log a skip message) when the output file already exists
    and ``force`` is False.  The caller should return immediately in that case.

    Parameters
    ----------
    output : Path
        The primary output file this stage produces.
    force  : bool
        When True, always return False (proceed with re-running the stage).
    label  : str
        Optional human-readable name for the output used in log messages.

    Returns
    -------
    bool
        True  → skip (output exists and force=False)
        False → proceed (output missing or force=True)
    """
    if not output.exists() or force:
        return False

    try:
        size = output.stat().st_size
    except OSError:
        return False

    if size == 0:
        log.info(f"Output exists but is empty — re-running: {output}")
        return False

    name = label or output.name
    log.info(
        f"{name} already exists ({size:,} bytes) — skipping. "
        "Pass --force to re-run."
    )
    return True
