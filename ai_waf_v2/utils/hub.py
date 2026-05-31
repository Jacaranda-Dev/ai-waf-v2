"""
ai_waf_v2.utils.hub
-------------------
Shared HuggingFace Hub push helpers.

Each pipeline stage that wants to save its outputs calls push_folder() or
push_file() at the end of run() when --save is set.

Auth
----
Set HF_TOKEN env var, or run `huggingface-cli login` once.
Set HF_ORG (or huggingface.org in pipeline.yaml) to your org/username slug.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_waf_v2.utils.logging import get_logger

log = get_logger(__name__)


def _repo_id(cfg, repo_key: str) -> str | None:
    hf = cfg.huggingface
    if not hf.org:
        log.warning(
            "HuggingFace org not set — export HF_ORG=<your-org> or set "
            "huggingface.org in pipeline.yaml"
        )
        return None
    repo_name = getattr(hf.repos, repo_key, None)
    if not repo_name:
        log.warning(f"No repo name configured for key '{repo_key}'")
        return None
    return f"{hf.org}/{repo_name}"


def _get_api():
    try:
        from huggingface_hub import HfApi
        return HfApi()
    except ImportError:
        log.error("huggingface_hub not installed — run: pip install huggingface_hub")
        return None


def _ensure_repo(api, repo_id: str, repo_type: str, private: bool) -> bool:
    try:
        from huggingface_hub import create_repo
        create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
        return True
    except Exception as exc:
        log.warning(f"Could not create/verify repo {repo_id}: {exc}")
        return False


def push_folder(
    cfg,
    folder: Path | str,
    repo_key: str,
    repo_type: str = "dataset",
    commit_message: str = "Update",
    revision: str = "main",
    dry_run: bool = False,
    ignore_patterns: list[str] | None = None,
) -> bool:
    """
    Push a local folder to the HuggingFace Hub.

    Parameters
    ----------
    cfg : PipelineConfig
        Full pipeline config (reads cfg.huggingface for org/private/repo names).
    folder : Path
        Local folder to upload. Skipped gracefully if it doesn't exist.
    repo_key : str
        Key into cfg.huggingface.repos, e.g. "dataset_synthesis".
    repo_type : str
        "dataset" or "model".
    commit_message : str
        Commit message shown in the Hub repo history.
    revision : str
        Branch or tag to push to (default "main").
    dry_run : bool
        Log what would be uploaded without actually uploading.

    Returns
    -------
    bool
        True if the push succeeded (or would have in dry-run), False if skipped.
    """
    folder = Path(folder)
    if not folder.exists():
        log.warning(f"[hub] Skipping push — folder not found: {folder}")
        return False

    files = [f for f in folder.rglob("*") if f.is_file()]
    if not files:
        log.warning(f"[hub] Skipping push — folder is empty: {folder}")
        return False

    rid = _repo_id(cfg, repo_key)
    if rid is None:
        return False

    total_mb = sum(f.stat().st_size for f in files) / 1e6

    if dry_run:
        log.info(
            f"[hub] dry-run: would push {folder} → {repo_type}:{rid}@{revision} "
            f"({len(files)} files, {total_mb:.1f} MB)"
        )
        return True

    api = _get_api()
    if api is None:
        return False

    hf = cfg.huggingface
    if not _ensure_repo(api, rid, repo_type, hf.private):
        return False

    try:
        api.upload_folder(
            folder_path=str(folder),
            repo_id=rid,
            repo_type=repo_type,
            revision=revision,
            commit_message=commit_message,
            ignore_patterns=ignore_patterns or ["*.tmp", "*.lock", "__pycache__", "*.ipc"],
        )
        log.info(f"[hub] Pushed {folder} → {repo_type}:{rid}@{revision} ({total_mb:.1f} MB)")
        return True
    except Exception as exc:
        log.warning(f"[hub] Push failed for {rid}: {exc}")
        return False


def push_text(
    cfg,
    content: str,
    path_in_repo: str,
    repo_key: str,
    repo_type: str = "model",
    revision: str = "main",
    dry_run: bool = False,
) -> bool:
    """Upload a string (e.g. a generated README.md or config.json) to a repo."""
    rid = _repo_id(cfg, repo_key)
    if rid is None:
        return False

    if dry_run:
        log.info(f"[hub] dry-run: would write {path_in_repo} to {repo_type}:{rid}@{revision}")
        return True

    api = _get_api()
    if api is None:
        return False

    hf = cfg.huggingface
    _ensure_repo(api, rid, repo_type, hf.private)
    try:
        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo=path_in_repo,
            repo_id=rid,
            repo_type=repo_type,
            revision=revision,
        )
        return True
    except Exception as exc:
        log.warning(f"[hub] Failed to write {path_in_repo} to {rid}: {exc}")
        return False
