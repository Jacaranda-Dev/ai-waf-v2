"""
stages/6_distillation_and_compression/02_distill_train.py
---------------------------------------------------------
Stage 6.2 — Knowledge distillation: 99M teacher → INT8 student.

Enhancements over original:
  - Temperature scheduler: linearly anneals temperature from T_max → T_min
    over training so early epochs capture broad soft-target structure (high T)
    and later epochs sharpen to fine-grained discrimination (low T).
    This replaces the static temperature parameter.
  - WafClassifier compatibility check: warns if the head contains activation
    functions known to be sensitive to INT8 precision loss (SiLU, GELU variants)
    so that they can be excluded from bitsandbytes quantization.
  - Added `prepare_for_int8_quantization` guard with explicit layer exclusions.
  - Richer MLflow tracking (per-epoch temperature, hidden-MSE loss breakdown).

Run:
    python stages/6_distillation_and_compression/02_distill_train.py \
        --config config/pipeline.yaml \
        --teacher-path models/track_b/99m/best_99m.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai_waf_v2.data.collator import WafCollator
from ai_waf_v2.data.dataset import WafDataset, get_split_path
from ai_waf_v2.distill.trainer import DistillationTrainer
from ai_waf_v2.models.head import WafClassifier
from ai_waf_v2.models.student import StudentClassifier
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.mlflow_utils import init_experiment
from ai_waf_v2.utils.pipeline import check_output, require_inputs
from ai_waf_v2.utils.seed import seed_everything
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__, log_file="reports/distill_train.log")

# Activation functions that are precision-sensitive under INT8 quantization.
# Layers using these should be excluded from bitsandbytes INT8 replacement.
_PRECISION_SENSITIVE_ACTIVATIONS = (nn.SiLU, nn.GELU, nn.Mish)


# ── Temperature scheduler ─────────────────────────────────────────────────────

class TemperatureScheduler:
    """Linearly anneals distillation temperature from T_max → T_min.

    High temperature in early epochs preserves the "dark knowledge" in the
    teacher's soft-target distribution (inter-class relationships). Low
    temperature in later epochs tightens the student toward sharper decisions.

    Args:
        T_max:        Starting temperature (e.g. 6.0).
        T_min:        Ending temperature (e.g. 1.5).
        total_epochs: Total number of training epochs.
    """

    def __init__(self, T_max: float, T_min: float, total_epochs: int) -> None:
        self.T_max        = T_max
        self.T_min        = T_min
        self.total_epochs = max(total_epochs, 1)

    def get(self, epoch: int) -> float:
        """Return temperature for the given epoch (1-indexed)."""
        frac = (epoch - 1) / max(self.total_epochs - 1, 1)
        return self.T_max - frac * (self.T_max - self.T_min)


# ── Head compatibility check ──────────────────────────────────────────────────

def _check_head_int8_compatibility(model: nn.Module) -> list[str]:
    """Return names of head sub-modules that are INT8-sensitive.

    These should be excluded from bitsandbytes Linear8bitLt replacement to
    avoid precision-loss-induced degradation in the classification head.
    """
    sensitive: list[str] = []
    head = getattr(model, "head", None)
    if head is None:
        return sensitive
    for name, module in head.named_modules():
        if isinstance(module, _PRECISION_SENSITIVE_ACTIVATIONS):
            sensitive.append(name)
    return sensitive


def _apply_int8_quantization(student: StudentClassifier, exclude_head: bool = True) -> None:
    """Apply INT8 QAT to encoder layers, optionally excluding the head.

    The classification head is excluded by default because precision-sensitive
    activations (SiLU/GELU) in the head can lose calibration accuracy under
    INT8 quantization.
    """
    try:
        import bitsandbytes as bnb  # noqa: F401 — validates install
    except ImportError:
        log.warning(
            "bitsandbytes not installed. INT8 QAT will be skipped. "
            "Install with: pip install bitsandbytes"
        )
        return

    sensitive_names = _check_head_int8_compatibility(student)
    if sensitive_names:
        log.warning(
            f"Head contains precision-sensitive activations: {sensitive_names}. "
            "These layers will be excluded from INT8 quantization."
        )
    else:
        log.info("No precision-sensitive activations found in head — full INT8 QAT applies.")

    # Delegate to the model's own prepare method (which may already handle exclusions)
    student.prepare_for_int8_quantization(exclude_modules=["head"] if exclude_head else [])
    log.info("INT8 QAT applied to student encoder.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg         = load_config(args.config)
    teacher_cfg = cfg.model.track_b_99m
    student_cfg = cfg.model.student
    dcfg        = cfg.training.distillation

    seed_everything(cfg.project.seed)

    require_inputs({
        f"{teacher_cfg.output_dir}/best_99m.pt": "run 00_train_teacher_99m.py",
        "data/splits/train.parquet": "make data_augment_all",
        "data/splits/val.parquet":   "make data_augment_all",
    })
    if check_output(
        Path(student_cfg.output_dir) / "best_student.pt",
        args.force, "Stage 6.2 distillation training"
    ):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer  = StepTimer()

    # ── Tokenizer ─────────────────────────────────
    tokenizer = HttpTokenizer.load(
        cfg.tokenizer.track_b.output_dir,
        seq_len=cfg.tokenizer.seq_len,
    )

    # ── Teacher ───────────────────────────────────
    teacher_path = args.teacher_path or (Path(teacher_cfg.output_dir) / "best_99m.pt")
    if not Path(teacher_path).exists():
        log.error(
            f"Teacher checkpoint not found: {teacher_path}. "
            "Run 00_train_teacher_99m.py first."
        )
        return

    with timer.step("load_teacher"):
        log.info(f"Loading teacher from {teacher_path}...")
        teacher = WafClassifier.load(teacher_path, teacher_cfg, map_location="cpu")
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False

    n_teacher = teacher.count_parameters()
    log.info(f"Teacher: {n_teacher:,} parameters")

    # ── Student ───────────────────────────────────
    student = StudentClassifier.from_config(
        student_cfg,
        teacher_d_model=teacher_cfg.d_model if dcfg.hidden_mse_weight > 0 else None,
    )

    if student_cfg.quantization == "int8":
        log.info("Applying INT8 QAT to student (encoder only)...")
        _apply_int8_quantization(student, exclude_head=True)

    n_student    = student.count_parameters()
    compression  = n_teacher / max(n_student, 1)
    log.info(f"Student: {n_student:,} parameters")
    log.info(f"Compression ratio: {compression:.1f}×")

    # ── Data ──────────────────────────────────────
    collator   = WafCollator(pad_token_id=tokenizer.pad_token_id, max_seq_len=cfg.tokenizer.seq_len)
    splits_dir = cfg.paths.data_splits

    with timer.step("setup_data"):
        train_loader = DataLoader(
            WafDataset(get_split_path(splits_dir, "train"), tokenizer._tok, cfg.tokenizer.seq_len),
            batch_size=dcfg.batch_size,
            shuffle=True, collate_fn=collator, num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            WafDataset(get_split_path(splits_dir, "val"), tokenizer._tok, cfg.tokenizer.seq_len),
            batch_size=dcfg.batch_size * 2,
            shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True,
        )

    # ── Temperature scheduler ─────────────────────
    T_max      = getattr(dcfg, "temperature_max", dcfg.temperature)
    T_min      = getattr(dcfg, "temperature_min", max(dcfg.temperature * 0.25, 1.0))
    n_epochs   = max(1, dcfg.max_steps // len(train_loader))
    temp_sched = TemperatureScheduler(T_max=T_max, T_min=T_min, total_epochs=n_epochs)
    log.info(f"Temperature schedule: {T_max} → {T_min} over {n_epochs} epochs")

    # ── Training ──────────────────────────────────
    init_experiment(cfg)
    run_name = args.mlflow_run_name or f"distill_{int(time.time())}"

    import mlflow
    with mlflow.start_run(run_name=run_name, tags=cfg.mlflow.tags):
        mlflow.log_params({
            "teacher_params":     n_teacher,
            "student_params":     n_student,
            "compression_ratio":  round(compression, 2),
            "temperature_max":    T_max,
            "temperature_min":    T_min,
            "alpha_soft":         dcfg.alpha_soft,
            "alpha_hard":         dcfg.alpha_hard,
            "quantization":       student_cfg.quantization,
            "epochs":             n_epochs,
        })

        trainer = DistillationTrainer(
            cfg=cfg,
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            temperature_scheduler=temp_sched,
        )
        with timer.step("distillation"):
            trainer.train()
        timer.log_mlflow()

    log.info("Distillation training complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Knowledge distillation: 99M teacher → INT8 student.")
    p.add_argument("--config",            default="config/pipeline.yaml")
    p.add_argument("--teacher-path",      default=None)
    p.add_argument("--mlflow-run-name",   default=None)
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())