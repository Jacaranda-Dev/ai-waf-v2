"""
stages/6_distillation_and_compression/02_distill_train.py
------------------------------------
Stage 5.2 — Knowledge distillation: 99M teacher → INT8 student.

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
from ai_waf_v2.utils.seed import seed_everything

log = get_logger(__name__, log_file="reports/distill_train.log")


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg         = load_config(args.config)
    teacher_cfg = cfg.model.track_b_99m
    student_cfg = cfg.model.student
    dcfg        = cfg.training.distillation

    seed_everything(cfg.project.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Tokenizer ────────────────────────────────
    tokenizer = HttpTokenizer.load(
        cfg.tokenizer.track_b.output_dir,
        seq_len=cfg.tokenizer.seq_len,
    )

    # ── Teacher ───────────────────────────────────
    teacher_path = args.teacher_path or (
        Path(teacher_cfg.output_dir) / "best_99m.pt"
    )
    if not Path(teacher_path).exists():
        log.error(f"Teacher checkpoint not found: {teacher_path}")
        return

    log.info(f"Loading teacher from {teacher_path}...")
    teacher = WafClassifier.load(teacher_path, teacher_cfg, map_location="cpu")
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    log.info(f"Teacher: {teacher.count_parameters():,} parameters")

    # ── Student ───────────────────────────────────
    student = StudentClassifier.from_config(
        student_cfg,
        teacher_d_model=teacher_cfg.d_model if dcfg.hidden_mse_weight > 0 else None,
    )

    if student_cfg.quantization == "int8":
        log.info("Applying INT8 QAT to student...")
        student.prepare_for_int8_quantization()

    log.info(f"Student: {student.count_parameters():,} parameters")
    compression = teacher.count_parameters() / max(1, student.count_parameters())
    log.info(f"Compression ratio: {compression:.1f}×")

    # ── Data ─────────────────────────────────────
    collator = WafCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_seq_len=cfg.tokenizer.seq_len,
    )
    splits_dir = cfg.paths.data_splits

    train_loader = DataLoader(
        WafDataset(get_split_path(splits_dir, "train"), tokenizer._tok, cfg.tokenizer.seq_len),
        batch_size=dcfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        WafDataset(get_split_path(splits_dir, "val"), tokenizer._tok, cfg.tokenizer.seq_len),
        batch_size=dcfg.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    # ── Train ─────────────────────────────────────
    init_experiment(cfg)
    run_name = f"distill_{int(time.time())}"

    import mlflow
    with mlflow.start_run(run_name=run_name, tags=cfg.mlflow.tags):
        mlflow.log_params({
            "teacher_params":     teacher.count_parameters(),
            "student_params":     student.count_parameters(),
            "compression_ratio":  round(compression, 2),
            "temperature":        dcfg.temperature,
            "alpha_soft":         dcfg.alpha_soft,
            "alpha_hard":         dcfg.alpha_hard,
            "quantization":       student_cfg.quantization,
        })

        trainer = DistillationTrainer(
            cfg=cfg,
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )
        trainer.train()

    log.info("Distillation training complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="config/pipeline.yaml")
    p.add_argument("--teacher-path", default=None)
    p.add_argument("--mlflow-run-name", default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())