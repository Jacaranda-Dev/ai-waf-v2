"""Stage 6.3 — Memory footprint: VRAM, RAM, and checkpoint sizes for all models."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def _gpu_mem_mb(model, device):
    if device.type != "cuda": return None
    torch.cuda.reset_peak_memory_stats(device)
    import torch.nn as nn
    dummy_ids  = torch.randint(1,100,(1,256)).to(device)
    dummy_mask = torch.ones(1,256,dtype=torch.long).to(device)
    with torch.no_grad():
        try: model(dummy_ids, dummy_mask)
        except Exception: pass
    return round(torch.cuda.max_memory_allocated(device)/1e6,1)
def run(args):
    configure_root(); cfg = load_config(args.config)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    checkpoints = [
        ("teacher_99m",  Path(cfg.model.track_b_99m.output_dir)/"best_99m.pt",   cfg.model.track_b_99m,  False),
        ("student",      Path(cfg.model.student.output_dir)/"best_student.pt",    cfg.model.student,      True),
    ]
    onnx_path = Path(cfg.model.student.output_dir)/"student.onnx"
    for label, ckpt, arch_cfg, is_student in checkpoints:
        if not ckpt.exists(): log.warning(f"{label}: checkpoint not found"); continue
        disk_mb = round(ckpt.stat().st_size/1e6,1)
        if is_student:
            from ai_waf_v2.models.student import StudentClassifier
            model = StudentClassifier.load(ckpt, arch_cfg, map_location=str(device))
        else:
            from ai_waf_v2.models.head import WafClassifier
            model = WafClassifier.load(ckpt, arch_cfg, map_location=str(device))
        model.to(device).eval()
        n_params = sum(p.numel() for p in model.parameters())
        vram_fp16 = round(n_params * 2 / 1e6, 1)
        vram_int8 = round(n_params * 1 / 1e6, 1)
        gpu_peak  = _gpu_mem_mb(model, device)
        results[label] = {"disk_mb": disk_mb, "n_params": n_params,
                           "vram_weights_fp16_mb": vram_fp16, "vram_weights_int8_mb": vram_int8,
                           "gpu_peak_activation_mb": gpu_peak}
        log.info(f"  {label:20s}: disk={disk_mb}MB  params={n_params:,}  vram_fp16={vram_fp16}MB")
        del model
    if onnx_path.exists():
        results["student_onnx"] = {"disk_mb": round(onnx_path.stat().st_size/1e6,1)}
        log.info(f"  {'student_onnx':20s}: disk={results['student_onnx']['disk_mb']}MB")
    out = Path(cfg.paths.reports)/"metrics"/"memory_footprint.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info(f"Memory footprint saved to {out}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="03_memory_footprint"):
            metrics: dict[str, float] = {}
            for label, info in results.items():
                if "disk_mb" in info:
                    metrics[f"{label}_disk_mb"] = float(info["disk_mb"])
                if "n_params" in info:
                    metrics[f"{label}_n_params"] = float(info["n_params"])
                if "vram_weights_fp16_mb" in info:
                    metrics[f"{label}_vram_fp16_mb"] = float(info["vram_weights_fp16_mb"])
                if "gpu_peak_activation_mb" in info and info["gpu_peak_activation_mb"] is not None:
                    metrics[f"{label}_gpu_peak_mb"] = float(info["gpu_peak_activation_mb"])
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(out))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)

def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())