"""Stage 7.5 — Obfuscation robustness: chained multi-tamper evaluation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pyarrow.parquet as pq, torch
from ai_waf_v2.data.schema import HttpRecord
from ai_waf_v2.eval.adversarial import TAMPER_REGISTRY, AdversarialEvaluator
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer
from ai_waf_v2.utils.reports import report_path
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    require_inputs({
        "data/splits/test.parquet": "make data_augment_all",
        f"{cfg.model.track_b_99m.output_dir}/best_99m.pt": "run 00_train_teacher_99m.py",
    })
    if check_output(report_path("obfuscation_robustness.json", cfg.paths.reports),
                    args.force, "Stage 7.5 obfuscation robustness"):
        return
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = HttpTokenizer.load(cfg.tokenizer.track_b.output_dir, cfg.tokenizer.seq_len)
    test_path = Path(cfg.paths.data_splits)/"test.parquet"
    if not test_path.exists(): log.error("No test split"); return
    table   = pq.read_table(test_path, filters=[("label","=",1)])
    records = [HttpRecord.from_dict(r) for r in table.to_pylist()][:1000]
    ckpt    = Path(cfg.model.track_b_99m.output_dir)/"best_99m.pt"
    if not ckpt.exists(): log.error("Teacher checkpoint not found"); return
    from ai_waf_v2.models.head import WafClassifier
    model = WafClassifier.load(ckpt, cfg.model.track_b_99m, map_location=str(device))
    model.to(device).eval()
    evaluator = AdversarialEvaluator(model=model, tokenizer=tokenizer,
                                      device=str(device), seq_len=cfg.tokenizer.seq_len)
    # Chained: apply 2 tampers simultaneously
    import uuid, random
    rng = random.Random(cfg.project.seed)
    timer = StepTimer()
    chained_results = []
    tamper_names = list(TAMPER_REGISTRY.keys())
    with timer.step("chained_evaluation"):
        for i in range(0, min(len(tamper_names)-1, 4)):
            t1_name, t2_name = tamper_names[i], tamper_names[i+1]
            t1_fn, t2_fn = TAMPER_REGISTRY[t1_name], TAMPER_REGISTRY[t2_name]
            chained_records = []
            for r in records:
                d = r.model_dump(); d["id"] = str(uuid.uuid4())
                qs = t2_fn(t1_fn(r.query_string))
                bd = t2_fn(t1_fn(r.body)) if r.body else ""
                d["query_string"] = qs; d["body"] = bd; d["source"] = f"chain_{t1_name}+{t2_name}"
                chained_records.append(HttpRecord(**d).build_raw())
            preds, probs, lbls = evaluator._predict_records(chained_records, [1]*len(chained_records))
            from ai_waf_v2.eval.metrics import compute_metrics
            m = compute_metrics(preds, probs, lbls)
            chained_results.append({"chain": f"{t1_name}+{t2_name}",
                                      "detection_rate": round(m["recall"],4),
                                      "evasion_rate": round(1-m["recall"],4)})
            log.info(f"  {t1_name}+{t2_name}: detection={m['recall']:.4f}")
    out = report_path("obfuscation_robustness.json", cfg.paths.reports)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": chained_results, "timings_s": timer.timings}, indent=2))
    log.info(f"Obfuscation robustness saved to {out}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import mlflow_run, log_metrics_dict
        with mlflow_run(cfg, run_name="05_obfuscation_robustness") as _run:
            mlflow.log_params({
                "n_records_cap":   1000,
                "n_chains_tested": len(chained_results),
            })
            metrics: dict[str, float] = {}
            if chained_results:
                metrics["mean_detection_rate"] = float(
                    sum(r["detection_rate"] for r in chained_results) / len(chained_results)
                )
                metrics["mean_evasion_rate"] = float(
                    sum(r["evasion_rate"] for r in chained_results) / len(chained_results)
                )
                for r in chained_results:
                    safe_chain = r["chain"].replace("+", "_")
                    metrics[f"detection_{safe_chain}"] = float(r["detection_rate"])
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(out))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true", help="Re-run even if outputs already exist")
    return p.parse_args()
if __name__ == "__main__": run(parse_args())