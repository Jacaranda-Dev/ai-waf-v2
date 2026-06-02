"""Stage 7.12 — SHAP analysis on the student model (small enough for KernelSHAP)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
import pyarrow.parquet as pq
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    require_inputs({
        f"{cfg.model.student.output_dir}/best_student.pt": "run 02_distill_train.py",
        "data/splits/test.parquet": "make data_augment_all",
    })
    if check_output(Path(cfg.paths.reports) / "metrics" / "shap_analysis.json",
                    args.force, "Stage 7.12 SHAP analysis"):
        return
    try: import shap
    except ImportError: log.error("shap not installed: pip install shap"); return
    device    = torch.device("cpu")  # SHAP works on CPU
    tokenizer = HttpTokenizer.load(cfg.tokenizer.track_b.output_dir, cfg.tokenizer.seq_len)
    ckpt = Path(cfg.model.student.output_dir)/"best_student.pt"
    if not ckpt.exists(): log.error("Student checkpoint not found"); return
    from ai_waf_v2.models.student import StudentClassifier
    model = StudentClassifier.load(ckpt, cfg.model.student, map_location="cpu")
    model.eval()
    # Load small background and test sets
    test_path = Path(cfg.paths.data_splits)/"test.parquet"
    if not test_path.exists(): log.error("test.parquet not found"); return
    df = pq.read_table(test_path, columns=["raw","label"]).to_pandas()
    def encode(texts):
        ids_list = []; mask_list = []
        for text in texts:
            enc  = tokenizer.encode(text); seq = cfg.tokenizer.seq_len
            ids  = enc.ids[:seq]; mask = enc.attention_mask[:seq]
            pad  = seq - len(ids); pad_id = tokenizer.pad_token_id
            ids_list.append(ids + [pad_id]*pad); mask_list.append(mask + [0]*pad)
        return np.array(ids_list, dtype=np.int64), np.array(mask_list, dtype=np.int64)
    def predict(ids_array):
        out = []
        for i in range(0, len(ids_array), 16):
            batch_ids  = torch.tensor(ids_array[i:i+16], dtype=torch.long)
            batch_mask = torch.ones_like(batch_ids)
            with torch.no_grad():
                _, probs = model.predict(batch_ids, batch_mask)
            out.extend(probs.tolist())
        return np.array(out)
    background_texts = df[df["label"]==0]["raw"].tolist()[:50]
    test_malicious   = df[df["label"]==1]["raw"].tolist()[:20]
    bg_ids, _    = encode(background_texts)
    test_ids, _  = encode(test_malicious)
    timer = StepTimer()
    log.info("Running KernelSHAP (this may take a few minutes)...")
    with timer.step("kernel_shap"):
        explainer = shap.KernelExplainer(predict, bg_ids[:20])
        shap_vals  = explainer.shap_values(test_ids[:5], nsamples=100)
    # Map back to tokens
    results = []
    for i, text in enumerate(test_malicious[:5]):
        enc   = tokenizer.encode(text)
        toks  = enc.tokens[:cfg.tokenizer.seq_len]
        svals = shap_vals[i][:len(toks)].tolist()
        top5  = sorted(enumerate(svals), key=lambda x: abs(x[1]), reverse=True)[:5]
        results.append({"text_preview": text[:100],
                         "tokens": toks[:20], "shap_values": svals[:20],
                         "top5": [(toks[j] if j < len(toks) else "PAD", round(v,5)) for j,v in top5]})
        top_token = results[-1]["top5"][0] if results[-1]["top5"] else "?"
        log.info(f"  top token: {top_token}")
    out = Path(cfg.paths.reports)/"metrics"/"shap_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info(f"SHAP analysis saved to {out}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import mlflow_run, log_metrics_dict
        with mlflow_run(cfg, run_name="12_shap_analysis") as _run:
            mlflow.log_params({
                "n_background_samples": len(background_texts),
                "n_test_samples":       len(test_malicious[:5]),
                "shap_nsamples":        100,
            })
            log_metrics_dict({
                "n_explained_samples": float(len(results)),
            })
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