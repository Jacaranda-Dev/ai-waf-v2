"""Stage 4.4 — Canary evaluation: check model accuracy on synthetic-only holdout."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ai_waf_v2.data.collator import WafCollator
from ai_waf_v2.data.dataset import WafDataset, get_split_path
from ai_waf_v2.eval.metrics import compute_metrics
from ai_waf_v2.models.head import WafClassifier
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = HttpTokenizer.load(cfg.tokenizer.track_b.output_dir, cfg.tokenizer.seq_len)
    ckpt = Path(args.model_path or (Path(cfg.model.track_b_99m.output_dir)/"best_99m.pt"))
    if not ckpt.exists(): log.error(f"Checkpoint not found: {ckpt}"); return
    model    = WafClassifier.load(ckpt, cfg.model.track_b_99m, map_location=str(device))
    model.to(device).eval()
    collator = WafCollator(pad_token_id=tokenizer.pad_token_id, max_seq_len=cfg.tokenizer.seq_len)
    canary_path = get_split_path(cfg.paths.data_splits, "canary")
    if not canary_path.exists(): log.warning("No canary split found"); return
    loader = DataLoader(WafDataset(canary_path, tokenizer._tok, cfg.tokenizer.seq_len),
                        batch_size=256, shuffle=False, collate_fn=collator, num_workers=2)
    all_p, all_pr, all_l = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            preds, probs = model.predict(ids, mask)
            all_p.append(preds.cpu()); all_pr.append(probs.cpu()); all_l.append(batch["labels"])
    metrics = compute_metrics(torch.cat(all_p), torch.cat(all_pr), torch.cat(all_l))
    log.info(f"Canary set: F1={metrics[\'f1\']:.4f}  AUC-PR={metrics[\'auc_pr\']:.4f}")
    if metrics["auc_pr"] < 0.85:
        log.warning("LOW canary AUC-PR — synthetic distribution may differ from real data!")
    else:
        log.info("Canary check PASSED — synthetic data generalises well.")
    out = Path(cfg.paths.reports)/"metrics"/"canary_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"canary_metrics": metrics, "passed": metrics["auc_pr"] >= 0.85}, indent=2))
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="config/pipeline.yaml")
    p.add_argument("--model-path", default=None)
    return p.parse_args()
if __name__ == "__main__": run(parse_args())
