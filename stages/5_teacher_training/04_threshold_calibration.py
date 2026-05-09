"""Stage 4.5 — Calibrate classification threshold on val set at target FPR."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ai_waf_v2.data.collator import WafCollator
from ai_waf_v2.data.dataset import WafDataset, get_split_path
from ai_waf_v2.eval.metrics import find_threshold_at_fpr, compute_threshold_sweep
from ai_waf_v2.models.head import WafClassifier
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = HttpTokenizer.load(cfg.tokenizer.track_b.output_dir, cfg.tokenizer.seq_len)
    ckpt = Path(args.model_path or (Path(cfg.model.track_b_99m.output_dir)/"best_99m.pt"))
    if not ckpt.exists(): log.error(f"Checkpoint not found: {ckpt}"); return
    model = WafClassifier.load(ckpt, cfg.model.track_b_99m, map_location=str(device))
    model.to(device).eval()
    collator   = WafCollator(pad_token_id=tokenizer.pad_token_id, max_seq_len=cfg.tokenizer.seq_len)
    val_loader = DataLoader(WafDataset(get_split_path(cfg.paths.data_splits,"val"),
                            tokenizer._tok, cfg.tokenizer.seq_len),
                            batch_size=256, shuffle=False, collate_fn=collator, num_workers=2)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type=="cuda" else torch.no_grad():
                _, probs = model.predict(ids, mask)
            all_probs.append(probs.cpu()); all_labels.append(batch["labels"])
    probs_t  = torch.cat(all_probs); labels_t = torch.cat(all_labels)
    target   = args.target_fpr or cfg.training.teacher.operating_fpr_target
    threshold = find_threshold_at_fpr(probs_t, labels_t, target_fpr=target)
    sweep     = compute_threshold_sweep(probs_t, labels_t, n_thresholds=200)
    result = {"calibrated_threshold": threshold, "target_fpr": target,
              "sweep": {"thresholds": sweep["thresholds"][:20],
                        "f1": sweep["f1"][:20], "fpr": sweep["fpr"][:20]}}
    out = Path(cfg.paths.reports)/"metrics"/"threshold_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info(f"Calibrated threshold={threshold:.4f} at FPR<={target}")
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="config/pipeline.yaml")
    p.add_argument("--model-path",  default=None)
    p.add_argument("--target-fpr",  type=float, default=None)
    return p.parse_args()
if __name__ == "__main__": run(parse_args())
