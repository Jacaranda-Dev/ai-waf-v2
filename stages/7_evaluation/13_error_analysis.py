"""Stage 7.3 — Error analysis: characterise FP and FN samples."""

from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import torch
import pyarrow.parquet as pq
from ai_waf_v2.data.collator import WafCollator
from ai_waf_v2.data.dataset import WafDataset, get_split_path
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from torch.utils.data import DataLoader
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = HttpTokenizer.load(cfg.tokenizer.track_b.output_dir, cfg.tokenizer.seq_len)
    ckpt = Path(cfg.model.track_b_99m.output_dir)/"best_99m.pt"
    if not ckpt.exists(): log.error("Checkpoint not found"); return
    from ai_waf_v2.models.head import WafClassifier
    model = WafClassifier.load(ckpt, cfg.model.track_b_99m, map_location=str(device))
    model.to(device).eval()
    test_path = Path(cfg.paths.data_splits)/"test.parquet"
    if not test_path.exists(): log.error("No test split"); return
    df = pq.read_table(test_path, columns=["raw","label","attack_class","source"]).to_pandas()
    collator = WafCollator(pad_token_id=tokenizer.pad_token_id, max_seq_len=cfg.tokenizer.seq_len,
                           include_attack_class=True)
    loader = DataLoader(WafDataset(test_path, tokenizer._tok, cfg.tokenizer.seq_len),
                        batch_size=256, shuffle=False, collate_fn=collator, num_workers=2)
    all_preds, all_probs, all_labels, all_classes = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            preds, probs = model.predict(ids, mask)
            all_preds.extend(preds.cpu().tolist()); all_probs.extend(probs.cpu().tolist())
            all_labels.extend(batch["labels"].tolist()); all_classes.extend(batch.get("attack_class",[""]*len(preds)))
    # False positives: label=0, pred=1
    fps = [(i,all_probs[i],all_classes[i]) for i,_ in enumerate(all_labels)
           if all_labels[i]==0 and all_preds[i]==1]
    # False negatives: label=1, pred=0
    fns = [(i,all_probs[i],all_classes[i]) for i,_ in enumerate(all_labels)
           if all_labels[i]==1 and all_preds[i]==0]
    fp_class_dist = dict(Counter(c for _,_,c in fps))
    fn_class_dist = dict(Counter(c for _,_,c in fns))
    log.info(f"FP count: {len(fps):,}  FN count: {len(fns):,}")
    log.info(f"FP class distribution: {fp_class_dist}")
    log.info(f"FN class distribution: {fn_class_dist}")
    # Sample worst FPs and FNs
    fps_sorted = sorted(fps, key=lambda x: x[1], reverse=True)[:10]
    fns_sorted = sorted(fns, key=lambda x: x[1])[:10]
    def get_text(i): return df.iloc[i]["raw"][:200] if i < len(df) else ""
    result = {
        "n_fp": len(fps), "n_fn": len(fns),
        "fp_class_distribution": fp_class_dist,
        "fn_class_distribution": fn_class_dist,
        "worst_fp_examples": [{"text": get_text(i), "prob": round(p,4), "class": c} for i,p,c in fps_sorted],
        "worst_fn_examples": [{"text": get_text(i), "prob": round(p,4), "class": c} for i,p,c in fns_sorted],
    }
    out = Path(cfg.paths.reports)/"metrics"/"error_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log.info(f"Error analysis saved to {out}")
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())