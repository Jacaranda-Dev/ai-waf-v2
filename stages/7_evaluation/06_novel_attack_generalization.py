"""Stage 6.4c — Evaluate generalisation to novel attack families not in training data."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import pyarrow.parquet as pq
from ai_waf_v2.data.schema import HttpRecord
from ai_waf_v2.eval.metrics import compute_metrics
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
# Novel attack patterns NOT typically in training data
NOVEL_ATTACKS = {
    "graphql_injection": [
        "GET /graphql?query={user(id:%221%22){name,email}}", 
        "POST /graphql body: {\"query\":\"{__schema{types{name}}}\"}",
        "GET /graphql?query={__typename%20mutation{deleteUser(id:1)}}",
    ],
    "jwt_manipulation": [
        "GET /api/user Authorization: Bearer eyJhbGciOiJub25lIn0.eyJ1c2VyIjoiYWRtaW4ifQ.",
        "GET /api/admin Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJyb2xlIjoiYWRtaW4ifQ.fake",
    ],
    "http2_desync": [
        "POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 4\r\n\r\n0\r\n\r\nGET /admin HTTP/1.1",
    ],
    "prototype_pollution": [
        "POST /api/merge body: {\"__proto__\":{\"admin\":true}}",
        "GET /api/user?__proto__[admin]=true",
    ],
}
def run(args):
    configure_root(); cfg = load_config(args.config)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = HttpTokenizer.load(cfg.tokenizer.track_b.output_dir, cfg.tokenizer.seq_len)
    ckpt = Path(cfg.model.track_b_99m.output_dir)/"best_99m.pt"
    if not ckpt.exists(): log.error("Teacher checkpoint not found"); return
    from ai_waf_v2.models.head import WafClassifier
    model = WafClassifier.load(ckpt, cfg.model.track_b_99m, map_location=str(device))
    model.to(device).eval()
    results = {}
    for attack_name, raw_reqs in NOVEL_ATTACKS.items():
        records = [HttpRecord(method="GET", path="/api/test", raw=raw,
                               label=1, attack_class=attack_name, source="novel_eval").build_raw()
                   for raw in raw_reqs]
        all_preds, all_probs = [], []
        for r in records:
            enc = tokenizer.encode(r.raw)
            ids  = torch.tensor([enc.ids[:cfg.tokenizer.seq_len] +
                   [tokenizer.pad_token_id]*(cfg.tokenizer.seq_len-len(enc.ids[:cfg.tokenizer.seq_len]))]).to(device)
            mask = torch.tensor([[1]*min(len(enc.ids),cfg.tokenizer.seq_len) +
                   [0]*(cfg.tokenizer.seq_len-min(len(enc.ids),cfg.tokenizer.seq_len))]).to(device)
            with torch.no_grad():
                preds, probs = model.predict(ids, mask)
            all_preds.extend(preds.cpu().tolist()); all_probs.extend(probs.cpu().tolist())
        detection_rate = sum(all_preds)/max(1,len(all_preds))
        results[attack_name] = {"n_samples": len(records),
                                 "detection_rate": round(detection_rate,4),
                                 "evasion_rate": round(1-detection_rate,4)}
        log.info(f"  {attack_name:30s}: detection={detection_rate:.4f}")
    out = Path(cfg.paths.reports)/"metrics"/"novel_attack_generalization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info(f"Novel attack generalisation saved to {out}")
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())
