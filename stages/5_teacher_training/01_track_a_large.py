"""Stage 4.1 — Fine-tune DeBERTa-v3-base (Track A large) on WAF data."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
from ai_waf_v2.data.dataset import get_split_path
from ai_waf_v2.eval.metrics import compute_metrics, find_threshold_at_fpr
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.mlflow_utils import init_experiment
from ai_waf_v2.utils.seed import seed_everything
log = get_logger(__name__, log_file="reports/train_track_a_large.log")

class _HFDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_path, tokenizer, seq_len=256):
        import pyarrow.parquet as pq
        t = pq.read_table(parquet_path, columns=["raw","label","attack_class"])
        self._texts   = t["raw"].to_pylist()
        self._labels  = t["label"].to_pylist()
        self._classes = t["attack_class"].to_pylist()
        self.tokenizer = tokenizer; self.seq_len = seq_len
    def __len__(self): return len(self._texts)
    def __getitem__(self, i):
        enc = self.tokenizer(self._texts[i], truncation=True, max_length=self.seq_len,
                             padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self._labels[i], dtype=torch.long),
                "attack_class": self._classes[i]}

def run(args):
    configure_root(); cfg = load_config(args.config); seed_everything(cfg.project.seed)
    tcfg = cfg.training.teacher; mcfg = cfg.model.track_a_large
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(mcfg.base_model)
    model     = AutoModelForSequenceClassification.from_pretrained(
        mcfg.base_model, num_labels=mcfg.num_labels, ignore_mismatched_sizes=True)
    model.to(device)
    if torch.__version__ >= "2.0" and device.type == "cuda":
        model = torch.compile(model)
    splits_dir = cfg.paths.data_splits
    train_ds = _HFDataset(get_split_path(splits_dir,"train"), tokenizer, cfg.tokenizer.seq_len)
    val_ds   = _HFDataset(get_split_path(splits_dir,"val"),   tokenizer, cfg.tokenizer.seq_len)
    def collate(batch):
        return {"input_ids":      torch.stack([b["input_ids"]      for b in batch]),
                "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                "labels":         torch.stack([b["labels"]         for b in batch]),
                "attack_class":   [b["attack_class"]               for b in batch]}
    train_loader = DataLoader(train_ds, tcfg.batch_size, shuffle=True,  collate_fn=collate, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   tcfg.batch_size*2, shuffle=False, collate_fn=collate, num_workers=2, pin_memory=True)
    no_decay = {"bias","LayerNorm.weight"}
    params = [{"params":[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay":tcfg.weight_decay},
              {"params":[p for n,p in model.named_parameters() if     any(nd in n for nd in no_decay)],"weight_decay":0.0}]
    optimizer = AdamW(params, lr=tcfg.peak_lr, betas=(tcfg.adam_beta1,tcfg.adam_beta2))
    scheduler = get_cosine_schedule_with_warmup(optimizer, tcfg.warmup_steps, tcfg.max_steps)
    use_bf16  = tcfg.precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast  = torch.amp.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.amp.autocast("cuda", enabled=False)
    out_dir = Path(mcfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    init_experiment(cfg); best_auc = 0.0; step = 0; global_step = 0
    model.train(); optimizer.zero_grad(); it = iter(train_loader)
    with mlflow.start_run(run_name=f"track_a_large_{int(time.time())}", tags=cfg.mlflow.tags):
        mlflow.log_params({"model": mcfg.base_model, "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        while step < tcfg.max_steps:
            try: batch = next(it)
            except StopIteration: it = iter(train_loader); batch = next(it)
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device); labels = batch["labels"].to(device)
            with autocast:
                out  = model(input_ids=ids, attention_mask=mask, labels=labels)
                loss = out.loss / tcfg.grad_accum_steps
            loss.backward()
            if (step+1) % tcfg.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip_norm)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(); global_step += 1
                if global_step % tcfg.eval_every_steps == 0:
                    model.eval(); all_p,all_pr,all_l = [],[],[]
                    with torch.no_grad():
                        for vb in val_loader:
                            vids = vb["input_ids"].to(device); vmask = vb["attention_mask"].to(device)
                            with autocast: logits = model(input_ids=vids, attention_mask=vmask).logits
                            probs = torch.softmax(logits,-1)[:,1]
                            preds = (probs >= 0.5).long()
                            all_p.append(preds.cpu()); all_pr.append(probs.cpu()); all_l.append(vb["labels"])
                    m = compute_metrics(torch.cat(all_p), torch.cat(all_pr), torch.cat(all_l))
                    mlflow.log_metrics({f"val/{k}": v for k,v in m.items() if isinstance(v,float)}, step=global_step)
                    log.info(f"step={global_step} F1={m[\'f1\']:.4f} AUC-PR={m[\'auc_pr\']:.4f}")
                    if m["auc_pr"] > best_auc:
                        best_auc = m["auc_pr"]
                        model.save_pretrained(str(out_dir/"best_model"))
                        tokenizer.save_pretrained(str(out_dir/"best_model"))
                        log.info(f"  ✓ best checkpoint (auc_pr={best_auc:.4f})")
                    model.train()
            step += 1
    model.save_pretrained(str(out_dir/"final_model"))
    log.info(f"Track A large training complete. Best AUC-PR: {best_auc:.4f}")
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())
