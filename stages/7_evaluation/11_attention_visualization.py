"""
stages/7_evaluation/11_attention_visualization.py
-------------------------------------------
Stage 7.11 — Visualise attention weights on representative malicious samples.

For each attack class, shows which token positions the model attends to
in the final encoder layer. Saves heatmap data as JSON (render with
any plotting tool — matplotlib, plotly, seaborn).

Run:
    python stages/7_evaluation/11_attention_visualization.py --config config/pipeline.yaml
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import pyarrow.parquet as pq

from ai_waf_v2.models.encoder import WafEncoder
from ai_waf_v2.models.head import WafClassifier
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)

SAMPLES_PER_CLASS = 3   # keep small — we're logging full attention matrices


class AttentionExtractor:
    """
    Register forward hooks on the last encoder layer's attention module
    to capture attention weight matrices during inference.
    """

    def __init__(self, model: WafClassifier) -> None:
        self.model    = model
        self.weights: list[torch.Tensor] = []
        self._hook    = None
        self._register()

    def _register(self) -> None:
        # Hook the last encoder layer's attention
        last_layer = self.model.encoder.layers[-1].attn

        def hook(module, inp, output):
            # output is the projected hidden, but we need the raw attention weights
            # We need to recompute them — hook into the SDPA call
            pass

        # Alternative approach: patch the forward to capture weights
        orig_forward = last_layer.forward

        def patched_forward(hidden, attention_mask):
            B, T, D = hidden.shape
            H, Dh   = last_layer.n_heads, last_layer.head_dim

            Q = last_layer.q_proj(hidden).view(B, T, H, Dh).transpose(1, 2)
            K = last_layer.k_proj(hidden).view(B, T, H, Dh).transpose(1, 2)
            V = last_layer.v_proj(hidden).view(B, T, H, Dh).transpose(1, 2)

            import math
            scale = math.sqrt(Dh)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B,H,T,T)

            key_mask  = attention_mask[:, None, None, :]
            attn_bias = torch.zeros_like(attn_scores)
            attn_bias = attn_bias.masked_fill(key_mask == 0, float("-inf"))
            attn_scores = attn_scores + attn_bias

            weights = torch.softmax(attn_scores, dim=-1)  # (B,H,T,T)
            self.weights.append(weights.detach().cpu())

            # Complete the forward pass
            out = torch.matmul(weights, V)
            out = out.transpose(1, 2).contiguous().view(B, T, D)
            return last_layer.o_proj(out)

        last_layer.forward = patched_forward

    def clear(self) -> None:
        self.weights.clear()


def _render_heatmaps(results: list[dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        log.warning("matplotlib not installed — skipping attention heatmap rendering")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Group by attack class
    by_class: dict[str, list[dict]] = {}
    for r in results:
        by_class.setdefault(r["attack_class"], []).append(r)

    for cls, samples in by_class.items():
        n = len(samples)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), squeeze=False)
        fig.suptitle(f"[CLS] Attention — {cls}", fontsize=13, fontweight="bold")

        for i, (ax, sample) in enumerate(zip(axes[:, 0], samples)):
            tokens = sample["tokens"]
            weights = sample["cls_attention"]
            if not tokens or not weights:
                ax.set_visible(False)
                continue

            colors = ["#d62728" if w == max(weights) else "#1f77b4" for w in weights]
            ax.bar(range(len(tokens)), weights, color=colors)
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(0, max(weights) * 1.25 if max(weights) > 0 else 1.0)
            ax.set_ylabel("Attention", fontsize=8)
            ax.set_title(
                f"pred={'malicious' if sample['pred'] else 'benign'}  "
                f"p={sample['prob_malicious']:.3f}",
                fontsize=9, loc="left",
            )

        fig.tight_layout()
        path = out_dir / f"{cls}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved {path}")

    log.info(f"Attention heatmaps written to {out_dir}/")


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg    = load_config(args.config)
    require_inputs({
        f"{cfg.model.track_b_99m.output_dir}/best_99m.pt": "run 00_train_teacher_99m.py",
        "data/splits/test.parquet": "make data_augment_all",
    })
    if check_output(
        Path(cfg.paths.reports) / "metrics" / "attention_visualization.json",
        args.force, "Stage 7.11 attention visualization"
    ):
        return
    device = torch.device("cpu")   # attention viz on CPU for simplicity

    tokenizer = HttpTokenizer.load(
        cfg.tokenizer.track_b.output_dir, cfg.tokenizer.seq_len
    )

    ckpt = Path(cfg.model.track_b_99m.output_dir) / "best_99m.pt"
    if not ckpt.exists():
        log.error(f"Checkpoint not found: {ckpt}")
        return

    model = WafClassifier.load(ckpt, cfg.model.track_b_99m, map_location="cpu")
    model.eval()

    extractor = AttentionExtractor(model)
    timer = StepTimer()

    # Load one sample per attack class from test split
    test_path = Path(cfg.paths.data_splits) / "test.parquet"
    if not test_path.exists():
        log.error("test.parquet not found")
        return

    df = pq.read_table(test_path, columns=["raw","label","attack_class"]).to_pandas()
    attack_classes = [c for c in df["attack_class"].unique() if c != "benign"]

    results: list[dict[str, Any]] = []

    with timer.step("extract_attention"):
        for cls in attack_classes[:6]:   # cap at 6 classes
            samples = df[df["attack_class"] == cls].head(SAMPLES_PER_CLASS)
            for _, row in samples.iterrows():
                text  = row["raw"]
                label = int(row["label"])
                enc   = tokenizer.encode(text)
                ids   = enc.ids[:cfg.tokenizer.seq_len]
                toks  = enc.tokens[:cfg.tokenizer.seq_len]
                pad_len = cfg.tokenizer.seq_len - len(ids)
                pad_id  = tokenizer.pad_token_id

                input_ids      = torch.tensor([ids + [pad_id]*pad_len], dtype=torch.long)
                attention_mask = torch.tensor([[1]*len(ids) + [0]*pad_len], dtype=torch.long)

                extractor.clear()
                with torch.no_grad():
                    out = model(input_ids, attention_mask)
                    pred   = torch.argmax(out["logits"], dim=-1).item()
                    prob   = torch.softmax(out["logits"], dim=-1)[0, 1].item()

                if extractor.weights:
                    attn_matrix = extractor.weights[0]  # (1, H, T, T)
                    # Average over heads; take CLS row (position 0)
                    cls_attn = attn_matrix[0].mean(0)[0, :len(toks)].tolist()
                else:
                    cls_attn = []

                # Find top-5 attended tokens
                top5 = sorted(enumerate(cls_attn), key=lambda x: x[1], reverse=True)[:5]

                results.append({
                    "attack_class":    cls,
                    "label":           label,
                    "pred":            pred,
                    "prob_malicious":  round(prob, 4),
                    "tokens":          toks[:20],        # first 20 tokens
                    "cls_attention":   cls_attn[:20],    # CLS attention to first 20 positions
                    "top5_attended":   [(toks[i] if i < len(toks) else "PAD", round(w, 4))
                                        for i, w in top5],
                })

                log.info(
                    f"  {cls:20s}: pred={pred} prob={prob:.3f}  "
                    f"top token='{results[-1]['top5_attended'][0][0] if results[-1]['top5_attended'] else '?'}'"
                )

    out = Path(cfg.paths.reports) / "metrics" / "attention_visualization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info(f"Attention visualization data saved to {out}")

    if not args.no_render:
        _render_heatmaps(results, out.parent / "attention_heatmaps")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import mlflow_run, log_metrics_dict
        with mlflow_run(cfg, run_name="11_attention_visualization") as _run:
            mlflow.log_params({
                "samples_per_class": SAMPLES_PER_CLASS,
                "n_attack_classes":  len(attack_classes[:6]),
                "n_samples_total":   len(results),
            })
            n_correct = sum(1 for r in results if r.get("pred") == r.get("label"))
            log_metrics_dict({
                "n_samples":         float(len(results)),
                "n_correct_pred":    float(n_correct),
                "accuracy":          float(n_correct / max(1, len(results))),
            })
            mlflow.log_artifact(str(out))
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    p.add_argument("--no-render", action="store_true",
                   help="Skip matplotlib heatmap generation")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())