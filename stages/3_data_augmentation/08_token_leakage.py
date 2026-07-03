"""
stages/3_data_augmentation/08_token_leakage.py
----------------------------------------------
Token↔label leakage probe for the augmented corpus.

Answers "is the dataset teaching the model incidental constants (evil.com, 4444)
instead of attack structure?" without needing a trained model:

  1. token_leakage — ranks tokens by how strongly their presence predicts the
     label (P(malicious|token), lift, PMI, mutual information). Tokens flagged
     `incidental` (host/number/id-shaped) that are also strong predictors are the
     red flags.
  2. Counterfactual swap — re-randomises incidental values (swap_fillers) and
     recomputes. If leakage is filler-driven, the top mutual information and the
     count of strong *incidental* predictors drop toward zero; structural
     predictors (SQL keywords, ../, <script>) are untouched.

Writes reports/3_data_augmentation/metrics/08_token_leakage.json.

Run:
    python stages/3_data_augmentation/08_token_leakage.py --config config/pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from ai_waf_v2.eval.leakage import swap_fillers, token_leakage
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import check_output, require_inputs
from ai_waf_v2.utils.reports import report_path

log = get_logger(__name__)


def _load_corpus(path: Path) -> tuple[list[str], list[int]]:
    table = pq.read_table(path, columns=["raw", "label"]).to_pylist()
    texts = [(r.get("raw") or "") for r in table]
    labels = [int(r.get("label") or 0) for r in table]
    return texts, labels


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    # Prefer the filtered corpus; fall back to the train split.
    filtered = Path(cfg.paths.data_filtered) / "filtered.parquet"
    train = Path(cfg.paths.data_splits) / "train.parquet"
    input_path = filtered if filtered.exists() else train

    require_inputs({str(input_path): "make data_filter (or data_split)"})
    out_path = report_path("token_leakage.json", cfg.paths.reports)
    if check_output(out_path, args.force, "Stage 3.8 token-leakage probe"):
        return

    texts, labels = _load_corpus(input_path)
    if not texts or len(set(labels)) < 2:
        log.error("Corpus needs both benign and malicious records for a leakage report.")
        return

    rep = token_leakage(texts, labels, min_df=args.min_df, top_k=args.top_k)

    # Counterfactual: re-randomise incidental values, recompute.
    rng = random.Random(cfg.project.seed)
    swapped = [swap_fillers(t, rng) for t in texts]
    rep_swapped = token_leakage(swapped, labels, min_df=args.min_df, top_k=args.top_k)

    suspected = [s for s in rep.top if s.incidental
                 and (s.p_malicious <= 0.02 or s.p_malicious >= 0.98)]

    result = {
        "input": str(input_path),
        "n_docs": rep.n_docs,
        "n_pos": rep.n_pos,
        "base_rate": rep.base_rate,
        "min_df": rep.min_df,
        "max_mi": rep.max_mi,
        "n_strong": rep.n_strong,
        "n_strong_incidental": rep.n_strong_incidental,
        "counterfactual_swap": {
            "max_mi_after_swap": rep_swapped.max_mi,
            "max_mi_drop": round(rep.max_mi - rep_swapped.max_mi, 6),
            "n_strong_incidental_after": rep_swapped.n_strong_incidental,
        },
        "suspected_shortcuts": [
            {"token": s.token, "df": s.df, "p_malicious": s.p_malicious, "mi": s.mi}
            for s in suspected[: args.top_k]
        ],
        "top_tokens": [
            {"token": s.token, "df": s.df, "p_malicious": s.p_malicious,
             "lift": s.lift, "mi": s.mi, "incidental": s.incidental}
            for s in rep.top
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    log.info(
        f"Token leakage: {rep.n_docs:,} docs (base rate {rep.base_rate}); "
        f"max MI={rep.max_mi:.3f} bits; "
        f"strong predictors={rep.n_strong} (incidental={rep.n_strong_incidental})"
    )
    log.info(
        f"Counterfactual filler swap: max MI {rep.max_mi:.3f} → {rep_swapped.max_mi:.3f} "
        f"(drop {result['counterfactual_swap']['max_mi_drop']:.3f}); "
        f"strong incidental {rep.n_strong_incidental} → {rep_swapped.n_strong_incidental}"
    )
    if suspected:
        preview = ", ".join(f"{s.token}(p={s.p_malicious})" for s in suspected[:10])
        log.warning(
            f"{len(suspected)} incidental token(s) predict the label near-deterministically "
            f"— likely shortcuts to route through §fillers§: {preview}"
        )
    else:
        log.info("No incidental token is a near-deterministic predictor — fillers look label-neutral.")
    log.info(f"Report written to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--min-df", type=int, default=10, help="Ignore tokens rarer than this")
    p.add_argument("--top-k", type=int, default=50, help="Tokens to keep in the report")
    p.add_argument("--force", action="store_true", help="Re-run even if the report exists")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
