"""Stage 4.2 — Fine-tune a small pretrained model (Track A small)."""
from __future__ import annotations
import argparse
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    # Reuse Track A large logic with small model config
    import importlib, sys, types
    cfg.model.track_a_large.base_model = cfg.model.track_a_small.base_model
    cfg.model.track_a_large.output_dir = cfg.model.track_a_small.output_dir
    from stages.train import o1_track_a_large as m
    m.run(args)
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())
