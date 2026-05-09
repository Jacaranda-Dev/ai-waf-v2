"""Stage 5.5b — Detailed ORT benchmark across batch sizes and execution providers."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from ai_waf_v2.eval.latency import OnnxLatencyBenchmark, LatencyBenchmark
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    onnx_path = Path(cfg.model.student.output_dir)/"student.onnx"
    if not onnx_path.exists(): log.error(f"ONNX not found: {onnx_path}"); return
    results = {}
    for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]:
        device = "cuda" if "CUDA" in provider else "cpu"
        try:
            bench   = OnnxLatencyBenchmark(onnx_path=onnx_path, device=device,
                                            seq_len=cfg.tokenizer.seq_len,
                                            vocab_size=cfg.model.student.vocab_size,
                                            n_warmup=cfg.evaluation.n_latency_warmup,
                                            n_runs=cfg.evaluation.n_latency_runs)
            res     = bench.run(cfg.evaluation.batch_sizes)
            slo_ok  = bench.check_slo(res, cfg.slo.latency_inline_p99_ms, cfg.slo.throughput_min_rps)
            results[provider] = {"results": res, "slo": slo_ok}
            log.info(f"\nORT {provider}:")
            LatencyBenchmark.print_table(res)
            log.info(f"SLO: {slo_ok}")
        except Exception as e:
            log.warning(f"{provider} failed: {e}")
            results[provider] = {"error": str(e)}
    out = Path(cfg.paths.reports)/"latency"/"ort_detailed_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info(f"ORT detailed benchmark saved to {out}")
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())
