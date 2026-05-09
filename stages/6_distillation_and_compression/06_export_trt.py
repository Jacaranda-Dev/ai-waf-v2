"""Stage 5.5 — TensorRT engine build from ONNX student model."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
log = get_logger(__name__)
def run(args):
    configure_root(); cfg = load_config(args.config)
    onnx_path = Path(cfg.model.student.output_dir)/"student.onnx"
    trt_path  = Path(cfg.model.student.output_dir)/"student.engine"
    if not onnx_path.exists(): log.error(f"ONNX model not found: {onnx_path}"); return
    try:
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser  = trt.OnnxParser(network, TRT_LOGGER)
        config  = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16); log.info("TRT: fp16 enabled")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()): log.error("ONNX parse failed"); return
        engine_bytes = builder.build_serialized_network(network, config)
        with open(trt_path, "wb") as f: f.write(engine_bytes)
        log.info(f"TRT engine saved to {trt_path}")
        # Quick latency check
        import time, numpy as np
        runtime = trt.Runtime(TRT_LOGGER)
        engine  = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()
        seq_len = cfg.tokenizer.seq_len
        dummy_ids  = np.ones((1, seq_len), dtype=np.int64)
        dummy_mask = np.ones((1, seq_len), dtype=np.int64)
        import pycuda.driver as cuda, pycuda.autoinit
        d_ids  = cuda.mem_alloc(dummy_ids.nbytes);  cuda.memcpy_htod(d_ids,  dummy_ids)
        d_mask = cuda.mem_alloc(dummy_mask.nbytes); cuda.memcpy_htod(d_mask, dummy_mask)
        out_size = 1 * 2; d_out = cuda.mem_alloc(out_size * 4)
        lats = []
        for _ in range(100):
            t0 = time.perf_counter()
            context.execute_v2(bindings=[int(d_ids), int(d_mask), int(d_out)])
            lats.append((time.perf_counter()-t0)*1000)
        import numpy as np
        log.info(f"TRT p99 latency (bs=1): {np.percentile(lats,99):.2f}ms")
        result = {"trt_engine": str(trt_path), "p99_ms_bs1": round(float(np.percentile(lats,99)),3)}
        out = Path(cfg.paths.reports)/"latency"/"trt_bench.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
    except ImportError:
        log.warning("TensorRT / pycuda not installed — skipping TRT export. "
                    "Install: pip install tensorrt pycuda")
def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="config/pipeline.yaml"); return p.parse_args()
if __name__ == "__main__": run(parse_args())
