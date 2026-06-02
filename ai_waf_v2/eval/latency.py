"""
ai_waf_v2.eval.latency
------------------
Hardware latency and throughput benchmarking for WAF models.

Reports p50 / p95 / p99 latency at configurable batch sizes,
on GPU and CPU separately, for all model variants including ONNX
and TensorRT engines.

Usage
-----
    from ai_waf_v2.eval.latency import LatencyBenchmark

    bench = LatencyBenchmark(model, tokenizer, device="cuda", seq_len=256)
    results = bench.run(batch_sizes=[1, 8, 32, 64, 256])
    bench.print_table(results)
    bench.save_json(results, "reports/latency/teacher_gpu.json")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class LatencyBenchmark:
    """
    Measure inference latency and throughput for any nn.Module that
    accepts (input_ids, attention_mask) and returns a dict or tensor.

    Parameters
    ----------
    model       : nn.Module — must already be on `device`
    device      : "cuda" or "cpu"
    seq_len     : int — sequence length used for synthetic inputs
    vocab_size  : int — used to generate random token IDs
    n_warmup    : int — number of warmup passes (not timed)
    n_runs      : int — number of timed passes per batch size
    use_bf16    : bool — wrap forward in autocast(bf16) if True
    """

    def __init__(
        self,
        model:      nn.Module,
        device:     str  = "cuda",
        seq_len:    int  = 256,
        vocab_size: int  = 8000,
        n_warmup:   int  = 50,
        n_runs:     int  = 500,
        use_bf16:   bool = True,
    ) -> None:
        self.model      = model
        self.device     = torch.device(device)
        self.seq_len    = seq_len
        self.vocab_size = vocab_size
        self.n_warmup   = n_warmup
        self.n_runs     = n_runs
        self.use_bf16   = use_bf16 and device == "cuda" and torch.cuda.is_bf16_supported()

        self.model.eval()

    def _make_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Generate a random synthetic batch on the target device."""
        return {
            "input_ids": torch.randint(
                1, self.vocab_size,
                (batch_size, self.seq_len),
                device=self.device,
            ),
            "attention_mask": torch.ones(
                batch_size, self.seq_len,
                dtype=torch.long,
                device=self.device,
            ),
        }

    def _forward(self, batch: dict[str, torch.Tensor]) -> None:
        ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if self.use_bf16
            else torch.amp.autocast("cuda", enabled=False)
        )
        with torch.no_grad(), ctx:
            self.model(**batch)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def run(self, batch_sizes: list[int] | None = None) -> list[dict[str, Any]]:
        """
        Run the benchmark for each batch size.

        Returns
        -------
        List of result dicts, one per batch size:
            {batch_size, n_runs, p50_ms, p95_ms, p99_ms,
             mean_ms, throughput_rps, device, model_params}
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 32, 64, 256]

        results = []

        for bs in batch_sizes:
            batch = self._make_batch(bs)

            # Warmup
            for _ in range(self.n_warmup):
                self._forward(batch)

            # Timed runs
            latencies_ms: list[float] = []
            for _ in range(self.n_runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                self._forward(batch)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

            lat = torch.tensor(latencies_ms)
            p50 = lat.quantile(0.50).item()
            p95 = lat.quantile(0.95).item()
            p99 = lat.quantile(0.99).item()
            mean = lat.mean().item()
            throughput = bs / (mean / 1000.0)  # requests/sec

            n_params = (
                sum(p.numel() for p in self.model.parameters())
                if self.model is not None else 0
            )

            result = {
                "batch_size":     bs,
                "n_runs":         self.n_runs,
                "p50_ms":         round(p50,       3),
                "p95_ms":         round(p95,       3),
                "p99_ms":         round(p99,       3),
                "mean_ms":        round(mean,      3),
                "throughput_rps": round(throughput, 1),
                "device":         str(self.device),
                "bf16":           self.use_bf16,
                "model_params":   n_params,
            }
            results.append(result)

        return results

    def check_slo(
        self,
        results:           list[dict[str, Any]],
        inline_p99_ms:     float = 5.0,
        throughput_min:    int   = 10_000,
    ) -> dict[str, bool]:
        """
        Check whether results satisfy WAF SLOs.

        Returns
        -------
        dict with bool values for each SLO check.
        """
        batch1   = next((r for r in results if r["batch_size"] == 1), None)
        batch64  = next((r for r in results if r["batch_size"] == 64), None)

        return {
            "inline_p99_ok": (
                batch1 is not None
                and batch1["p99_ms"] <= inline_p99_ms
            ),
            "throughput_ok": (
                batch64 is not None
                and batch64["throughput_rps"] >= throughput_min
            ),
        }

    @staticmethod
    def print_table(results: list[dict[str, Any]]) -> None:
        """Print a formatted latency table to stdout."""
        header = f"{'BS':>6}  {'p50 ms':>8}  {'p95 ms':>8}  {'p99 ms':>8}  {'RPS':>10}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r['batch_size']:>6}  "
                f"{r['p50_ms']:>8.2f}  "
                f"{r['p95_ms']:>8.2f}  "
                f"{r['p99_ms']:>8.2f}  "
                f"{r['throughput_rps']:>10.0f}"
            )

    @staticmethod
    def save_json(results: list[dict[str, Any]], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(results, f, indent=2)


class OnnxLatencyBenchmark(LatencyBenchmark):
    """
    Latency benchmark for ONNX Runtime sessions.

    Replaces the PyTorch model forward with an ORT session run.
    """

    def __init__(
        self,
        onnx_path:  str | Path,
        device:     str  = "cuda",
        seq_len:    int  = 256,
        vocab_size: int  = 8000,
        n_warmup:   int  = 50,
        n_runs:     int  = 500,
    ) -> None:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session    = ort.InferenceSession(str(onnx_path), providers=providers)
        # Verify the requested provider is actually active; ORT silently falls
        # back to CPU when e.g. onnxruntime-gpu is not installed, which would
        # cause the benchmark to hang running a Transformer on CPU.
        if device == "cuda":
            active = self.session.get_providers()
            if "CUDAExecutionProvider" not in active:
                raise RuntimeError(
                    "CUDAExecutionProvider not available — install onnxruntime-gpu. "
                    f"Active providers: {active}"
                )
        self.model      = None   # no nn.Module for ORT sessions
        self.device     = torch.device(device)
        self.seq_len    = seq_len
        self.vocab_size = vocab_size
        self.n_warmup   = n_warmup
        self.n_runs     = n_runs
        self.use_bf16   = False  # ORT handles precision internally

    def _make_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        # Return CPU tensors (ORT expects numpy-compatible)
        return {
            "input_ids": torch.randint(
                1, self.vocab_size, (batch_size, self.seq_len)
            ),
            "attention_mask": torch.ones(
                batch_size, self.seq_len, dtype=torch.long
            ),
        }

    def _forward(self, batch: dict[str, torch.Tensor]) -> None:
        self.session.run(
            None,
            {
                "input_ids":      batch["input_ids"].numpy(),
                "attention_mask": batch["attention_mask"].numpy(),
            },
        )