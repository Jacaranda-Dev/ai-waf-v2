from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class StepTimer:
    """
    Thread-safe wall-clock timer for named pipeline steps.

    Usage::

        timer = StepTimer()

        with timer.step("gap_analysis"):
            gaps = governor.gaps()

        with timer.step("synthesis_xss"):
            records = synthesize(...)

        timer.save("reports/metrics/timings_01_attack_synthesis.json")
        timer.log_mlflow()   # logs timing_{name}_s for each step

    Steps recorded from multiple threads are safe — a lock serialises
    dict writes.  Nested steps with the same name overwrite the earlier
    value, so use unique names per step.
    """

    def __init__(self) -> None:
        self._timings: dict[str, float] = {}
        self._lock = threading.Lock()

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = round(time.perf_counter() - t0, 3)
            with self._lock:
                self._timings[name] = elapsed

    @property
    def timings(self) -> dict[str, float]:
        with self._lock:
            return dict(self._timings)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.timings, indent=2))

    def log_mlflow(self) -> None:
        try:
            import mlflow
            for name, elapsed in self.timings.items():
                mlflow.log_metric(f"timing_{name}_s", elapsed)
        except Exception:
            pass
