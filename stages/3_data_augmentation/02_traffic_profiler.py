"""
stages/3_data_augmentation/02_traffic_profiler.py
-------------------------------------------------
Module B — Traffic Distribution Profiler

Architecture:
  - Extracts realistic metadata distributions (method mix, UA fingerprints,
    Accept/Content-Type headers, auth/referer rates) from CAIDA PCAP traces
  - Writes traffic_distribution.json for Module C (03_request_framing.py),
    which applies those distributions when framing BOTH attack and benign records
  - Falls back to an empty profile when no PCAP is available;
    Module C then uses its own internal defaults

Run:
    python stages/3_data_augmentation/02_traffic_profiler.py \
        --config config/pipeline.yaml [--pcap-dir /path/to/caida]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import check_output
from ai_waf_v2.utils.timing import StepTimer

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PCAP metadata extractor
# ─────────────────────────────────────────────────────────────────────────────

class PcapMetadataExtractor:
    """
    Parses HTTP flows from PCAP/NFCAPD files and extracts statistical
    distributions over headers and path structure for use in benign generation.

    Supports dpkt (preferred) and scapy (fallback).
    Gracefully stubs when neither is available.
    """

    def __init__(self, pcap_dir: Path, max_packets: int = 100_000):
        self.pcap_dir    = pcap_dir
        self.max_packets = max_packets

    def _extract_dpkt(self, pcap_path: Path) -> list[dict]:
        """Extract HTTP request dicts using dpkt."""
        import dpkt
        flows = []
        with open(pcap_path, "rb") as f:
            try:
                pcap = dpkt.pcap.Reader(f)
            except Exception:
                pcap = dpkt.pcapng.Reader(f)

            for _, buf in pcap:
                if len(flows) >= self.max_packets:
                    break
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    tcp = eth.data.data
                    if not isinstance(tcp, dpkt.tcp.TCP) or not tcp.data:
                        continue
                    http = dpkt.http.Request(tcp.data)
                    flows.append({
                        "method":     http.method,
                        "uri":        http.uri,
                        "headers":    dict(http.headers),
                        "body":       http.body.decode("utf-8", errors="replace") if http.body else "",
                    })
                except Exception:
                    continue
        return flows

    def extract(self) -> list[dict]:
        """Try dpkt, then scapy, then return empty list with a warning."""
        if not self.pcap_dir or not self.pcap_dir.exists():
            return []

        pcap_files = list(self.pcap_dir.glob("*.pcap")) + list(self.pcap_dir.glob("*.pcapng"))
        if not pcap_files:
            log.warning(f"No PCAP files found in {self.pcap_dir}")
            return []

        all_flows: list[dict] = []
        for f in pcap_files:
            try:
                flows = self._extract_dpkt(f)
                all_flows.extend(flows)
                log.info(f"  Extracted {len(flows):,} HTTP flows from {f.name}")
            except ImportError:
                log.warning("dpkt not installed — pip install dpkt; PCAP extraction skipped")
                break
            except Exception as e:
                log.warning(f"  Failed to parse {f.name}: {e}")

        log.info(f"Total flows extracted from PCAP: {len(all_flows):,}")
        return all_flows


# ─────────────────────────────────────────────────────────────────────────────
# Distribution builder
# ─────────────────────────────────────────────────────────────────────────────

class TrafficDistribution:
    """
    Computes and holds observed distributions from PCAP flows.
    Can be serialised to JSON for Module B to consume.
    """

    def __init__(self):
        self.user_agents:    dict[str, float] = {}
        self.methods:        dict[str, float] = {}
        self.path_depths:    dict[int, float] = {}   # path segment count → freq
        self.content_types:  dict[str, float] = {}
        self.accept_headers: dict[str, float] = {}
        self.has_auth:       float = 0.0              # fraction with Authorization header
        self.has_referer:    float = 0.0

    @classmethod
    def from_flows(cls, flows: list[dict]) -> "TrafficDistribution":
        d = cls()
        if not flows:
            return d

        ua_counts  = collections.Counter()
        meth_counts = collections.Counter()
        depth_counts = collections.Counter()
        ct_counts  = collections.Counter()
        accept_counts = collections.Counter()
        n_auth = n_ref = 0

        for flow in flows:
            h = {k.lower(): v for k, v in flow.get("headers", {}).items()}
            ua_counts[h.get("user-agent", "unknown")] += 1
            meth_counts[flow.get("method", "GET")] += 1
            uri   = flow.get("uri", "/")
            depth = max(0, uri.count("/") - 1)
            depth_counts[depth] += 1
            ct_counts[h.get("content-type", "")] += 1
            accept_counts[h.get("accept", "")] += 1
            if "authorization" in h: n_auth += 1
            if "referer" in h:       n_ref  += 1

        def _Normalize(counter: collections.Counter) -> dict:
            total = sum(counter.values()) or 1
            return {k: round(v / total, 6) for k, v in counter.most_common(50)}

        n = len(flows) or 1
        d.user_agents    = _Normalize(ua_counts)
        d.methods        = _Normalize(meth_counts)
        d.path_depths    = {k: round(v / n, 6) for k, v in depth_counts.most_common(10)}
        d.content_types  = _Normalize(ct_counts)
        d.accept_headers = _Normalize(accept_counts)
        d.has_auth       = round(n_auth / n, 4)
        d.has_referer    = round(n_ref  / n, 4)
        return d

    def to_dict(self) -> dict:
        return {
            "user_agents":    self.user_agents,
            "methods":        self.methods,
            "path_depths":    self.path_depths,
            "content_types":  self.content_types,
            "accept_headers": self.accept_headers,
            "has_auth":       self.has_auth,
            "has_referer":    self.has_referer,
        }

    def sample_method(self, rng: random.Random) -> str:
        if not self.methods:
            return rng.choice(["GET", "POST"])
        population = list(self.methods.keys())
        weights    = list(self.methods.values())
        return rng.choices(population, weights=weights, k=1)[0]

    def sample_ua(self, rng: random.Random, fallback: list[str]) -> str:
        if not self.user_agents:
            return rng.choice(fallback)
        population = list(self.user_agents.keys())
        weights    = list(self.user_agents.values())
        return rng.choices(population, weights=weights, k=1)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Aligned benign record generator
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)

    dist_path = Path(cfg.paths.reports) / "metrics" / "traffic_distribution.json"

    if check_output(dist_path, args.force, "Stage 3.2 benign enrichment"):
        return

    benign_cfg = getattr(cfg.augmentation, "benign", None)
    replay_on  = getattr(benign_cfg, "replay_enabled", False) if benign_cfg else False
    pcap_dir   = Path(args.pcap_dir or (getattr(benign_cfg, "replay_path", "") if benign_cfg else ""))

    dist_path.parent.mkdir(parents=True, exist_ok=True)

    timer = StepTimer()

    # ── Extract PCAP distributions ────────────────────────────────────────
    flows: list[dict] = []
    if replay_on and pcap_dir.exists():
        log.info(f"Extracting metadata distributions from PCAP traces: {pcap_dir}")
        extractor = PcapMetadataExtractor(pcap_dir, max_packets=200_000)
        with timer.step("pcap_extract"):
            flows = extractor.extract()
    else:
        log.info("PCAP replay disabled or path not found — using internal default distributions")

    with timer.step("distribution_fit"):
        dist = TrafficDistribution.from_flows(flows)
    dist_path.write_text(json.dumps(dist.to_dict(), indent=2))
    log.info(f"Distribution profile written → {dist_path}")

    stats = {
        "n_pcap_flows": len(flows),
        "pcap_dir":     str(pcap_dir) if pcap_dir else None,
        "distribution": dist.to_dict(),
        "timings_s":    timer.timings,
    }
    sp = Path(cfg.paths.reports) / "metrics" / "traffic_profiler.json"
    sp.write_text(json.dumps(stats, indent=2))

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="02_traffic_profiler"):
            mlflow.log_params({
                "replay_enabled": replay_on,
                "pcap_dir":       str(pcap_dir) if pcap_dir else "none",
            })
            log_metrics_dict({
                "n_pcap_flows": float(len(flows)),
            })
            timer.log_mlflow()
            mlflow.log_artifact(str(sp))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="config/pipeline.yaml")
    p.add_argument("--pcap-dir", default=None, help="Path to directory containing PCAP files")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if outputs already exist")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())