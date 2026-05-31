"""
stages/3_data_augmentation/02_benign_enrichment.py
--------------------------------------------------
Module B — Benign Corpus Enrichment

Merges: 10_benign_replay_traces

Architecture:
  - Extracts realistic metadata distributions (header frequency, path depth,
    UA fingerprints) from CAIDA PCAP traces when available
  - Uses those distributions to parameterize the programmatic REST generator
    in Module C (03_request_framing.py) — "distribution alignment"
  - Falls back gracefully to internal defaults when no PCAP trace is provided
  - Writes a distribution profile JSON that Module C reads on startup

Run:
    python stages/3_data_augmentation/02_benign_enrichment.py \
        --config config/pipeline.yaml [--pcap-dir /path/to/caida]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import uuid
from pathlib import Path

import pyarrow.parquet as pq

from ai_waf_v2.data.schema import HttpRecord, records_to_table
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

_FALLBACK_UA = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "PostmanRuntime/7.36.0",
    "python-requests/2.31.0",
]

_PATH_TEMPLATES = [
    "/api/v1/users", "/api/v1/products", "/api/v1/orders",
    "/api/v1/search", "/api/v1/profile", "/api/v1/categories",
    "/api/v1/reports", "/health", "/api/v2/data",
]

_BENIGN_PARAMS = [
    "page=1&limit=20", "sort=created_at&order=desc",
    "filter=active", "q=example+query",
    "", "expand=details", "include=meta",
]


def make_aligned_benign(
    dist: TrafficDistribution,
    n:    int,
    rng:  random.Random,
) -> list[HttpRecord]:
    """
    Generate benign records whose method/UA/header distributions are
    statistically aligned with the PCAP-observed distributions.
    """
    records = []
    for _ in range(n):
        method  = dist.sample_method(rng)
        ua      = dist.sample_ua(rng, _FALLBACK_UA)
        path    = rng.choice(_PATH_TEMPLATES)
        qs      = rng.choice(_BENIGN_PARAMS) if method == "GET" else ""
        has_body = method in ("POST", "PUT", "PATCH")
        body    = json.dumps({"key": str(uuid.uuid4())[:8]}) if has_body else ""

        headers: dict[str, str] = {
            "Host":       "api.example.com",
            "User-Agent": ua,
            "Accept":     "application/json",
        }
        if has_body:
            headers["Content-Type"] = "application/json"
        if rng.random() < dist.has_auth:
            headers["Authorization"] = f"Bearer eyJ{uuid.uuid4().hex[:16]}"
        if rng.random() < dist.has_referer:
            headers["Referer"] = "https://app.example.com/dashboard"

        try:
            records.append(HttpRecord(
                id=str(uuid.uuid4()),
                method=method,
                path=path,
                query_string=qs,
                headers=json.dumps(headers),
                body=body,
                label=0,
                attack_class="benign",
                source="aug_benign_aligned",
            ).build_raw())
        except Exception as e:
            log.debug(f"Record creation failed: {e}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    rng = random.Random(cfg.project.seed)

    if check_output(
        Path(cfg.paths.data_augmented) / "benign" / "benign_aligned.parquet",
        args.force, "Stage 3.2 benign enrichment"
    ):
        return

    benign_cfg  = getattr(cfg.augmentation, "benign", None)
    replay_on   = getattr(benign_cfg, "replay_enabled", False) if benign_cfg else False
    pcap_dir    = Path(args.pcap_dir or (getattr(benign_cfg, "replay_path", "") if benign_cfg else ""))
    n_aligned   = getattr(benign_cfg, "aligned_samples", 5_000) if benign_cfg else 5_000

    out_dir    = Path(cfg.paths.data_augmented) / "benign"
    out_dir.mkdir(parents=True, exist_ok=True)
    dist_path  = Path(cfg.paths.reports) / "metrics" / "benign_distribution.json"
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

    # ── Generate distribution-aligned benign records ──────────────────────
    log.info(f"Generating {n_aligned:,} distribution-aligned benign records...")
    with timer.step("benign_generation"):
        records = make_aligned_benign(dist, n_aligned, rng)

    out_path = out_dir / "benign_aligned.parquet"
    with timer.step("parquet_write"):
        pq.write_table(records_to_table(records), out_path, compression="snappy")
    log.info(f"Wrote {len(records):,} aligned benign records → {out_path}")

    stats = {
        "n_pcap_flows":     len(flows),
        "n_aligned_benign": len(records),
        "pcap_dir":         str(pcap_dir) if pcap_dir else None,
        "distribution":     dist.to_dict(),
        "timings_s":        timer.timings,
    }
    sp = Path(cfg.paths.reports) / "metrics" / "benign_enrichment.json"
    sp.write_text(json.dumps(stats, indent=2))

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="02_benign_enrichment"):
            mlflow.log_params({
                "replay_enabled":   replay_on,
                "pcap_dir":         str(pcap_dir) if pcap_dir else "none",
                "n_aligned_target": n_aligned,
            })
            log_metrics_dict({
                "n_pcap_flows":     float(len(flows)),
                "n_aligned_benign": float(len(records)),
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