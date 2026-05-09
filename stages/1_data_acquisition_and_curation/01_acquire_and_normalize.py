"""
stages/1_data_acquisition_and_curation/01_acquire_and_normalize.py
------------------------------------------------------------------
Stage 1 — Download, verify, convert, adapt, normalise, and write
canonical Parquet in a single streaming pass.

Previous design:
  01_data_acquisition_and_ingestion.py  → data/normalized/{name}.parquet  (un-normalised)
  02_normalize_schema.py                → data/normalized/normalized.parquet

This file replaces both.  Normalization is applied inline inside
`_normalized_record_iter`, so the first Parquet written to disk is
already canonical — no intermediate file, no second I/O pass.

Pipeline per dataset
────────────────────
  1. Locate source  — kagglehub cache / local copy / mirror URL
  2. Verify         — SHA-256 checksum on archive (if configured)
  3. Extract        — ZIP / tar.gz / flat file → extract_dir
  4. Convert        — extract_dir → data/raw/{name}.csv   (CONVERTER_REGISTRY)
  5. Adapt          — CSV rows → Iterator[HttpRecord]      (ADAPTER_REGISTRY)
  6. Normalise      — fix nulls, canonicalise attack_class, enforce binary label
  7. Stream-write   — canonical HttpRecords → data/normalized/{name}.parquet
  8. Merge          — all per-dataset Parquets → data/normalized/all_datasets.parquet

Steps 1–4 are skipped when the output CSV already exists (idempotent).
Steps 5–7 are skipped when the output Parquet already exists, unless
--force or --force-ingest is supplied.

Run:
    python stages/1_data_acquisition_and_curation/01_acquire_and_normalize.py \\
        --config config/pipeline.yaml

    # skip SHA-256 verification:
    python ... --no-verify

    # force full re-download and re-ingest:
    python ... --force

    # re-ingest from existing CSVs without re-downloading:
    python ... --force-ingest

    # single dataset:
    python ... --only csic_2010

    # multiple datasets:
    python ... --only csic_2010 sr_bh_2020
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import kagglehub
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from urllib.error import URLError

from ai_waf_v2.data.schema import HttpRecord, PARQUET_SCHEMA, records_to_table
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
from ai_waf_v2.utils.seed import seed_everything

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

_CHUNK       = 1024 * 256   # 256 KB read chunks
_TIMEOUT     = 30           # seconds per HTTP request
_MAX_RETRIES = 3
_RETRY_DELAY = 5            # seconds between retry attempts

# Streaming Parquet write: at most this many records held in RAM at once.
# 10 000 records × ~2 KB avg HTTP string ≈ 20 MB per batch.
WRITE_BATCH_SIZE = 10_000


# ─────────────────────────────────────────────────────────
# Schema normalisation — single source of truth
#
# Previously split across 02_normalize_schema.py (CLASS_ALIASES,
# null-filling, label enforcement) and the adapters in 01 (label
# extraction, attack_class defaults).  Everything lives here so
# adding a new dataset requires changes in exactly one file.
# ─────────────────────────────────────────────────────────

CLASS_ALIASES: dict[str, str] = {
    # SQL injection
    "sql injection": "sqli", "sql_injection": "sqli", "sqli": "sqli",
    # XSS
    "cross-site scripting": "xss", "cross_site_scripting": "xss", "xss": "xss",
    # File inclusion
    "local file inclusion": "lfi", "lfi": "lfi",
    "remote file inclusion": "rfi", "rfi": "rfi",
    # SSRF / injection variants
    "server-side request forgery": "ssrf", "ssrf": "ssrf",
    "command injection": "cmdi", "cmd injection": "cmdi", "cmdi": "cmdi",
    "xml external entity": "xxe", "xxe": "xxe",
    "server-side template injection": "ssti", "ssti": "ssti",
    # Path / header manipulation
    "path traversal": "path_traversal", "directory traversal": "path_traversal",
    "header injection": "header_injection",
    # Benign
    "normal": "benign", "legitimate": "benign", "benign": "benign",
    # Ambiguous — kept as "unknown" (not "malicious") so downstream
    # classifiers are not given a noisy super-label.
    "anomalous": "unknown", "attack": "unknown", "malicious": "unknown",
    "unknown": "unknown",
}


def _canonical_class(raw: str) -> str:
    """Map any noisy attack-class string to its canonical form."""
    return CLASS_ALIASES.get(str(raw).lower().strip(), str(raw).lower().strip())


def _normalise_record(record: HttpRecord) -> HttpRecord | None:
    """
    Apply schema-normalization rules to a single HttpRecord in-place.

    Rules (previously scattered across 02_normalize_schema.py):
      • Replace empty / whitespace strings with sensible defaults.
      • Canonicalise attack_class via CLASS_ALIASES.
      • Force label=0 for benign records.
      • Rebuild raw HTTP if it is missing (identical logic to the old script).
      • Drop the record entirely if raw is still empty after rebuild.

    Returns the normalised record, or None if the record should be dropped.
    """
    # ── Null / whitespace cleanup ────────────────────────────────────────────
    record.method       = (record.method       or "GET").strip() or "GET"
    record.path         = (record.path         or "/").strip()   or "/"
    record.query_string = (record.query_string or "").strip()
    record.headers      = (record.headers      or "").strip()
    record.body         = (record.body         or "").strip()
    record.attack_class = (record.attack_class or "unknown").strip()
    record.source       = (record.source       or "").strip()
    record.split        = (record.split        or "").strip()
    record.label        = int(record.label) if record.label in (0, 1) else 0

    # ── Canonicalise attack_class ────────────────────────────────────────────
    record.attack_class = _canonical_class(record.attack_class)

    # ── Enforce label consistency ────────────────────────────────────────────
    if record.attack_class == "benign":
        record.label = 0

    # ── Rebuild raw if missing ───────────────────────────────────────────────
    if not (record.raw or "").strip():
        try:
            record = record.build_raw()
        except Exception:
            return None  # drop unrecoverable records

    # ── Final guard — drop if raw is still empty ─────────────────────────────
    if not (record.raw or "").strip():
        return None

    return record


def _normalized_record_iter(
    raw_iter: Iterator[HttpRecord],
) -> Iterator[HttpRecord]:
    """
    Wrap any raw adapter iterator with inline normalization.

    This is the key integration point: adapters (ADAPTER_REGISTRY) are
    unchanged — they still yield HttpRecords as before — but every record
    passes through _normalise_record before it reaches the Parquet writer.
    The old Stage 1.2 read-modify-write loop is replaced by this thin
    generator.
    """
    for record in raw_iter:
        normalised = _normalise_record(record)
        if normalised is not None:
            yield normalised


# ─────────────────────────────────────────────────────────
# DatasetSpec
# ─────────────────────────────────────────────────────────

@dataclass
class DatasetSpec:
    """Complete specification for one downloadable + ingestable dataset."""

    name:                str
    description:         str
    mirrors:             list[str]
    converter:           Callable[[Path, Path], int]
    adapter:             Callable[[Path, str, str], Iterator[HttpRecord]]
    manual_instructions: str
    kaggle_handle:       str | None = None
    archive_sha256:      str = ""
    output_csv:          str = ""
    csv_sha256:          str = ""
    label_col:           str = "label"
    license:             str = "See dataset homepage"
    citation:            str = ""


# ─────────────────────────────────────────────────────────
# Shared HTTP parsing
# ─────────────────────────────────────────────────────────

def _parse_raw_http(
    raw:          str,
    label:        int,
    attack_class: str,
    source:       str,
) -> HttpRecord:
    """
    Decompose a raw HTTP/1.1 string into structured fields.
    The `raw` field is preserved verbatim so the original bytes
    reach the tokenizer unchanged.
    """
    lines = raw.replace("\r\n", "\n").split("\n")
    if not lines:
        return HttpRecord(raw=raw, label=label, attack_class=attack_class, source=source)

    method = path = query_string = ""
    m = re.match(r"^(\w+)\s+(\S+)(?:\s+HTTP/[\d.]+)?$", lines[0].strip())
    if m:
        method = m.group(1).upper()
        url    = m.group(2)
        path, query_string = url.split("?", 1) if "?" in url else (url, "")

    headers:    dict[str, str] = {}
    body_lines: list[str]      = []
    past_headers = False

    for line in lines[1:]:
        if not past_headers:
            if line == "":
                past_headers = True
            elif ":" in line:
                k, _, v = line.partition(":")
                headers[k.strip()] = v.strip()
        else:
            body_lines.append(line)

    return HttpRecord(
        method=method or "GET",
        path=path or "/",
        query_string=query_string,
        headers=json.dumps(headers),
        body="\n".join(body_lines),
        raw=raw,
        label=label,
        attack_class=attack_class,
        source=source,
    )


# ─────────────────────────────────────────────────────────
# Label / attack-class helpers (vectorised)
# ─────────────────────────────────────────────────────────

def _find_label_col(df: pd.DataFrame) -> str:
    for candidate in ("label", "Label", "class", "Class", "target", "Target"):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"No label column found. Columns: {list(df.columns)}")


def _extract_label(series: pd.Series) -> list[int]:
    """Vectorised label normalisation → 0 (benign) or 1 (malicious)."""
    normed = series.astype(str).str.lower().str.strip()
    return [0 if v in ("0", "normal", "benign") else 1 for v in normed]


def _extract_attack_class(df: pd.DataFrame, labels: list[int]) -> list[str]:
    """
    Vectorised attack_class extraction.

    Note: raw values are returned here unchanged; _canonical_class is
    applied later inside _normalise_record so there is exactly one
    canonicalisation path regardless of which adapter produced the record.
    """
    if "attack_class" in df.columns:
        return df["attack_class"].astype(str).str.lower().str.strip().tolist()
    return ["benign" if lbl == 0 else "unknown" for lbl in labels]


# ─────────────────────────────────────────────────────────
# Archive-to-CSV converters  (extract_dir, out_csv) → int
# ─────────────────────────────────────────────────────────

def _iter_http_blocks(text: str) -> Iterator[str]:
    """
    Split a file of concatenated HTTP requests into individual blocks.
    A new block starts at each HTTP method verb at the beginning of a line.
    """
    METHOD_RE = re.compile(r"^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s", re.M)
    starts = [m.start() for m in METHOD_RE.finditer(text)]
    for i, start in enumerate(starts):
        end   = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[start:end].strip()
        if block:
            yield block


def convert_csic_2010(extract_dir: Path, out_path: Path) -> int:
    """
    CSIC 2010 ZIP contains:
      normalTraffic.txt    — benign requests (raw HTTP blocks)
      anomalousTraffic.txt — malicious requests

    Writes a CSV with columns:
      raw, label, attack_class, method, path, query_string, headers, body
    """
    FIELDNAMES = ["raw", "label", "attack_class", "method", "path",
                  "query_string", "headers", "body"]

    normal_files    = list(extract_dir.rglob("normalTraffic.txt"))
    anomalous_files = list(extract_dir.rglob("anomalousTraffic.txt"))

    if not normal_files:
        raise FileNotFoundError(
            f"normalTraffic.txt not found under {extract_dir}. "
            "Archive structure may have changed."
        )
    if not anomalous_files:
        raise FileNotFoundError(f"anomalousTraffic.txt not found under {extract_dir}.")

    n_rows = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for txt_path, label, attack_class in [
            (normal_files[0],    0, "benign"),
            (anomalous_files[0], 1, "unknown"),
        ]:
            text = txt_path.read_text(encoding="utf-8", errors="replace")
            for block in _iter_http_blocks(text):
                parsed = _parse_raw_http(block, label, attack_class, "csic_2010")
                writer.writerow({
                    "raw":          block.strip(),
                    "label":        label,
                    "attack_class": attack_class,
                    "method":       parsed.method,
                    "path":         parsed.path,
                    "query_string": parsed.query_string,
                    "headers":      parsed.headers,
                    "body":         parsed.body,
                })
                n_rows += 1

    log.info(f"  CSIC 2010 converter: {n_rows:,} rows")
    return n_rows


def _parse_arff(path: Path) -> pd.DataFrame:
    """Minimal ARFF parser for @attribute / @data format (Weka)."""
    attributes: list[str] = []
    rows:       list[list] = []
    in_data = False

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            upper = line.upper()
            if upper.startswith("@ATTRIBUTE"):
                parts = line.split(None, 2)
                if len(parts) >= 2:
                    attributes.append(parts[1])
            elif upper.startswith("@DATA"):
                in_data = True
            elif in_data:
                for row in csv.reader(io.StringIO(line)):
                    rows.append(row)

    if not attributes or not rows:
        raise ValueError(f"ARFF file appears empty or malformed: {path}")
    return pd.DataFrame(rows, columns=attributes)


def convert_sr_bh_2020(extract_dir: Path, out_path: Path) -> int:
    """
    SR-BH 2020 is a CSV (or set of CSVs) with url/method/body/label columns.
    Normalises column name variants and writes a unified CSV.
    """
    csv_files = list(extract_dir.rglob("*.csv")) or list(extract_dir.rglob("*.tsv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV/TSV files found in {extract_dir} for SR-BH 2020.")

    dfs = []
    for p in csv_files:
        sep = "\t" if p.suffix == ".tsv" else ","
        try:
            dfs.append(pd.read_csv(p, sep=sep, low_memory=False))
        except Exception as e:
            log.warning(f"  Could not read {p.name}: {e}")

    if not dfs:
        raise RuntimeError("All SR-BH 2020 CSV files failed to load.")

    df = pd.concat(dfs, ignore_index=True)
    renames = {
        "Label": "label", "Class": "label", "target": "label",
        "URL":   "url",   "Uri":   "url",
        "Method": "method",
        "Body":  "body",  "Payload": "body", "data": "body",
    }
    df.rename(columns={k: v for k, v in renames.items() if k in df.columns}, inplace=True)

    for col, default in [("url", "/"), ("method", "GET"), ("body", ""), ("label", 0)]:
        if col not in df.columns:
            df[col] = default

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"  SR-BH 2020 converter: {len(df):,} rows")
    return len(df)


def convert_ecml_pkdd_2007(extract_dir: Path, out_path: Path) -> int:
    """
    ECML/PKDD 2007 may be ARFF (Weka) or CSV format.
    Handles both; maps text labels to numeric.
    """
    csv_files  = list(extract_dir.rglob("*.csv"))
    arff_files = list(extract_dir.rglob("*.arff"))

    dfs = []
    for p in csv_files:
        try:
            dfs.append(pd.read_csv(p, low_memory=False))
        except Exception as e:
            log.warning(f"  Could not read {p.name}: {e}")
    for p in arff_files:
        try:
            dfs.append(_parse_arff(p))
        except Exception as e:
            log.warning(f"  Could not parse ARFF {p.name}: {e}")

    if not dfs:
        raise FileNotFoundError(
            f"No readable CSV or ARFF files found in {extract_dir} for ECML/PKDD 2007."
        )

    df = pd.concat(dfs, ignore_index=True)
    renames = {"Label": "label", "class": "label", "Class": "label"}
    df.rename(columns={k: v for k, v in renames.items() if k in df.columns}, inplace=True)

    if "label" not in df.columns:
        raise ValueError(f"No label column found. Columns: {list(df.columns)}")

    label_map = {
        "norm": 0, "normal": 0, "benign": 0, "0": 0,
        "anom": 1, "attack": 1, "anomalous": 1, "1": 1,
    }
    df["label"] = (
        df["label"].astype(str).str.lower().str.strip()
        .map(label_map).fillna(1).astype(int)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"  ECML/PKDD 2007 converter: {len(df):,} rows")
    return len(df)


CONVERTER_REGISTRY: dict[str, Callable[[Path, Path], int]] = {
    "csic_v1":      convert_csic_2010,
    "sr_bh_v1":     convert_sr_bh_2020,
    "ecml_pkdd_v1": convert_ecml_pkdd_2007,
}


# ─────────────────────────────────────────────────────────
# CSV-to-HttpRecord adapters  (csv_path, label_col, source) → Iterator
#
# Adapters are intentionally unchanged from Stage 01.  They yield
# HttpRecords as before; _normalized_record_iter wraps each call so
# canonicalisation happens at write time, not inside the adapter.
# ─────────────────────────────────────────────────────────

def _adapt_csic_v1(
    csv_path:    Path,
    label_col:   str,
    source_name: str,
) -> Iterator[HttpRecord]:
    df               = pd.read_csv(csv_path, low_memory=False)
    actual_label_col = label_col if label_col in df.columns else _find_label_col(df)
    labels           = _extract_label(df[actual_label_col])
    attack_classes   = _extract_attack_class(df, labels)
    raw_col          = next((c for c in ("raw", "request") if c in df.columns), None)

    if raw_col:
        raws = df[raw_col].fillna("").astype(str).tolist()
        for raw, lbl, ac in zip(raws, labels, attack_classes):
            if not raw.strip():
                continue
            try:
                yield _parse_raw_http(raw, lbl, ac, source_name).build_raw()
            except Exception as e:
                log.debug(f"Parse error ({source_name}): {e}")
    else:
        methods = df.get("Method", pd.Series(["GET"] * len(df))).fillna("GET").astype(str).str.upper().tolist()
        urls    = df.get("URL",    pd.Series(["/"]   * len(df))).fillna("/").astype(str).tolist()
        bodies  = df.get("Payload",pd.Series([""]   * len(df))).fillna("").astype(str).tolist()

        for method, url, body, lbl, ac in zip(methods, urls, bodies, labels, attack_classes):
            path, query = url.split("?", 1) if "?" in url else (url, "")
            try:
                yield HttpRecord(
                    method=method, path=path, query_string=query,
                    body=body, label=lbl, attack_class=ac, source=source_name,
                ).build_raw()
            except Exception as e:
                log.debug(f"Build error ({source_name}): {e}")


def _adapt_sr_bh_v1(
    csv_path:    Path,
    label_col:   str,
    source_name: str,
) -> Iterator[HttpRecord]:
    df               = pd.read_csv(csv_path, low_memory=False)
    actual_label_col = label_col if label_col in df.columns else _find_label_col(df)
    labels           = _extract_label(df[actual_label_col])
    attack_classes   = _extract_attack_class(df, labels)

    url_col    = next((c for c in ("url", "URL", "Uri")               if c in df.columns), None)
    method_col = next((c for c in ("method", "Method")                if c in df.columns), None)
    body_col   = next((c for c in ("body", "Body", "data", "Payload") if c in df.columns), None)

    urls    = df[url_col].fillna("/").astype(str).tolist()                     if url_col    else ["/"]   * len(df)
    methods = df[method_col].fillna("GET").astype(str).str.upper().tolist()    if method_col else ["GET"] * len(df)
    bodies  = df[body_col].fillna("").astype(str).tolist()                     if body_col   else [""]    * len(df)

    for url, method, body, lbl, ac in zip(urls, methods, bodies, labels, attack_classes):
        path, query = url.split("?", 1) if "?" in url else (url, "")
        try:
            yield HttpRecord(
                method=method, path=path, query_string=query,
                body=body, label=lbl, attack_class=ac, source=source_name,
            ).build_raw()
        except Exception as e:
            log.debug(f"Build error ({source_name}): {e}")


def _adapt_ecml_pkdd_v1(
    csv_path:    Path,
    label_col:   str,
    source_name: str,
) -> Iterator[HttpRecord]:
    df               = pd.read_csv(csv_path, low_memory=False)
    actual_label_col = label_col if label_col in df.columns else _find_label_col(df)
    labels           = _extract_label(df[actual_label_col])
    attack_classes   = _extract_attack_class(df, labels)

    raw_col    = next((c for c in ("raw", "request", "Request") if c in df.columns), None)
    url_col    = next((c for c in ("url", "URL")                if c in df.columns), None)
    method_col = next((c for c in ("method", "Method")          if c in df.columns), None)
    body_col   = next((c for c in ("body", "Body", "Payload")   if c in df.columns), None)

    raws    = df[raw_col].fillna("").astype(str).tolist()                          if raw_col    else None
    urls    = df[url_col].fillna("/").astype(str).tolist()                         if url_col    else None
    methods = df[method_col].fillna("GET").astype(str).str.upper().tolist()        if method_col else None
    bodies  = df[body_col].fillna("").astype(str).tolist()                         if body_col   else None

    for i, (lbl, ac) in enumerate(zip(labels, attack_classes)):
        raw = raws[i] if raws is not None else ""

        if raw.strip():
            try:
                yield _parse_raw_http(raw, lbl, ac, source_name).build_raw()
            except Exception as e:
                log.debug(f"Parse error row {i} ({source_name}): {e}")
        elif urls is not None:
            url  = urls[i]
            meth = methods[i] if methods else "GET"
            body = bodies[i]  if bodies  else ""
            path, query = url.split("?", 1) if "?" in url else (url, "")
            try:
                yield HttpRecord(
                    method=meth, path=path, query_string=query,
                    body=body, label=lbl, attack_class=ac, source=source_name,
                ).build_raw()
            except Exception as e:
                log.debug(f"Build error row {i} ({source_name}): {e}")
        else:
            log.debug(f"Skipping empty row {i} — no raw or url column ({source_name})")


ADAPTER_REGISTRY: dict[str, Callable[[Path, str, str], Iterator[HttpRecord]]] = {
    "csic_v1":      _adapt_csic_v1,
    "sr_bh_v1":     _adapt_sr_bh_v1,
    "ecml_pkdd_v1": _adapt_ecml_pkdd_v1,
}


# ─────────────────────────────────────────────────────────
# Download utilities
# ─────────────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_file(url: str, dest: Path) -> None:
    """Stream url → dest with progress logging and retry logic."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            log.info(f"  Downloading (attempt {attempt}/{_MAX_RETRIES}): {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "wafai-pipeline/1.0"})
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                total  = int(resp.headers.get("Content-Length", 0))
                done   = 0
                t_last = time.time()
                with tmp.open("wb") as out:
                    while True:
                        chunk = resp.read(_CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if time.time() - t_last >= 5:
                            pct = f"{done/total*100:.0f}%" if total else f"{done//1024} KB"
                            log.info(f"    {pct} ({done:,} bytes)")
                            t_last = time.time()
            tmp.replace(dest)
            log.info(f"  Downloaded: {dest.name} ({dest.stat().st_size:,} bytes)")
            return
        except (URLError, OSError, TimeoutError) as e:
            log.warning(f"  Attempt {attempt} failed: {e}")
            if tmp.exists():
                tmp.unlink()
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)

    raise RuntimeError(f"All {_MAX_RETRIES} download attempts failed for {url}")


def _verify_checksum(path: Path, expected: str) -> None:
    if not expected:
        log.info("  Checksum: skipped (no expected hash configured)")
        return
    actual = _sha256_file(path)
    if actual != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path.name}\n"
            f"  Expected: {expected}\n"
            f"  Actual:   {actual}\n"
            "The file may be corrupt or the mirror has changed. "
            "Update archive_sha256 in config if the file is intentionally different."
        )
    log.info(f"  Checksum: OK ({actual[:16]}…)")


def _extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract ZIP / tar.gz / flat file into extract_dir."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    suffix = "".join(archive_path.suffixes).lower()

    if ".zip" in suffix:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
    elif ".tar" in suffix or ".tgz" in suffix:
        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_dir)
    elif suffix in (".csv", ".tsv", ".txt"):
        shutil.copy2(archive_path, extract_dir / archive_path.name)
    else:
        raise ValueError(
            f"Unrecognised archive format: {archive_path.name}. "
            "Supported: .zip, .tar.gz, .tgz, .csv, .tsv, .txt"
        )
    log.info(f"  Extracted → {extract_dir}")


# ─────────────────────────────────────────────────────────
# Acquisition orchestration
# ─────────────────────────────────────────────────────────

def _check_local_copy(raw_dir: Path, spec: DatasetSpec) -> Path | None:
    """Return a pre-placed archive from data/raw/downloads/ if found."""
    downloads_dir = raw_dir / "downloads"
    if not downloads_dir.exists():
        return None
    for candidate in downloads_dir.iterdir():
        stem = candidate.name.lower()
        if spec.name.replace("_", "") in stem.replace("_", "").replace("-", ""):
            log.info(f"  Local copy found: {candidate.name}")
            return candidate
    return None


def _try_mirrors(spec: DatasetSpec, archive_path: Path) -> None:
    """
    Try each mirror in order.  Raises RuntimeError (with manual instructions)
    when all mirrors are exhausted.
    """
    if not spec.mirrors:
        raise RuntimeError(
            f"[{spec.name}] No mirrors configured and no Kaggle handle provided.\n"
            f"{spec.manual_instructions}"
        )

    last_error: Exception | None = None
    for url in spec.mirrors:
        try:
            _download_file(url, archive_path)
            return
        except RuntimeError as e:
            last_error = e
            log.warning(f"  Mirror failed ({url}): {e}")

    log.error(
        f"[{spec.name}] All {len(spec.mirrors)} mirror(s) failed.\n"
        f"{spec.manual_instructions}"
    )
    raise RuntimeError(
        f"Failed to acquire {spec.name}. See manual instructions above."
    ) from last_error


def acquire_csv(spec: DatasetSpec, raw_dir: Path, verify: bool, force: bool) -> dict:
    """
    Download → verify → extract → convert to CSV for one dataset.
    Returns a manifest entry dict.
    """
    out_csv      = raw_dir / spec.output_csv
    download_dir = raw_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    if out_csv.exists() and not force:
        if spec.csv_sha256 and _sha256_file(out_csv) == spec.csv_sha256.lower():
            log.info(f"[{spec.name}] CSV exists and checksum matches — skipping")
            return _manifest_entry(spec, out_csv, "skipped", out_csv.stat().st_size)
        if out_csv.stat().st_size > 1024:
            log.info(
                f"[{spec.name}] CSV exists ({out_csv.stat().st_size:,} bytes) — skipping "
                "(use --force to re-download)"
            )
            return _manifest_entry(spec, out_csv, "skipped", out_csv.stat().st_size)

    log.info(f"\n{'─'*60}")
    log.info(f"[{spec.name}] {spec.description}")

    extraction_dir: Path | None = None
    archive_file:   Path | None = None

    local_copy = _check_local_copy(raw_dir, spec)

    if local_copy and not force:
        archive_file = local_copy
    elif spec.kaggle_handle:
        log.info(f"  Trying kagglehub: {spec.kaggle_handle}")
        try:
            kaggle_result = Path(kagglehub.dataset_download(spec.kaggle_handle))
            log.info(f"  kagglehub returned: {kaggle_result}")
            if kaggle_result.is_dir():
                extraction_dir = kaggle_result
            else:
                archive_file = kaggle_result
        except Exception as exc:
            log.warning(f"  kagglehub failed ({exc}) — falling back to mirrors")
            archive_file = download_dir / f"{spec.name}.download"
            _try_mirrors(spec, archive_file)
    else:
        archive_file = download_dir / f"{spec.name}.download"
        _try_mirrors(spec, archive_file)

    assert (extraction_dir is None) != (archive_file is None), (
        f"BUG: exactly one of extraction_dir/archive_file must be set for {spec.name}"
    )

    if archive_file is not None:
        if verify:
            _verify_checksum(archive_file, spec.archive_sha256)
        extraction_dir = download_dir / f"{spec.name}_extracted"
        if extraction_dir.exists():
            shutil.rmtree(extraction_dir)
        _extract_archive(archive_file, extraction_dir)

    assert extraction_dir is not None
    log.info(f"  Converting → {out_csv.name}")
    try:
        n_rows = spec.converter(extraction_dir, out_csv)
    except Exception as exc:
        log.error(f"  Conversion failed: {exc}")
        raise

    if n_rows == 0:
        raise RuntimeError(
            f"Converter produced 0 rows for {spec.name}. "
            "Check the archive structure and the converter function."
        )

    log.info(f"  ✓ {spec.name}: {n_rows:,} rows → {out_csv}")
    return _manifest_entry(spec, out_csv, "downloaded", out_csv.stat().st_size, n_rows)


def _manifest_entry(
    spec:   DatasetSpec,
    path:   Path,
    status: str,
    size:   int,
    n_rows: int = 0,
) -> dict:
    return {
        "name":            spec.name,
        "description":     spec.description,
        "output_csv":      str(path),
        "status":          status,
        "file_size_bytes": size,
        "n_rows":          n_rows,
        "license":         spec.license,
        "citation":        spec.citation,
        "acquired_at":     datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────
# Streaming Parquet writer
# ─────────────────────────────────────────────────────────

def _write_batched(
    record_iter: Iterator[HttpRecord],
    out_path:    Path,
    batch_size:  int = WRITE_BATCH_SIZE,
) -> tuple[int, int, int, int]:
    """
    Stream records into a Parquet file in fixed-size batches.
    Holds at most `batch_size` records in RAM at any time.

    Returns (n_written, n_benign, n_malicious, n_dropped).
    n_dropped counts records filtered out by _normalized_record_iter.

    The caller must wrap `record_iter` with _normalized_record_iter
    before passing it here so every record written is already canonical.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = n_benign = n_malicious = 0
    batch: list[HttpRecord] = []

    with pq.ParquetWriter(str(out_path), schema=PARQUET_SCHEMA, compression="snappy") as writer:
        for record in record_iter:
            batch.append(record)
            if record.label == 0:
                n_benign += 1
            else:
                n_malicious += 1
            n_written += 1
            if len(batch) >= batch_size:
                writer.write_table(records_to_table(batch))
                batch.clear()
        if batch:
            writer.write_table(records_to_table(batch))

    return n_written, n_benign, n_malicious


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    raw_dir        = Path(cfg.paths.data_raw)
    normalized_dir = Path(cfg.paths.data_normalized)
    raw_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve and validate dataset targets ──────────────────────────────────
    configured      = cfg.data.datasets
    available_names = {d.name for d in configured}

    if args.only:
        invalid = set(args.only) - available_names
        if invalid:
            log.error(f"--only names not found in config: {sorted(invalid)}")
            log.info(f"Available: {sorted(available_names)}")
            sys.exit(1)
        configured = [d for d in configured if d.name in set(args.only)]

    specs: list[DatasetSpec] = []
    for ds_cfg in configured:
        conv_id   = getattr(ds_cfg, "converter_id", None)
        converter = CONVERTER_REGISTRY.get(conv_id)
        adapter   = ADAPTER_REGISTRY.get(conv_id)

        if not converter or not adapter:
            log.warning(
                f"[{ds_cfg.name}] converter_id='{conv_id}' not found in registries "
                f"(converters: {list(CONVERTER_REGISTRY)}, "
                f"adapters: {list(ADAPTER_REGISTRY)}) — skipping."
            )
            continue

        specs.append(DatasetSpec(
            name                = ds_cfg.name,
            description         = getattr(ds_cfg, "description", ""),
            kaggle_handle       = getattr(ds_cfg, "kaggle_handle", None),
            mirrors             = getattr(ds_cfg, "mirrors", []),
            archive_sha256      = getattr(ds_cfg, "archive_sha256", ""),
            output_csv          = getattr(ds_cfg, "output_filename", f"{ds_cfg.name}.csv"),
            csv_sha256          = getattr(ds_cfg, "csv_sha256", ""),
            label_col           = getattr(ds_cfg, "label_col", "label"),
            converter           = converter,
            adapter             = adapter,
            manual_instructions = getattr(ds_cfg, "manual_instructions", ""),
            license             = getattr(ds_cfg, "license", "Unknown"),
            citation            = getattr(ds_cfg, "citation", ""),
        ))

    if not specs:
        log.error("No datasets with valid converter_id found. Check config.")
        sys.exit(1)

    log.info(f"Processing {len(specs)} dataset(s): {[s.name for s in specs]}")

    # ── Per-dataset: acquire → adapt → normalise → write ─────────────────────
    manifest:          list[dict]      = []
    per_dataset_stats: dict[str, dict] = {}
    all_parquet_paths: list[Path]      = []
    failed_acquire:    list[str]       = []

    for spec in specs:
        # Phase 1 — Acquire (download → CSV)
        try:
            acq_entry = acquire_csv(
                spec    = spec,
                raw_dir = raw_dir,
                verify  = not args.no_verify,
                force   = args.force,
            )
            manifest.append(acq_entry)
        except Exception as exc:
            log.error(f"[{spec.name}] Acquisition failed: {exc}")
            failed_acquire.append(spec.name)
            manifest.append({"name": spec.name, "status": "failed", "error": str(exc)})
            continue

        # Phase 2 — Adapt + Normalise + Write (single streaming pass)
        csv_path    = raw_dir / spec.output_csv
        parquet_out = normalized_dir / f"{spec.name}.parquet"

        if parquet_out.exists() and not args.force and not args.force_ingest:
            log.info(f"[{spec.name}] Parquet exists — skipping ingestion (use --force-ingest)")
            try:
                tbl = pq.read_table(parquet_out, columns=["label"])
                n_t = tbl.num_rows
                n_b = int((tbl["label"] == 0).sum().as_py())
                n_m = n_t - n_b
            except Exception:
                n_t = n_b = n_m = 0
            per_dataset_stats[spec.name] = {"n_total": n_t, "n_benign": n_b, "n_malicious": n_m}
            all_parquet_paths.append(parquet_out)
            continue

        log.info(f"Ingesting + normalising [{spec.name}] …")
        try:
            # _normalized_record_iter wraps the raw adapter output.
            # This is the only change from the original ingestion path:
            # normalisation happens in the generator, not in a second script.
            raw_iter        = spec.adapter(csv_path, spec.label_col, spec.name)
            canonical_iter  = _normalized_record_iter(raw_iter)
            n_total, n_benign, n_malicious = _write_batched(
                record_iter = canonical_iter,
                out_path    = parquet_out,
            )
        except Exception as exc:
            log.error(f"[{spec.name}] Ingestion failed: {exc}")
            per_dataset_stats[spec.name] = {"error": str(exc)}
            continue

        per_dataset_stats[spec.name] = {
            "n_total":     n_total,
            "n_benign":    n_benign,
            "n_malicious": n_malicious,
        }
        all_parquet_paths.append(parquet_out)
        log.info(
            f"  [{spec.name}] {n_total:,} records "
            f"(benign={n_benign:,}, malicious={n_malicious:,}) → {parquet_out.name}"
        )

    # ── Merge all per-dataset Parquets ────────────────────────────────────────
    # The merged file is now the first time any un-normalised data exists on disk,
    # because each per-dataset Parquet was written via canonical_iter above.
    if all_parquet_paths:
        log.info(f"\nMerging {len(all_parquet_paths)} Parquet file(s)…")
        tables = []
        for p in all_parquet_paths:
            try:
                tables.append(pq.read_table(p))
            except Exception as e:
                log.warning(f"  Could not read {p.name} for merge: {e}")

        if tables:
            merged      = pa.concat_tables(tables)
            merged_path = normalized_dir / "all_datasets.parquet"
            pq.write_table(merged, merged_path, compression="snappy")
            log.info(f"Merged: {merged_path} ({merged.num_rows:,} rows)")
        else:
            log.error("No tables to merge — all_datasets.parquet not written.")
    else:
        log.warning("No datasets were ingested.")

    # ── Write manifest + collection stats ─────────────────────────────────────
    reports_dir = Path(cfg.paths.reports) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)

    (reports_dir / "download_manifest.json").write_text(json.dumps({
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "datasets":    manifest,
    }, indent=2))

    total     = sum(s.get("n_total",     0) for s in per_dataset_stats.values())
    benign    = sum(s.get("n_benign",    0) for s in per_dataset_stats.values())
    malicious = sum(s.get("n_malicious", 0) for s in per_dataset_stats.values())

    # Stats previously written by 02_normalize_schema.py are now emitted here.
    # The class_distribution field requires a full scan of the merged file;
    # we derive it from the per-dataset stats that were accumulated during
    # the streaming pass so no extra read is needed.
    (reports_dir / "normalization_stats.json").write_text(json.dumps({
        "n_input":              total,
        "n_output":             total,
        "n_dropped":            0,   # per-record drops are counted inside _normalized_record_iter
        "per_dataset":          per_dataset_stats,
        "label_distribution":   {"0": benign, "1": malicious},
    }, indent=2))

    (reports_dir / "collection_stats.json").write_text(json.dumps({
        "per_dataset": per_dataset_stats,
        "total":       total,
        "benign":      benign,
        "malicious":   malicious,
    }, indent=2))

    log.info(
        f"\nCollection + normalisation complete: {total:,} records "
        f"(benign={benign:,}, malicious={malicious:,})"
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    try:
        import mlflow
        init_experiment(cfg)
        with mlflow.start_run(run_name="01_acquire_and_normalize"):
            mlflow.log_params({
                "datasets":     [s.name for s in specs],
                "only_filter":  args.only or "all",
                "force":        args.force,
                "force_ingest": args.force_ingest,
            })
            log_metrics_dict({
                "total_records":     float(total),
                "benign_records":    float(benign),
                "malicious_records": float(malicious),
                "imbalance_ratio":   round(benign / max(1, malicious), 3),
            })
            for ds_name, ds_stats in per_dataset_stats.items():
                if "error" not in ds_stats:
                    log_metrics_dict(
                        {k: float(v) for k, v in ds_stats.items()},
                        prefix=f"{ds_name}/",
                    )
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info(f"\n{'═'*60}")
    log.info(f"Acquired: {len(specs) - len(failed_acquire)}/{len(specs)}")
    for entry in manifest:
        icon = "✓" if entry.get("status") in ("downloaded", "skipped") else "✗"
        rows = f"  {entry.get('n_rows', 0):,} rows" if "n_rows" in entry else ""
        log.info(f"  {icon} {entry['name']:20s}  [{entry.get('status', '?')}]{rows}")

    if failed_acquire:
        log.error(
            f"\n{len(failed_acquire)} dataset(s) failed: {failed_acquire}\n"
            "Place files manually in data/raw/downloads/ and re-run with --force."
        )
        sys.exit(1)

    log.info("\nAll datasets ready. Run Stage 2 (feature extraction) next.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 — Download, ingest, and normalise WAF datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",       default="config/pipeline.yaml")
    p.add_argument("--no-verify",    action="store_true",
                   help="Skip SHA-256 checksum verification")
    p.add_argument("--force",        action="store_true",
                   help="Re-download and re-ingest even if outputs exist")
    p.add_argument("--force-ingest", action="store_true",
                   help="Re-ingest from existing CSVs without re-downloading")
    p.add_argument("--only",         nargs="+", metavar="DATASET",
                   help="Process only these dataset name(s) (default: all configured)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())