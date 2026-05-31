"""
stages/1_data_acquisition_and_curation/01_acquire_and_normalize.py
------------------------------------------------------------------
Stage 1 — Download, verify, convert, adapt, Normalize, and write
canonical Parquet in a single streaming pass.


This file replaces both.  Normalization is applied inline inside
`_normalized_record_iter`, so the first Parquet written to disk is
already canonical — no intermediate file, no second I/O pass.

Pipeline per dataset
────────────────────
  1. Locate source  — kagglehub cache / local copy / mirror URL
  2. Verify         — SHA-256 checksum on archive (if configured)
  3. Extract        — ZIP / tar.gz / flat file → extract_dir (if necessary)
  4. Convert        — extract_dir → data/raw/{name}.csv   (CONVERTER_REGISTRY)
  5. Adapt          — CSV rows → Iterator[HttpRecord]      (ADAPTER_REGISTRY)
  6. Normalize      — fix nulls, canonicalise attack_class, enforce binary label
  7. Stream-write   — canonical HttpRecords → data/normalized/{name}.parquet
  8. Merge          — all per-dataset Parquets → data/normalized/all_datasets.parquet

Steps 1–4 are skipped when the output CSV already exists (idempotent).
Steps 5–7 are skipped when the output Parquet already exists, unless
--force or --force-ingest is supplied.

Run:
    python stages/1_data_acquisition_and_curation/01_acquire_and_normalize.py \
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
from urllib.parse import urlparse

import kagglehub
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc   
import pyarrow.parquet as pq
from urllib.error import URLError

from ai_waf_v2.data.schema import (
    HttpRecord, PARQUET_SCHEMA, records_to_table,
    CLASS_ALIASES, canonical_class as _canonical_class,
)
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
from ai_waf_v2.utils.seed import seed_everything
from ai_waf_v2.utils.timing import StepTimer

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



def _Normalize_record(record: HttpRecord) -> HttpRecord | None:
    """
    Apply schema-normalization rules to a single HttpRecord in-place.

    Rules:
      • Replace empty / whitespace strings with sensible defaults.
      • Canonicalise attack_class via CLASS_ALIASES.
      • Force label=0 for benign records.
      • Rebuild raw HTTP if it is missing (identical logic to the old script).
      • Drop the record entirely if raw is still empty after rebuild.

    Returns the Normalized record, or None if the record should be dropped.
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
    passes through _Normalize_record before it reaches the Parquet writer.
    
    """
    for record in raw_iter:
        Normalized = _Normalize_record(record)
        if Normalized is not None:
            yield Normalized


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
        method       = m.group(1).upper()
        url          = m.group(2)
        # urlparse handles both absolute (http://host/path?qs) and relative
        # (/path?qs) URLs, stripping scheme+host from CSIC 2010's request lines.
        _parsed      = urlparse(url)
        path         = _parsed.path or "/"
        query_string = _parsed.query

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
    for candidate in ("label", "Label", "classification", "class", "Class", "target", "Target"):
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
    applied later inside _Normalize_record so there is exactly one
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


def _convert_csic_v1(extract_dir: Path, out_path: Path) -> int:
    """
    CSIC 2010: Kaggle delivers this dataset as CSV(s) — concatenate and save as-is.
    Falls back to parsing normalTraffic.txt / anomalousTraffic.txt for the original ZIP.
    """
    # ── Kaggle delivery: pre-built CSV(s) ─────────────────────────────────────
    csv_files = list(extract_dir.rglob("*.csv"))
    if csv_files:
        dfs = []
        for p in sorted(csv_files):
            try:
                dfs.append(pd.read_csv(p, low_memory=False))
            except Exception as e:
                log.warning(f"  Could not read {p.name}: {e}")
        if not dfs:
            raise RuntimeError("All CSIC 2010 CSV files failed to load.")
        df = pd.concat(dfs, ignore_index=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        #log.info(f"  CSIC 2010 converter (CSV): {len(df):,} rows")
        return len(df)

    # ── Original ZIP format: normalTraffic.txt + anomalousTraffic.txt ─────────
    FIELDNAMES = ["raw", "label", "attack_class", "method", "path",
                  "query_string", "headers", "body"]

    normal_files    = list(extract_dir.rglob("normalTraffic.txt"))
    anomalous_files = list(extract_dir.rglob("anomalousTraffic.txt"))

    if not normal_files:
        raise FileNotFoundError(
            f"Neither CSV files nor normalTraffic.txt found under {extract_dir}. "
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

    log.info(f"  CSIC 2010 converter (TXT): {n_rows:,} rows")
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


def _convert_sr_bh_v1(extract_dir: Path, out_path: Path) -> int:
    """SR-BH 2020: concatenate CSV/TSV files and save as-is."""
    csv_files = list(extract_dir.rglob("*.csv")) or list(extract_dir.rglob("*.tsv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV/TSV files found in {extract_dir} for SR-BH 2020.")

    dfs = []
    for p in sorted(csv_files):
        sep = "\t" if p.suffix == ".tsv" else ","
        try:
            dfs.append(pd.read_csv(p, sep=sep, low_memory=False))
        except Exception as e:
            log.warning(f"  Could not read {p.name}: {e}")

    if not dfs:
        raise RuntimeError("All SR-BH 2020 CSV files failed to load.")

    df = pd.concat(dfs, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"  SR-BH 2020 converter: {len(df):,} rows")
    return len(df)


def _convert_http_params_v1(extract_dir: Path, out_path: Path) -> int:
    """HTTP Params Dataset: concatenate all CSVs from the archive and save as-is."""
    csv_files = list(extract_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {extract_dir} for HTTP Params Dataset.")

    dfs = []
    for p in sorted(csv_files):
        try:
            dfs.append(pd.read_csv(p, low_memory=False))
        except Exception as e:
            log.warning(f"  Could not read {p.name}: {e}")

    if not dfs:
        raise RuntimeError("All HTTP Params CSV files failed to load.")

    df = pd.concat(dfs, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(f"  HTTP Params converter: {len(df):,} rows")
    return len(df)


CONVERTER_REGISTRY: dict[str, Callable[[Path, Path], int]] = {
    "csic_v1":          _convert_csic_v1,
    "sr_bh_v1":         _convert_sr_bh_v1,
    "http_params_v1":   _convert_http_params_v1,
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
        # CSIC 2010 Kaggle delivery: one column per header field.
        # 'lenght' is a typo in the original dataset for 'Content-Length'.
        _HEADER_COL_MAP = {
            "User-Agent":      "User-Agent",
            "Pragma":          "Pragma",
            "Cache-Control":   "Cache-Control",
            "Accept":          "Accept",
            "Accept-encoding": "Accept-Encoding",
            "Accept-charset":  "Accept-Charset",
            "language":        "Accept-Language",
            "host":            "Host",
            "cookie":          "Cookie",
            "content-type":    "Content-Type",
            "connection":      "Connection",
            "lenght":          "Content-Length",
        }

        methods = df.get("Method",  pd.Series(["GET"] * len(df))).fillna("GET").astype(str).str.upper().tolist()
        urls    = df.get("URL",     pd.Series(["/"]   * len(df))).fillna("/").astype(str).tolist()
        bodies  = df.get("content", pd.Series([""]   * len(df))).fillna("").astype(str).tolist()

        # Pre-extract header columns as lists to avoid per-row iloc overhead
        header_cols: dict[str, list[str]] = {}
        for col, hdr in _HEADER_COL_MAP.items():
            if col in df.columns:
                header_cols[hdr] = df[col].fillna("").astype(str).tolist()

        for i, (method, url, body, lbl, ac) in enumerate(zip(methods, urls, bodies, labels, attack_classes)):
            _parsed = urlparse(url)
            path    = _parsed.path or "/"
            query   = _parsed.query
            headers: dict[str, str] = {}
            for hdr, vals in header_cols.items():
                val = vals[i].strip()
                if not val or val.lower() == "nan":
                    continue
                prefix = hdr + ": "
                if val.lower().startswith(prefix.lower()):
                    val = val[len(prefix):]
                headers[hdr] = val
            try:
                yield HttpRecord(
                    method=method, path=path, query_string=query,
                    headers=json.dumps(headers),
                    body=body, label=lbl, attack_class=ac, source=source_name,
                ).build_raw()
            except Exception as e:
                log.debug(f"Build error ({source_name}): {e}")


_SR_BH_FIELD_RE = re.compile(
    r'(?<!\S)(Method|Host|Body|Content-Type|User-Agent|Referer|Cookie|Accept-Encoding|Accept-Language|Accept)\s*:',
    re.IGNORECASE,
)


def _parse_sr_bh_text(text: str, label: int, attack_class: str, source: str) -> HttpRecord:
    """
    Parse SR-BH 2020 'FieldName: value FieldName: value ...' single-line format.

    Fields seen in the dataset: Method, Host, Body, Content-Type, User-Agent,
    Referer, Cookie, Accept, Accept-Language, Accept-Encoding.
    Host contains the URL path (and optional query string) rather than a hostname.
    """
    matches = list(_SR_BH_FIELD_RE.finditer(text))

    fields: dict[str, str] = {}
    for i, m in enumerate(matches):
        key   = m.group(1).lower()
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fields[key] = text[start:end].strip()

    method   = (fields.get("method") or "GET").split()[0].upper() or "GET"
    host_val = fields.get("host", "/").strip()
    path, query_string = host_val.split("?", 1) if "?" in host_val else (host_val or "/", "")

    body = fields.get("body", "").strip()

    headers: dict[str, str] = {}
    for src, hdr in [
        ("content-type",    "Content-Type"),
        ("user-agent",      "User-Agent"),
        ("referer",         "Referer"),
        ("cookie",          "Cookie"),
        ("accept",          "Accept"),
        ("accept-encoding", "Accept-Encoding"),
        ("accept-language", "Accept-Language"),
    ]:
        val = fields.get(src, "").strip()
        if val:
            headers[hdr] = val

    return HttpRecord(
        method=method,
        path=path or "/",
        query_string=query_string,
        headers=json.dumps(headers),
        body=body,
        label=label,
        attack_class=attack_class,
        source=source,
    ).build_raw()


def _adapt_sr_bh_v1(
    csv_path:    Path,
    label_col:   str,
    source_name: str,
) -> Iterator[HttpRecord]:
    df               = pd.read_csv(csv_path, low_memory=False)
    actual_label_col = label_col if label_col in df.columns else _find_label_col(df)
    labels           = _extract_label(df[actual_label_col])

    if "category" in df.columns:
         attack_classes = (                                                                      
                  df["category"]                                                                      
                  .astype(str)                                                                        
                  .str.lower()                                                                        
                  .str.strip()                                                                        
                  .str.replace(r'^\d+\s*[-–]\s*', '', regex=True)                                     
                  .str.strip()                                                                        
                  .tolist()                                                                           
              )  
    else:
        attack_classes = _extract_attack_class(df, labels)

    texts = df["text"].fillna("").astype(str).tolist()

    for text, lbl, ac in zip(texts, labels, attack_classes):
        if not text.strip():
            continue
        try:
            yield _parse_sr_bh_text(text, lbl, ac, source_name)
        except Exception as e:
            log.debug(f"Parse error ({source_name}): {e}")

# HTTP envelope for wrapping bare parameter payloads: canonical_class → (method, path, param_name)
_HTTP_PARAMS_ENVELOPE: dict[str, tuple[str, str, str]] = {
    "sqli":             ("GET",  "/search",   "q"),
    "xss":              ("GET",  "/search",   "q"),
    "lfi":              ("GET",  "/view",     "file"),
    "rfi":              ("GET",  "/include",  "file"),
    "cmdi":             ("GET",  "/exec",     "cmd"),
    "path_traversal":   ("GET",  "/download", "path"),
    "ssrf":             ("GET",  "/fetch",    "url"),
    "header_injection": ("POST", "/submit",   "data"),
    "ssti":             ("GET",  "/render",   "template"),
    "xxe":              ("POST", "/api/xml",  "data"),
    "benign":           ("GET",  "/search",   "q"),
    "unknown":          ("GET",  "/search",   "q"),
}


def _adapt_http_params_v1(
    csv_path:    Path,
    label_col:   str,
    source_name: str,
) -> Iterator[HttpRecord]:
    """
    HTTP Params Dataset adapter.

    The payload column (sentence/payload) contains a raw attack or benign string.
    Each value is wrapped in a synthetic HTTP request envelope (GET query param or
    POST form body) chosen by canonical attack class so the tokenizer sees
    realistic HTTP structure.
    """
    df = pd.read_csv(csv_path, low_memory=False)

    payload_col = next(
        (c for c in ("sentence", "Sentence", "payload", "Payload",
                     "text", "Text", "request", "Request")
         if c in df.columns),
        None,
    )
    if payload_col is None:
        raise ValueError(
            f"HTTP Params adapter: no payload column found. Columns: {list(df.columns)}"
        )

    type_col = next(
        (c for c in ("type", "Type", "attack_type", "attack_class",
                     "category", "Category", "class", "Class")
         if c in df.columns),
        None,
    )

    actual_label_col = label_col if label_col in df.columns else _find_label_col(df)
    labels   = _extract_label(df[actual_label_col])
    payloads = df[payload_col].fillna("").astype(str).tolist()
    raw_types = (
        df[type_col].fillna("").astype(str).str.lower().str.strip().tolist()
        if type_col else ["benign" if lbl == 0 else "unknown" for lbl in labels]
    )

    for payload, lbl, raw_type in zip(payloads, labels, raw_types):
        payload = payload.strip()
        if not payload:
            continue

        canonical = _canonical_class(raw_type) if raw_type else ("benign" if lbl == 0 else "unknown")
        method, path, param = _HTTP_PARAMS_ENVELOPE.get(
            canonical, _HTTP_PARAMS_ENVELOPE["unknown"]
        )

        if method == "GET":
            record = HttpRecord(
                method=method,
                path=path,
                query_string=f"{param}={payload}",
                headers=json.dumps({"Host": "localhost", "Accept": "*/*"}),
                body="",
                label=lbl,
                attack_class=canonical,
                source=source_name,
            )
        else:
            record = HttpRecord(
                method=method,
                path=path,
                query_string="",
                headers=json.dumps({
                    "Host": "localhost",
                    "Content-Type": "application/x-www-form-urlencoded",
                }),
                body=f"{param}={payload}",
                label=lbl,
                attack_class=canonical,
                source=source_name,
            )

        try:
            yield record.build_raw()
        except Exception as e:
            log.debug(f"Build error ({source_name}): {e}")


ADAPTER_REGISTRY: dict[str, Callable[[Path, str, str], Iterator[HttpRecord]]] = {
    "csic_v1":          _adapt_csic_v1,
    "sr_bh_v1":         _adapt_sr_bh_v1,
    "http_params_v1":   _adapt_http_params_v1,
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

    extraction_dir: Path | None = None
    archive_file:   Path | None = None

    local_copy = _check_local_copy(raw_dir, spec)

    if local_copy and not force:
        archive_file = local_copy
    elif spec.kaggle_handle:
        log.info(f"  Trying downloading from kagglehub: {spec.kaggle_handle}")
        try:
            kaggle_result = Path(kagglehub.dataset_download(spec.kaggle_handle))
            log.info(f"  kagglehub returned temporary file: {kaggle_result}")
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
    log.info(f"  Converting to csv → {out_csv.name}")
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
    configure_root() # for logging
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)
    timer = StepTimer()

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
        # log.info(f"Found configured dataset: {ds_cfg.name}, id: {conv_id}, converter: {converter}, adapter: {adapter}")

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

    # ── Per-dataset: acquire → adapt → Normalize → write ─────────────────────
    manifest:          list[dict]      = []
    per_dataset_stats: dict[str, dict] = {}
    all_parquet_paths: list[Path]      = []
    failed_acquire:    list[str]       = []

    for spec in specs:
        log.info(f" =========== Dataset {spec.name}:")
        log.info(f"{' '*8}==== Phase 1/2 — Acquire (download → CSV)")
        # Phase 1 — Acquire (download → CSV)
        try:
            with timer.step(f"acquire_{spec.name}"):
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

        # Phase 2 — Adapt + Normalize + Write (single streaming pass)
        log.info(f"{' '*8}==== Phase 2/2 — Adapt + Normalize + Write ")
        csv_path    = raw_dir / spec.output_csv
        parquet_out = normalized_dir / f"{spec.name}.parquet"

        if parquet_out.exists() and not args.force and not args.force_ingest:
            try:
                tbl = pq.read_table(parquet_out, columns=["label"])
                n_t = tbl.num_rows
            except Exception:
                n_t = 0
            if n_t > 0:
                n_b = int(pc.sum(pc.equal(tbl["label"], 0)).as_py()) 
                n_m = n_t - n_b
                log.info(
                    f"[{spec.name}] Parquet exists ({n_t:,} rows) — skipping ingestion "
                    "(use --force-ingest)"
                )
                per_dataset_stats[spec.name] = {"n_total": n_t, "n_benign": n_b, "n_malicious": n_m}
                all_parquet_paths.append(parquet_out)
                continue
            log.info(f"[{spec.name}] Parquet exists but is empty — re-ingesting from CSV")

        log.info(f"  Ingesting + normalising [{spec.name}] …")
        try:
            with timer.step(f"normalize_{spec.name}"):
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
    total = benign = malicious = 0
    if all_parquet_paths:
        log.info(f"\nMerging {len(all_parquet_paths)} Parquet file(s)…")
        tables = []
        for p in all_parquet_paths:
            try:
                tables.append(pq.read_table(p))
            except Exception as e:
                log.warning(f"  Could not read {p.name} for merge: {e}")

        if tables:
            with timer.step("merge_parquets"):
                merged      = pa.concat_tables(tables)
                merged_path = normalized_dir / "all_datasets.parquet"
                pq.write_table(merged, merged_path, compression="snappy")
            total     = merged.num_rows
            benign    = int(pc.sum(pc.equal(merged["label"], 0)).as_py()) 
            malicious = total - benign
            log.info(f"Merged: {merged_path} ({total:,} rows)")
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
        "timings_s":   timer.timings,
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
                 "datasets":     json.dumps([s.name for s in specs]),                                       
                 "only_filter":  json.dumps(args.only) if args.only else "all",                             
                 "force":        str(args.force),                                                           
                 "force_ingest": str(args.force_ingest),   
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
            timer.log_mlflow()
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True) 

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info(f"\n{'═'*60}")
    log.info(f"Acquired: {len(specs) - len(failed_acquire)}/{len(specs)}")
    for entry in manifest:
        name = entry["name"]
        icon = "✓" if entry.get("status") in ("downloaded", "skipped") else "✗"
        n_total = per_dataset_stats.get(name, {}).get("n_total", 0)
        rows = f"  {n_total:,} rows" if n_total else ""
        log.info(f"  {icon} {name:20s}  [{entry.get('status', '?')}]{rows}")

    if failed_acquire:
        log.error(
            f"\n{len(failed_acquire)} dataset(s) failed: {failed_acquire}\n"
            "Place files manually in data/raw/downloads/ and re-run with --force."
        )
        sys.exit(1)

    log.info("\nAll datasets ready. Run Stage 2 (feature extraction) next.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 1 — Download, ingest, and Normalize WAF datasets",
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