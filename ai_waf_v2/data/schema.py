"""
ai_waf_v2.data.schema
-----------------
Canonical record format for HTTP requests throughout the pipeline.

Every stage reads and writes Parquet files whose rows conform to
HttpRecord.  The Pydantic model is used for validation during
ingestion; PyArrow schema is used for efficient Parquet I/O.

Columns
-------
id             : str   — UUID, stable across pipeline stages
method         : str   — HTTP method (GET, POST, ...)
path           : str   — request path (/api/v1/users)
query_string   : str   — raw query string (id=1&page=2)
headers        : str   — JSON-serialised dict of header name→value
body           : str   — raw request body (empty string if none)
raw            : str   — full reconstructed HTTP/1.1 request text
                         (what the tokenizer sees)
label          : int   — 0=benign, 1=malicious
attack_class   : str   — e.g. "sqli", "xss", "benign"
source         : str   — dataset or generation method
                         (e.g. "csic_2010", "aug_rules", "aug_llm")
split          : str   — train | val | test | adversarial | canary
                         (empty until Stage 3 split)
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────
# Canonical attack-class mapping
# Single source of truth imported by all pipeline stages.
# ─────────────────────────────────────────────────────────

CLASS_ALIASES: dict[str, str] = {
    # SQL injection
    "sql injection": "sqli", "sql_injection": "sqli", "sqli": "sqli",
    # XSS
    "cross-site scripting": "xss", "cross_site_scripting": "xss", "xss": "xss",
    # File inclusion
    "local file inclusion": "lfi", "lfi": "lfi",
    "remote file inclusion": "rfi", "rfi": "rfi",
    # SSRF
    "server-side request forgery": "ssrf", "ssrf": "ssrf",
    # Command injection
    "command injection": "cmdi", "cmd injection": "cmdi", "cmdi": "cmdi",
    # XXE / SSTI / path / header
    "xml external entity": "xxe", "xxe": "xxe",
    "server-side template injection": "ssti", "ssti": "ssti",
    "path traversal": "path_traversal", "directory traversal": "path_traversal",
    "path_traversal": "path_traversal",
    "header injection": "header_injection", "header_injection": "header_injection",
    # Benign
    "normal": "benign", "legitimate": "benign", "benign": "benign",
    # SR-BH 2020 broad categories — too coarse to map to a single canonical
    # class; kept as malicious (label=1) but class is marked unknown.
    "injection": "unknown",
    "manipulation": "unknown",
    "scanning for vulnerable software": "unknown",
    "fake the source of data": "unknown",
    "http abusion": "unknown",
    # Generic ambiguous labels
    "anomalous": "unknown", "attack": "unknown", "malicious": "unknown",
    "unknown": "unknown",
}


def canonical_class(raw: str) -> str:
    """Map any noisy attack-class string to its canonical form."""
    return CLASS_ALIASES.get(str(raw).lower().strip(), "unknown")


# ─────────────────────────────────────────────────────────
# Pydantic model — used for validation during ingestion
# ─────────────────────────────────────────────────────────

class HttpRecord(BaseModel):
    """A single HTTP request with its classification label."""

    id:           str = Field(default_factory=lambda: str(uuid.uuid4()))
    method:       str = "GET"
    path:         str = "/"
    query_string: str = ""
    headers:      str = "{}"   # JSON string to keep schema flat
    body:         str = ""
    raw:          str = ""     # populated by build_raw()
    label:        int = 0      # 0=benign, 1=malicious
    attack_class: str = "benign"
    source:       str = ""
    split:        str = ""

    model_config = {"str_strip_whitespace": False}

    @field_validator("label")
    @classmethod
    def _label_range(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {v}")
        return v

    @field_validator("method")
    @classmethod
    def _method_upper(cls, v: str) -> str:
        return v.upper()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HttpRecord":
        """Build from a raw dict, tolerating extra keys."""
        known = cls.model_fields.keys()
        return cls(**{k: v for k, v in data.items() if k in known})

    def headers_dict(self) -> dict[str, str]:
        """Deserialise the JSON headers string."""
        try:
            return json.loads(self.headers)
        except (json.JSONDecodeError, TypeError):
            return {}


    def build_raw(self) -> "HttpRecord":
        """
        Reconstruct a raw HTTP/1.1 request string and store it in self.raw.
        This is what the tokenizer receives.

        Format:
            {METHOD} {path}?{query_string} HTTP/1.1\\r\\n
            {Header-Name}: {value}\\r\\n
            ...\\r\\n
            {body}
        """
        qs = f"?{self.query_string}" if self.query_string else ""
        request_line = f"{self.method} {self.path}{qs} HTTP/1.1"

        header_lines = "\r\n".join(
            f"{k}: {v}" for k, v in self.headers_dict().items()
        )

        parts = [request_line]
        if header_lines:
            parts.append(header_lines)
        parts.append("")        # blank line separating headers from body
        if self.body:
            parts.append(self.body)

        object.__setattr__(self, "raw", "\r\n".join(parts))
        return self
    


# ─────────────────────────────────────────────────────────
# PyArrow schema — used for Parquet I/O
# ─────────────────────────────────────────────────────────

PARQUET_SCHEMA = pa.schema([
    pa.field("id",           pa.string(),  nullable=False),
    pa.field("method",       pa.string(),  nullable=False),
    pa.field("path",         pa.string(),  nullable=False),
    pa.field("query_string", pa.string(),  nullable=False),
    pa.field("headers",      pa.string(),  nullable=False),
    pa.field("body",         pa.string(),  nullable=False),
    pa.field("raw",          pa.string(),  nullable=False),
    pa.field("label",        pa.int8(),    nullable=False),
    pa.field("attack_class", pa.string(),  nullable=False),
    pa.field("source",       pa.string(),  nullable=False),
    pa.field("split",        pa.string(),  nullable=False),
])

# Columns safe to use as features (exclude label metadata)
FEATURE_COLUMNS = ["raw"]
LABEL_COLUMN    = "label"
META_COLUMNS    = ["id", "method", "path", "query_string",
                   "headers", "body", "attack_class", "source", "split"]


# ─────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────

def records_to_table(records: list[HttpRecord]) -> pa.Table:
    """Convert a list of HttpRecord to a PyArrow table."""
    rows = [r.model_dump() for r in records]
    return pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA)


def table_to_records(table: pa.Table) -> list[HttpRecord]:
    """Convert a PyArrow table back to HttpRecord objects."""
    return [HttpRecord.from_dict(row) for row in table.to_pylist()]