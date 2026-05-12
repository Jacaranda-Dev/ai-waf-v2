"""
stages/3_data_augmentation/02_request_framing.py
-------------------------------------------------
Module B — Unified HTTP Request Framing Engine

Merges: 08_api_llm_framing, 09_benign_rest_traffic

Architecture:
  - Decouples payload generation from HTTP envelope construction
  - Single Jinja2-based template engine for ALL request types (attack + benign)
    → Ensures grammar-generated and LLM-generated payloads share the same
      header/path distribution, preventing the model from learning "synthetic" fingerprints
  - Consumes raw payload strings from Module A (01_attack_synthesis.py)
  - Also generates benign traffic via programmatic REST patterns + cloud LLM
  - Cloud LLM used ONLY for HTTP framing context, never for raw attack payloads

Run:
    python stages/3_data_augmentation/02_request_framing.py \
        --config config/pipeline.yaml [--provider anthropic|google]
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path
from string import Template

import pyarrow.parquet as pq

from ai_waf_v2.data.schema import HttpRecord, records_to_table
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP metadata distributions
# (Single source of truth for both attack and benign framing)
# ─────────────────────────────────────────────────────────────────────────────

class HttpMetadataDistribution:
    """
    Encapsulates the realistic header/path/UA distributions used to frame
    BOTH attack and benign HTTP records.  Keeping this in one place ensures
    the model cannot distinguish attack vs. benign based on synthetic metadata.
    """

    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15",
        "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36",
        "PostmanRuntime/7.37.0",
        "python-requests/2.31.0",
        "axios/1.6.8",
        "okhttp/4.12.0",
        "Go-http-client/2.0",
    ]

    ACCEPT_HEADERS = [
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "application/json, text/plain, */*",
        "application/json",
        "*/*",
        "text/html, */*",
    ]

    LANGUAGES = ["en-US,en;q=0.9", "en-GB,en;q=0.8", "es-ES,es;q=0.7,en;q=0.3", "fr-FR,fr;q=0.9"]

    REFERERS = [
        "https://www.google.com/",
        "https://www.bing.com/",
        "https://duckduckgo.com/",
        "https://app.example.com/dashboard",
        "https://app.example.com/search",
        "",   # direct / no referrer
    ]

    HOST_VARIANTS = [
        "api.example.com",
        "app.example.com",
        "www.example.com",
        "staging.example.com",
        "example.com",
    ]

    AUTH_PATTERNS = [
        lambda r: f"Bearer eyJhbGciOiJIUzI1NiJ9.{uuid.uuid4().hex[:24]}.sig",
        lambda r: f"Token {uuid.uuid4().hex}",
        lambda r: "",   # unauthenticated
        lambda r: f"Basic {uuid.uuid4().hex[:16]}",
    ]

    CONTENT_TYPES = [
        "application/x-www-form-urlencoded",
        "application/json",
        "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        "text/plain",
    ]

    def sample_headers(
        self,
        rng:          random.Random,
        method:       str,
        has_body:     bool = False,
        extra:        dict | None = None,
    ) -> dict:
        auth_fn = rng.choice(self.AUTH_PATTERNS)
        h = {
            "Host":             rng.choice(self.HOST_VARIANTS),
            "User-Agent":       rng.choice(self.USER_AGENTS),
            "Accept":           rng.choice(self.ACCEPT_HEADERS),
            "Accept-Language":  rng.choice(self.LANGUAGES),
            "Connection":       rng.choice(["keep-alive", "close"]),
        }
        referer = rng.choice(self.REFERERS)
        if referer:
            h["Referer"] = referer

        auth = auth_fn(rng)
        if auth:
            h["Authorization"] = auth

        if has_body:
            h["Content-Type"] = rng.choice(self.CONTENT_TYPES)

        if extra:
            h.update(extra)
        return h


DIST = HttpMetadataDistribution()   # module-level singleton


# ─────────────────────────────────────────────────────────────────────────────
# Attack request framer
# ─────────────────────────────────────────────────────────────────────────────

_ATTACK_META = {
    "sqli":  dict(params=["id", "q", "search", "user_id", "filter"],
                  endpoints=["/api/v1/users", "/api/v1/products", "/search", "/admin/users"]),
    "xss":   dict(params=["comment", "name", "message", "q", "text", "body"],
                  endpoints=["/comment", "/api/v1/comments", "/feedback", "/search", "/profile"]),
    "lfi":   dict(params=["file", "page", "path", "doc", "include"],
                  endpoints=["/index.php", "/download.php", "/view.php", "/api/file"]),
    "ssrf":  dict(params=["url", "uri", "dest", "redirect", "callback", "webhook"],
                  endpoints=["/api/v1/fetch", "/proxy", "/redirect", "/api/external"]),
    "cmdi":  dict(params=["cmd", "host", "ping", "exec", "run", "input"],
                  endpoints=["/api/ping", "/admin/exec", "/api/command", "/api/v1/run"]),
}


def frame_attack_payload(
    payload:      str,
    attack_class: str,
    source:       str,
    rng:          random.Random,
) -> HttpRecord | None:
    """Embed a raw payload string into a realistically framed HTTP request."""
    meta     = _ATTACK_META.get(attack_class, dict(
        params=["input"], endpoints=["/api/data"]
    ))
    param    = rng.choice(meta["params"])
    endpoint = rng.choice(meta["endpoints"])
    method   = rng.choice(["GET", "POST"])

    if method == "GET":
        qs, body = f"{param}={payload}", ""
    else:
        qs, body = "", f"{param}={payload}"

    headers = DIST.sample_headers(rng, method, has_body=bool(body))

    try:
        return HttpRecord(
            id=str(uuid.uuid4()),
            method=method,
            path=endpoint,
            query_string=qs,
            headers=json.dumps(headers),
            body=body,
            label=1,
            attack_class=attack_class,
            source=source,
        ).build_raw()
    except Exception as e:
        log.debug(f"Attack framing failed: {e}")
        return None


def reframe_synthesized(
    synth_path: Path,
    rng:        random.Random,
) -> list[HttpRecord]:
    """
    Re-frame records produced by Module A with the shared DIST metadata,
    replacing the stub headers baked in during synthesis.
    """
    if not synth_path.exists():
        log.warning(f"Synthesis output not found: {synth_path} — skipping attack reframing")
        return []

    records_out = []
    for row in pq.read_table(synth_path).to_pylist():
        payload      = row.get("query_string") or row.get("body") or ""
        attack_class = row.get("attack_class", "unknown")
        r = frame_attack_payload(payload, attack_class, row.get("source", "aug_reframed"), rng)
        if r:
            records_out.append(r)

    log.info(f"Reframed {len(records_out):,} attack records with shared metadata distribution")
    return records_out


# ─────────────────────────────────────────────────────────────────────────────
# Benign request generator  (programmatic REST patterns)
# ─────────────────────────────────────────────────────────────────────────────

_BENIGN_ENDPOINTS = [
    ("GET",    "/api/v1/users",          "page={p}&limit={l}"),
    ("GET",    "/api/v1/products",       "category={cat}&sort=price&order=asc"),
    ("POST",   "/api/v1/auth/login",     ""),
    ("POST",   "/api/v1/orders",         ""),
    ("GET",    "/api/v1/search",         "q={q}&page=1"),
    ("PUT",    "/api/v1/users/{id}",     ""),
    ("DELETE", "/api/v1/cart/{id}",      ""),
    ("GET",    "/api/v1/categories",     ""),
    ("POST",   "/api/v1/auth/refresh",   ""),
    ("GET",    "/api/v1/profile",        ""),
    ("GET",    "/api/v1/orders/{id}",    ""),
    ("PATCH",  "/api/v1/users/{id}",     ""),
    ("GET",    "/healthz",               ""),
    ("GET",    "/api/v1/reports",        "from={from}&to={to}"),
]

_BENIGN_QUERIES   = ["blue+shirt", "summer+sale", "size+medium", "best+sellers",
                     "new+arrivals", "red+shoes", "organic+food", "laptop+stand"]
_CATEGORIES       = ["electronics", "clothing", "furniture", "books", "sports", "toys"]
_DATE_RANGE_PAIRS = [("2024-01-01", "2024-03-31"), ("2024-04-01", "2024-06-30")]


def _make_benign_record(rng: random.Random) -> HttpRecord:
    method, path_tmpl, qs_tmpl = rng.choice(_BENIGN_ENDPOINTS)
    path = path_tmpl.replace("{id}", str(rng.randint(1, 9999)))
    from_, to_ = rng.choice(_DATE_RANGE_PAIRS)
    qs = qs_tmpl.format(
        p=rng.randint(1, 100),
        l=rng.choice([10, 20, 50]),
        cat=rng.choice(_CATEGORIES),
        q=rng.choice(_BENIGN_QUERIES),
        **{"from": from_, "to": to_},
    )

    has_body = method in ("POST", "PUT", "PATCH")
    headers  = DIST.sample_headers(rng, method, has_body=has_body)
    body     = ""
    if has_body:
        body = json.dumps({"key": str(uuid.uuid4())[:8], "value": rng.randint(1, 100)})
        headers["Content-Type"] = "application/json"

    return HttpRecord(
        id=str(uuid.uuid4()),
        method=method,
        path=path,
        query_string=qs,
        headers=json.dumps(headers),
        body=body,
        label=0,
        attack_class="benign",
        source="aug_benign_rest",
    ).build_raw()


# ─────────────────────────────────────────────────────────────────────────────
# Cloud LLM framing  (benign only — never attack payload generation)
# ─────────────────────────────────────────────────────────────────────────────

_BENIGN_SYSTEM = (
    "You are a synthetic data generator for ML model training. "
    "Generate realistic HTTP/1.1 requests from a production REST API. "
    "Output ONLY a JSON array of objects — no markdown, no explanation. "
    "Each object must have: method, path, query_string, headers (dict), body. "
    "All requests must be benign/legitimate. Vary endpoint paths, user agents, "
    "content types, auth header formats, and request patterns realistically."
)

_BENIGN_PROMPTS = [
    "Generate {n} HTTP POST requests from an iOS mobile e-commerce app. Include realistic Bearer tokens, JSON bodies with cart/order operations.",
    "Generate {n} HTTP GET requests from a browser to a REST API. Include Accept headers, session cookies, pagination params (page, limit, offset).",
    "Generate {n} HTTP requests for user auth flow: login, token refresh, logout, password reset. Include CSRF tokens and realistic form bodies.",
    "Generate {n} HTTP GET requests to a search endpoint where the query contains SQL words like SELECT, ORDER, GROUP used in natural English — NOT injections.",
    "Generate {n} HTTP POST requests for a GraphQL API. Include JSON body with 'query' field containing legitimate GraphQL.",
    "Generate {n} HTTP requests for an admin dashboard: user management, reports, config. Include admin auth headers.",
]


def _call_llm(
    provider: str,
    system:   str,
    user:     str,
    model:    str,
    max_tok:  int,
) -> list[dict] | None:
    def _parse(text: str) -> list[dict] | None:
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
        try:
            r = json.loads(text)
            return r if isinstance(r, list) else ([r] if isinstance(r, dict) else None)
        except json.JSONDecodeError:
            return None

    if provider == "anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg    = client.messages.create(
                model=model, max_tokens=max_tok,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return _parse(msg.content[0].text)
        except Exception as e:
            log.warning(f"Anthropic API error: {e}")
            return None
    else:  # google
        try:
            import google.generativeai as genai
            genai.configure()
            gm  = genai.GenerativeModel(model_name=model, system_instruction=system)
            rsp = gm.generate_content(user, generation_config={"max_output_tokens": max_tok})
            return _parse(rsp.text)
        except Exception as e:
            log.warning(f"Google API error: {e}")
            return None


def _llm_dict_to_record(d: dict) -> HttpRecord | None:
    """Convert an LLM-generated dict → HttpRecord using the shared DIST for missing fields."""
    try:
        headers = d.get("headers", {})
        if not isinstance(headers, dict):
            headers = {}
        # Ensure LLM-generated records share the same host/UA distribution
        if "Host" not in headers:
            headers["Host"] = random.choice(DIST.HOST_VARIANTS)
        if "ua" in d:
            headers["User-Agent"] = d["ua"]

        return HttpRecord(
            id=str(uuid.uuid4()),
            method=str(d.get("method", "GET")).upper(),
            path=str(d.get("path", "/")),
            query_string=str(d.get("query_string", "")),
            headers=json.dumps(headers),
            body=str(d.get("body", "")),
            label=0,
            attack_class=str(d.get("attack_class", "benign")),
            source="aug_api_llm_benign",
        ).build_raw()
    except Exception:
        return None


def generate_llm_benign(
    provider:  str,
    model:     str,
    max_tok:   int,
    n_benign:  int,
    n_edge:    int,
    rng:       random.Random,
) -> list[HttpRecord]:
    """Generate benign (+ edge-case benign) HTTP records via the cloud LLM."""
    BATCH = 10
    records: list[HttpRecord] = []

    log.info(f"LLM benign generation: target={n_benign:,}, edge_case={n_edge:,} via {provider}")

    # Standard benign
    while len(records) < n_benign:
        prompt = rng.choice(_BENIGN_PROMPTS).format(n=BATCH)
        result = _call_llm(provider, _BENIGN_SYSTEM, prompt, model, max_tok)
        if result:
            for d in result:
                r = _llm_dict_to_record(d)
                if r:
                    records.append(r)
        else:
            time.sleep(1.0)
        time.sleep(0.2)

    edge_records: list[HttpRecord] = []
    edge_prompts = [
        f"Generate {BATCH} HTTP GET requests where search queries contain SELECT, DROP, UNION or SLEEP in natural English sentences — NOT SQL injections.",
        f"Generate {BATCH} HTTP POST requests where JSON body contains HTML-like content that is NOT XSS, e.g. blog posts with <em> or <strong> tags.",
        f"Generate {BATCH} HTTP GET requests to file endpoints with relative paths like ../docs/readme.md — legitimate paths, NOT path traversals.",
    ]
    while len(edge_records) < n_edge:
        prompt = rng.choice(edge_prompts)
        result = _call_llm(provider, _BENIGN_SYSTEM, prompt, model, max_tok)
        if result:
            for d in result:
                r = _llm_dict_to_record(d)
                if r:
                    object.__setattr__(r, "attack_class", "benign_edge_case")
                    object.__setattr__(r, "source", "aug_api_llm_edge")
                    edge_records.append(r)
        time.sleep(0.3)

    combined = records[:n_benign] + edge_records[:n_edge]
    log.info(f"LLM benign: {len(records[:n_benign]):,} standard + {len(edge_records[:n_edge]):,} edge-case")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg     = load_config(args.config)
    aug_cfg = cfg.augmentation
    rng     = random.Random(cfg.project.seed)

    out_dir = Path(cfg.paths.data_augmented) / "framed"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[HttpRecord] = []

    # ── 1. Re-frame Module A attack payloads ──────────────────────────────
    synth_path = Path(cfg.paths.data_augmented) / "synthesis" / "synthesized_attacks.parquet"
    attack_records = reframe_synthesized(synth_path, rng)
    all_records.extend(attack_records)

    # ── 2. Programmatic benign REST traffic ───────────────────────────────
    n_rest = getattr(getattr(aug_cfg, "benign", None), "rest_samples", 10_000)
    benign_rest = [_make_benign_record(rng) for _ in range(n_rest)]
    all_records.extend(benign_rest)
    log.info(f"Generated {len(benign_rest):,} programmatic benign REST records")

    # ── 3. Optional: Cloud LLM benign framing ─────────────────────────────
    api_llm_cfg = getattr(aug_cfg, "api_llm", None)
    provider    = args.provider or (api_llm_cfg.provider if api_llm_cfg else None)

    if provider and api_llm_cfg:
        llm_records = generate_llm_benign(
            provider  = provider,
            model     = api_llm_cfg.model,
            max_tok   = api_llm_cfg.max_tokens,
            n_benign  = api_llm_cfg.samples_benign,
            n_edge    = api_llm_cfg.samples_edge_case,
            rng       = rng,
        )
        all_records.extend(llm_records)
    else:
        log.info("Cloud LLM framing skipped (no provider configured)")

    # ── 4. Write ──────────────────────────────────────────────────────────
    out_path = out_dir / "framed_records.parquet"
    pq.write_table(records_to_table(all_records), out_path, compression="snappy")

    n_attack = sum(1 for r in all_records if r.label == 1)
    n_benign = sum(1 for r in all_records if r.label == 0)
    log.info(
        f"Framing complete: {len(all_records):,} total records → {out_path}\n"
        f"  attack={n_attack:,}  benign={n_benign:,}"
    )

    stats_path = Path(cfg.paths.reports) / "metrics" / "request_framing.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({
        "n_attack_reframed":    len(attack_records),
        "n_benign_rest":        len(benign_rest),
        "n_llm_benign":         len(all_records) - len(attack_records) - len(benign_rest),
        "total":                len(all_records),
        "provider":             provider,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="config/pipeline.yaml")
    p.add_argument("--provider", default=None, choices=["anthropic", "google"])
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())