"""
stages/3_data_augmentation/08_api_llm_framing.py
--------------------------------------
Stage 2.4 — Cloud LLM (Anthropic / Google) augmentation.

Uses a cloud LLM API to generate:
  1. Realistic HTTP request framing around known attack payloads
     (adds convincing headers, user agents, session cookies)
  2. Diverse benign traffic that superficially resembles attacks
     (legitimate SQL-like queries, JSON with angle brackets, etc.)
  3. Benign edge cases that rule-based WAFs over-block

The LLM is NOT used to generate raw attack payloads — only the
surrounding HTTP context.  This avoids content filter issues while
still producing realistic training data.

Run:
    python stages/3_data_augmentation/08_api_llm_framing.py \
        --config config/pipeline.yaml \
        --provider anthropic
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import pyarrow.parquet as pq

from ai_waf_v2.data.schema import HttpRecord, records_to_table
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────

BENIGN_SYSTEM = """You are a synthetic data generator for ML model training.
Generate realistic HTTP/1.1 requests from a production REST API.
Output ONLY a JSON array of objects — no markdown, no explanation.
Each object must have: method, path, query_string, headers (dict), body.
All requests must be benign/legitimate. Vary endpoint paths, user agents,
content types, auth header formats, and request patterns realistically."""

BENIGN_USER_TEMPLATES = [
    "Generate {n} HTTP POST requests from an iOS mobile e-commerce app to a REST API. Include realistic Bearer tokens, JSON bodies with cart/order operations, and iOS User-Agent strings.",
    "Generate {n} HTTP GET requests from a web browser to a REST API. Include realistic Accept headers, session cookies, pagination parameters (page, limit, offset), and sort parameters.",
    "Generate {n} HTTP requests for a user authentication flow: login, token refresh, logout, password reset. Include CSRF tokens, Content-Type headers, and realistic form bodies.",
    "Generate {n} HTTP GET requests for a search endpoint with diverse benign search queries. Queries may include SQL keywords like SELECT, ORDER, GROUP used in natural language (e.g. 'select all products'). These are legitimate search terms, not injections.",
    "Generate {n} HTTP POST requests for a file upload API. Include multipart/form-data Content-Type, realistic filenames, and legitimate file metadata.",
    "Generate {n} HTTP requests to a GraphQL API. Include POST with JSON body containing 'query' field with legitimate GraphQL queries.",
    "Generate {n} HTTP requests for an admin dashboard: user management, report generation, configuration endpoints. Include admin auth headers.",
]

FRAMING_SYSTEM = """You are a synthetic data generator for security ML model training.
Given an attack payload, generate realistic HTTP request context (not new payloads).
Output ONLY a JSON array — no markdown, no explanation.
Each object: method, path, headers (dict), user_agent, referer.
Make the HTTP context (not the payload) look like a realistic browser or API client."""


def _call_anthropic(
    system: str,
    user: str,
    model: str,
    max_tokens: int,
) -> list[dict] | None:
    """Call Anthropic API and parse JSON response."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = message.content[0].text
        return _safe_parse_json(text)
    except Exception as e:
        log.warning(f"Anthropic API call failed: {e}")
        return None


def _call_google(
    system: str,
    user: str,
    model: str,
    max_tokens: int,
) -> list[dict] | None:
    """Call Google Gemini API and parse JSON response."""
    try:
        import google.generativeai as genai
        genai.configure()
        gmodel = genai.GenerativeModel(
            model_name=model,
            system_instruction=system,
        )
        response = gmodel.generate_content(
            user,
            generation_config={"max_output_tokens": max_tokens},
        )
        return _safe_parse_json(response.text)
    except Exception as e:
        log.warning(f"Google API call failed: {e}")
        return None


def _safe_parse_json(text: str) -> list[dict] | None:
    """Strip markdown fences and parse JSON safely."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        text = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        )
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
        return None
    except json.JSONDecodeError:
        return None


def _dict_to_record(
    d: dict,
    label: int,
    attack_class: str,
    source: str,
) -> HttpRecord | None:
    """Convert an LLM-generated dict to an HttpRecord."""
    try:
        headers = d.get("headers", {})
        if not isinstance(headers, dict):
            headers = {}
        # Inject user-agent and referer if provided at top level
        if "user_agent" in d:
            headers["User-Agent"] = d["user_agent"]
        if "referer" in d:
            headers["Referer"] = d["referer"]

        r = HttpRecord(
            id=str(uuid.uuid4()),
            method=str(d.get("method", "GET")).upper(),
            path=str(d.get("path", "/")),
            query_string=str(d.get("query_string", "")),
            headers=json.dumps(headers),
            body=str(d.get("body", "")),
            label=label,
            attack_class=attack_class,
            source=source,
        ).build_raw()
        return r
    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    aug_cfg = cfg.augmentation.api_llm

    provider = args.provider or aug_cfg.provider
    model    = aug_cfg.model
    n_benign = aug_cfg.samples_benign
    n_edge   = aug_cfg.samples_edge_case

    call_fn = _call_anthropic if provider == "anthropic" else _call_google

    out_dir = Path(cfg.paths.data_augmented) / "api_llm"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Batch size per API call (LLMs produce ~10-20 samples per call reliably)
    BATCH = 10
    all_benign:    list[HttpRecord] = []
    all_edge_case: list[HttpRecord] = []

    import random
    rng = random.Random(cfg.project.seed)

    # ── Benign traffic generation ─────────────────
    log.info(f"Generating {n_benign:,} benign samples via {provider}...")
    templates = BENIGN_USER_TEMPLATES.copy()

    while len(all_benign) < n_benign:
        template = rng.choice(templates)
        prompt   = template.format(n=BATCH)

        result = call_fn(BENIGN_SYSTEM, prompt, model, aug_cfg.max_tokens)

        if result:
            for d in result:
                r = _dict_to_record(d, label=0, attack_class="benign", source="aug_api_llm_benign")
                if r:
                    all_benign.append(r)
        else:
            time.sleep(1.0)  # back-off on failure

        if len(all_benign) % 100 == 0:
            log.info(f"  Benign: {len(all_benign):,}/{n_benign:,}")

        time.sleep(0.2)   # rate-limit courtesy delay

    # ── Edge case benign traffic (looks like attack but isn't) ──
    log.info(f"Generating {n_edge:,} benign edge-case samples...")
    edge_prompts = [
        f"Generate {BATCH} HTTP GET requests with search queries that contain words like SELECT, DROP, UNION, or SLEEP used in natural English sentences — NOT SQL injections. These are legitimate searches like 'SELECT best pizza near me' or 'how to sleep better'.",
        f"Generate {BATCH} HTTP POST requests where the JSON body contains HTML-like content that is NOT XSS — for example, a blog post body with <em> or <strong> tags, or a comment with angle bracket emoticons like <3.",
        f"Generate {BATCH} HTTP GET requests to file download endpoints with path parameters like ../docs/readme.md or ../images/photo.jpg — these are legitimate relative paths in a doc system, NOT path traversals.",
    ]

    while len(all_edge_case) < n_edge:
        prompt = rng.choice(edge_prompts).replace(str(BATCH), str(BATCH))
        result = call_fn(BENIGN_SYSTEM, prompt, model, aug_cfg.max_tokens)

        if result:
            for d in result:
                r = _dict_to_record(d, label=0, attack_class="benign_edge_case", source="aug_api_llm_edge")
                if r:
                    all_edge_case.append(r)
        time.sleep(0.3)

    # ── Write outputs ─────────────────────────────
    all_records = all_benign[:n_benign] + all_edge_case[:n_edge]
    if all_records:
        out_path = out_dir / "api_llm_augmented.parquet"
        pq.write_table(records_to_table(all_records), out_path, compression="snappy")
        log.info(
            f"Saved {len(all_records):,} LLM-generated records to {out_path}\n"
            f"  benign={len(all_benign[:n_benign]):,}  "
            f"edge_case={len(all_edge_case[:n_edge]):,}"
        )

    stats_path = Path(cfg.paths.reports) / "metrics" / "augmentation_api_llm.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({
        "n_benign":     len(all_benign[:n_benign]),
        "n_edge_case":  len(all_edge_case[:n_edge]),
        "provider":     provider,
        "model":        model,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="config/pipeline.yaml")
    p.add_argument("--provider", default=None, choices=["anthropic", "google"])
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())