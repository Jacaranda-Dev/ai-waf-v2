"""
stages/3_data_augmentation/07_local_llm_payloads.py
-----------------------------------------
Stage 2.3 — Generate attack payload variations using a locally-hosted
security-fine-tuned LLM (e.g. Llama-3-8B-Instruct fine-tuned on CVE/exploit data,
SecRoBERTa-generation, or any GGUF model via llama-cpp-python).

Runs entirely offline — no API keys, reproducible across runs.
Targets attack classes that the grammar-based approach under-covers.

Run:
    python stages/3_data_augmentation/07_local_llm_payloads.py \
        --config config/pipeline.yaml \
        --model-path /path/to/model.gguf
"""
from __future__ import annotations
import argparse, json, uuid
from pathlib import Path
import pyarrow.parquet as pq
from ai_waf_v2.data.schema import HttpRecord, records_to_table
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)

PROMPT_TEMPLATES = {
    "sqli": (
        "Generate 5 novel SQL injection payloads that evade simple keyword filters. "
        "Use advanced techniques: time-based blind, second-order, error-based. "
        "Output one payload per line, no explanation."
    ),
    "xss": (
        "Generate 5 XSS payloads that bypass CSP and WAF. "
        "Use DOM-based, mutation-based, and polyglot techniques. "
        "Output one payload per line."
    ),
    "ssrf": (
        "Generate 5 SSRF payloads targeting cloud metadata endpoints. "
        "Use IP encoding tricks, DNS rebinding patterns, and protocol wrappers. "
        "Output one payload per line."
    ),
    "cmdi": (
        "Generate 5 OS command injection payloads that bypass simple sanitization. "
        "Use shell metacharacters, environment variables, and encoding tricks. "
        "Output one payload per line."
    ),
    "lfi": (
        "Generate 5 local file inclusion payloads using path traversal variations. "
        "Include null-byte injection, encoding tricks, and PHP wrapper methods. "
        "Output one payload per line."
    ),
}

def _load_local_llm(model_path: str):
    """Load a GGUF model via llama-cpp-python."""
    try:
        from llama_cpp import Llama
        return Llama(model_path=model_path, n_ctx=512, n_gpu_layers=-1, verbose=False)
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed. "
            "pip install llama-cpp-python --extra-index-url "
            "https://abetlen.github.io/llama-cpp-python/whl/cu124"
        )

def _generate_payloads(llm, prompt: str, max_tokens: int = 256) -> list[str]:
    """Run generation and parse one payload per line."""
    response = llm(
        f"[INST] {prompt} [/INST]",
        max_tokens=max_tokens,
        temperature=0.9,
        top_p=0.95,
        stop=["\n\n", "###"],
    )
    text = response["choices"][0]["text"]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Filter lines that look like payloads (not instructions/headers)
    payloads = [l for l in lines if len(l) > 3 and not l.endswith(":")]
    return payloads

def _wrap_in_http(payload: str, attack_class: str, rng) -> HttpRecord:
    """Embed a raw payload into a minimal HTTP request."""
    import random
    endpoints = {
        "sqli":  "/api/v1/search",
        "xss":   "/api/v1/comment",
        "ssrf":  "/api/v1/fetch",
        "cmdi":  "/api/v1/ping",
        "lfi":   "/api/v1/file",
    }
    params = {"sqli": "q", "xss": "msg", "ssrf": "url", "cmdi": "host", "lfi": "path"}
    methods = {"sqli": "GET", "xss": "POST", "ssrf": "GET", "cmdi": "POST", "lfi": "GET"}

    method   = methods.get(attack_class, "GET")
    endpoint = endpoints.get(attack_class, "/api/v1/data")
    param    = params.get(attack_class, "input")

    if method == "GET":
        qs, body = f"{param}={payload}", ""
    else:
        qs, body = "", f"{param}={payload}"

    return HttpRecord(
        id=str(uuid.uuid4()), method=method,
        path=endpoint, query_string=qs, body=body,
        headers=json.dumps({"Host": "target.local",
                             "User-Agent": "Mozilla/5.0"}),
        label=1, attack_class=attack_class,
        source="aug_local_llm",
    ).build_raw()

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    import random
    rng = random.Random(cfg.project.seed)

    model_path = args.model_path or cfg.augmentation.local_llm.model_path
    if not model_path or not Path(model_path).exists():
        log.warning(
            f"Local LLM model not found at '{model_path}'. "
            "Set LOCAL_LLM_PATH in .env or pass --model-path. "
            "Skipping local LLM augmentation."
        )
        return

    n_per_class = cfg.augmentation.local_llm.samples_per_gap_class
    targets     = cfg.augmentation.grammar.targets

    log.info(f"Loading local LLM from {model_path}...")
    llm = _load_local_llm(model_path)
    log.info("Model loaded.")

    out_dir = Path(cfg.paths.data_augmented) / "local_llm"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[HttpRecord] = []
    stats: dict[str, int] = {}

    for attack_class in targets:
        if attack_class not in PROMPT_TEMPLATES:
            continue

        log.info(f"  Generating payloads for: {attack_class}")
        prompt  = PROMPT_TEMPLATES[attack_class]
        records = []

        while len(records) < n_per_class:
            payloads = _generate_payloads(llm, prompt, cfg.augmentation.local_llm.max_new_tokens)
            for payload in payloads:
                if len(records) >= n_per_class:
                    break
                records.append(_wrap_in_http(payload, attack_class, rng))

        stats[attack_class] = len(records)
        all_records.extend(records)
        log.info(f"    {attack_class}: {len(records):,} samples")

    if all_records:
        pq.write_table(
            records_to_table(all_records),
            out_dir / "local_llm_payloads.parquet",
            compression="snappy",
        )

    sp = Path(cfg.paths.reports) / "metrics" / "augmentation_local_llm.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({"total": len(all_records), "per_class": stats}, indent=2))
    log.info(f"Local LLM augmentation: {len(all_records):,} samples generated")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="config/pipeline.yaml")
    p.add_argument("--model-path", default=None)
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())