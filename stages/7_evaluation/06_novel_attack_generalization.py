"""
stages/7_evaluation/06_novel_attack_generalization.py
---------------------------------------------------
Stage 7.4c — Evaluate generalisation to novel attack families not in training.

Enhancement (critique §1):
  Replaces hardcoded sample strings with a grammar-based fuzzer that generates
  500–1000 syntactic variations per attack class.  Confidence intervals (95%,
  Wilson score) are computed over the resulting detection rates.

  Attack grammars are defined as lightweight production rules so researchers
  can extend coverage without modifying core evaluation logic.

Run:
    python stages/7_evaluation/06_novel_attack_generalization.py \
        --config config/pipeline.yaml \
        [--n-samples 500]              # variations per attack class
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import string
import urllib.parse
from pathlib import Path
from typing import Callable

import torch

from ai_waf_v2.data.schema import HttpRecord
from ai_waf_v2.tokenizer.http_tokenizer import HttpTokenizer
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Grammar-based fuzzer primitives
# ─────────────────────────────────────────────────────────────────────────────

class Rng:
    """Thin wrapper around random.Random for reproducibility."""
    def __init__(self, seed: int) -> None:
        self._r = random.Random(seed)

    def choice(self, seq):   return self._r.choice(seq)
    def randint(self, a, b): return self._r.randint(a, b)
    def random(self):        return self._r.random()
    def sample(self, pop, k): return self._r.sample(pop, k)

    def hex(self, n: int = 4) -> str:
        return "".join(self._r.choice("0123456789abcdef") for _ in range(n))

    def rand_str(self, length: int = 8, charset: str = string.ascii_lowercase) -> str:
        return "".join(self._r.choice(charset) for _ in range(length))

    def url_encode(self, s: str) -> str:
        return urllib.parse.quote(s, safe="")

    def rand_jwt_alg(self) -> str:
        return self._r.choice(["none", "HS256", "RS256", "HS512"])

    def rand_base64(self, payload: dict) -> str:
        import base64, json as _json
        raw = _json.dumps(payload).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


# ─────────────────────────────────────────────────────────────────────────────
# Attack grammars
# Each grammar is a Callable[[Rng], str] that produces one raw HTTP request.
# ─────────────────────────────────────────────────────────────────────────────

def _graphql_injection(rng: Rng) -> str:
    """
    Grammar covers:
      - Field enumeration via __schema / __type introspection
      - Mutation abuse (delete / update / create verbs)
      - Nested object traversal with injected field selection
      - Batch query arrays (DoS / BOLA surface)
      - Variable exfiltration via aliases
    """
    # Random field name
    resource  = rng.choice(["user", "admin", "order", "payment", "session"])
    field_id  = rng.randint(1, 999)
    exfil_fld = rng.choice(["email", "password", "ssn", "token", "secret"])
    mutation  = rng.choice(["deleteUser", "updateRole", "createAdmin", "resetPassword"])
    intro_typ = rng.choice(["__schema{types{name,fields{name}}}", "__type(name:\"User\"){fields{name,type{name}}}"])

    variants = [
        # Simple introspection
        f"GET /graphql?query={{{intro_typ}}}",
        # Field enum with id
        f"GET /graphql?query={{{resource}(id:\"{field_id}\"){{{exfil_fld},id}}}}",
        # Mutation
        f"POST /graphql\r\nContent-Type: application/json\r\n\r\n"
        f"{{\"query\":\"mutation{{{mutation}(id:{field_id})}}\",\"variables\":{{}}}}",
        # Batch
        f"POST /graphql\r\nContent-Type: application/json\r\n\r\n"
        f"[{{\"query\":\"{{user(id:{field_id}){{{exfil_fld}}}}}\"}},{{\"query\":\"{{__typename}}\"}}]",
        # Alias exfil
        f"GET /graphql?query={{alias:{resource}(id:{field_id}){{{exfil_fld}}}}}",
    ]
    base = rng.choice(variants)

    # Apply random encoding mutation (adds syntactic variety)
    enc_mode = rng.choice(["none", "url", "unicode"])
    if enc_mode == "url":
        base = rng.url_encode(base)
    elif enc_mode == "unicode":
        base = base.replace("{", "%7B").replace("}", "%7D")
    return base


def _jwt_manipulation(rng: Rng) -> str:
    """
    Grammar covers:
      - alg=none bypass (no signature)
      - HS256/RS256 confusion
      - Tampered payload (role/sub/exp claims)
      - kid injection (SQL / path traversal in key-id)
    """
    path = rng.choice(["/api/user", "/api/admin", "/dashboard", "/api/v2/profile"])
    alg  = rng.rand_jwt_alg()

    hdr     = rng.rand_base64({"alg": alg, "typ": "JWT"})
    payload = rng.rand_base64({
        "sub":  rng.rand_str(8),
        "role": rng.choice(["admin", "superuser", "root", "god"]),
        "exp":  9999999999,
    })
    sig = "" if alg == "none" else rng.hex(32)
    token = f"{hdr}.{payload}.{sig}"

    variants = [
        f"GET {path}\r\nAuthorization: Bearer {token}",
        f"GET {path}\r\nX-Auth-Token: {token}",
        f"GET {path}?token={token}",
    ]
    # kid injection variant
    hdr_kid = rng.rand_base64({
        "alg": "HS256",
        "kid": rng.choice([
            "../../etc/passwd",
            f"1' OR '1'='1",
            f"| ls /",
        ]),
    })
    token_kid = f"{hdr_kid}.{payload}.{rng.hex(32)}"
    variants.append(f"GET {path}\r\nAuthorization: Bearer {token_kid}")

    return rng.choice(variants)


def _http2_desync(rng: Rng) -> str:
    """
    Grammar covers:
      - CL.TE and TE.CL desync patterns
      - Varying smuggled method / path / payload size
    """
    smuggled_path = rng.choice(["/admin", "/internal", "/.git/config",
                                 "/api/private", "/debug"])
    body_chunk    = rng.hex(rng.randint(1, 8))
    cl_val        = rng.randint(3, 8)

    variants = [
        # CL.TE
        (
            f"POST / HTTP/1.1\r\n"
            f"Host: target\r\n"
            f"Content-Length: {cl_val}\r\n"
            f"Transfer-Encoding: chunked\r\n\r\n"
            f"0\r\n\r\n"
            f"GET {smuggled_path} HTTP/1.1\r\n"
            f"Host: target\r\n\r\n"
        ),
        # TE.CL
        (
            f"POST / HTTP/1.1\r\n"
            f"Host: target\r\n"
            f"Transfer-Encoding: chunked\r\n"
            f"Content-Length: {rng.randint(1, 4)}\r\n\r\n"
            f"{len(body_chunk):x}\r\n"
            f"{body_chunk}\r\n"
            f"0\r\n\r\n"
        ),
        # Obfuscated TE header
        (
            f"POST / HTTP/1.1\r\n"
            f"Host: target\r\n"
            f"Transfer-Encoding : chunked\r\n"     # space before colon
            f"Content-Length: {cl_val}\r\n\r\n"
            f"0\r\n\r\n"
            f"GET {smuggled_path} HTTP/1.1\r\nFoo: bar\r\n\r\n"
        ),
    ]
    return rng.choice(variants)


def _prototype_pollution(rng: Rng) -> str:
    """
    Grammar covers:
      - __proto__ in JSON body (merge / assign patterns)
      - constructor.prototype in query-string
      - Nested depth variations
      - Content-type spoofing
    """
    admin_key = rng.choice(["admin", "isAdmin", "role", "privilege", "superuser"])
    value     = rng.choice(["true", "1", "\"admin\"", "{}"])
    path      = rng.choice(["/api/merge", "/api/extend", "/api/update",
                             "/utils/assign", "/api/v1/settings"])

    variants = [
        # Flat __proto__
        f"POST {path}\r\nContent-Type: application/json\r\n\r\n"
        f"{{\"__proto__\":{{\"{admin_key}\":{value}}}}}",
        # Nested constructor.prototype
        f"POST {path}\r\nContent-Type: application/json\r\n\r\n"
        f"{{\"constructor\":{{\"prototype\":{{\"{admin_key}\":{value}}}}}}}",
        # Query-string vector
        f"GET {path}?__proto__[{admin_key}]={value}",
        f"GET {path}?constructor[prototype][{admin_key}]={value}",
        # URL-encoded
        f"POST {path}\r\nContent-Type: application/x-www-form-urlencoded\r\n\r\n"
        f"__proto__[{admin_key}]={value}",
        # Deep nesting
        f"POST {path}\r\nContent-Type: application/json\r\n\r\n"
        f"{{\"a\":{{\"__proto__\":{{\"{admin_key}\":{value}}}}}}}",
    ]
    return rng.choice(variants)


def _ssrf_injection(rng: Rng) -> str:
    """SSRF via open-redirect parameters and URL scheme abuse."""
    internal_ip  = f"192.168.{rng.randint(0,255)}.{rng.randint(1,254)}"
    internal_host = rng.choice(["metadata.google.internal", "169.254.169.254",
                                 "localhost", "127.0.0.1", internal_ip])
    param        = rng.choice(["url", "redirect", "next", "callback", "dest", "path"])
    port         = rng.choice(["80", "443", "8080", "9200", "6379", "5432"])
    scheme       = rng.choice(["http", "https", "dict", "file", "gopher"])
    endpoint     = rng.choice(["/latest/meta-data/iam/security-credentials/",
                                "/etc/passwd", "/api/internal/admin"])

    target = f"{scheme}://{internal_host}:{port}{endpoint}"
    encoded = rng.choice([target, rng.url_encode(target),
                           target.replace(".", "%2e"), target.replace("/", "%2f")])

    path = rng.choice(["/fetch", "/api/proxy", "/api/request", "/webhook"])
    return f"GET {path}?{param}={encoded}\r\nHost: target"


def _xxe_injection(rng: Rng) -> str:
    """XXE via DTD external entity declarations in XML body."""
    entity_name = rng.rand_str(6)
    target_file = rng.choice(["/etc/passwd", "/etc/shadow", "C:/Windows/win.ini",
                               "/proc/self/environ"])
    wrapper     = rng.choice(["<!DOCTYPE", "<?xml"])

    doc = (
        f'POST /api/xml HTTP/1.1\r\n'
        f'Content-Type: application/xml\r\n\r\n'
        f'<?xml version="1.0"?>\r\n'
        f'<!DOCTYPE foo [\r\n'
        f'  <!ENTITY {entity_name} SYSTEM "file://{target_file}">\r\n'
        f']>\r\n'
        f'<foo>&{entity_name};</foo>'
    )
    return doc


# Registry of all grammar functions
ATTACK_GRAMMARS: dict[str, Callable[[Rng], str]] = {
    "graphql_injection":     _graphql_injection,
    "jwt_manipulation":      _jwt_manipulation,
    "http2_desync":          _http2_desync,
    "prototype_pollution":   _prototype_pollution,
    "ssrf_injection":        _ssrf_injection,
    "xxe_injection":         _xxe_injection,
}


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score confidence interval for a proportion.

    Critique §1: provides statistically meaningful detection probability bounds
    rather than a single point estimate.
    """
    if n_total == 0:
        return 0.0, 0.0
    p    = n_success / n_total
    denom = 1 + z ** 2 / n_total
    centre = (p + z ** 2 / (2 * n_total)) / denom
    half   = z * math.sqrt(p * (1 - p) / n_total + z ** 2 / (4 * n_total ** 2)) / denom
    return max(0.0, round(centre - half, 5)), min(1.0, round(centre + half, 5))


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def _batch_predict(
    model: torch.nn.Module,
    texts: list[str],
    tokenizer,
    seq_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> list[int]:
    """Tokenise and run inference for a list of raw HTTP strings."""
    pad_id = tokenizer.pad_token_id
    all_preds: list[int] = []

    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        ids_list  = []
        mask_list = []
        for text in chunk:
            enc   = tokenizer.encode(text)
            ids   = enc.ids[:seq_len]
            pad   = seq_len - len(ids)
            ids_list.append(ids + [pad_id] * pad)
            mask_list.append([1] * len(ids) + [0] * pad)

        ids_t  = torch.tensor(ids_list,  dtype=torch.long).to(device)
        mask_t = torch.tensor(mask_list, dtype=torch.long).to(device)

        with torch.no_grad():
            preds, _ = model.predict(ids_t, mask_t)
        all_preds.extend(preds.cpu().tolist())

    return all_preds


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = HttpTokenizer.load(
        cfg.tokenizer.track_b.output_dir,
        cfg.tokenizer.seq_len,
    )

    ckpt = Path(cfg.model.track_b_99m.output_dir) / "best_99m.pt"
    if not ckpt.exists():
        log.error("Teacher checkpoint not found")
        return

    from ai_waf_v2.models.head import WafClassifier
    model = WafClassifier.load(ckpt, cfg.model.track_b_99m, map_location=str(device))
    model.to(device).eval()

    n_samples = args.n_samples
    seed      = args.seed
    results: dict[str, dict] = {}

    log.info(
        f"Generating {n_samples} samples per attack class "
        f"(seed={seed}, {len(ATTACK_GRAMMARS)} classes)"
    )

    for attack_name, grammar_fn in ATTACK_GRAMMARS.items():
        rng = Rng(seed)   # fresh RNG per class for reproducibility

        # Generate variations
        raw_texts = [grammar_fn(rng) for _ in range(n_samples)]

        # Deduplicate (grammar may occasionally produce identical strings)
        unique_texts = list(dict.fromkeys(raw_texts))
        if len(unique_texts) < n_samples:
            log.debug(f"  {attack_name}: {n_samples - len(unique_texts)} duplicates removed")

        # Inference
        preds = _batch_predict(
            model, unique_texts, tokenizer,
            cfg.tokenizer.seq_len, device,
        )

        n_detected = sum(preds)
        n_total    = len(preds)
        detection_rate = n_detected / max(1, n_total)
        ci_low, ci_high = _wilson_ci(n_detected, n_total)

        # Analyse edge-case failures (samples not detected)
        evasion_examples = [
            unique_texts[i] for i, p in enumerate(preds) if p == 0
        ][:5]   # at most 5 representative evasion examples

        results[attack_name] = {
            "n_generated":      n_samples,
            "n_unique":         n_total,
            "n_detected":       n_detected,
            "detection_rate":   round(detection_rate, 5),
            "evasion_rate":     round(1 - detection_rate, 5),
            "ci_95_low":        ci_low,
            "ci_95_high":       ci_high,
            "evasion_examples": evasion_examples,
        }

        log.info(
            f"  {attack_name:28s}: detection={detection_rate:.4f}  "
            f"CI=[{ci_low:.4f}, {ci_high:.4f}]  "
            f"n={n_total}"
        )

    out = Path(cfg.paths.reports) / "metrics" / "novel_attack_generalization.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info(f"\nNovel attack generalisation saved to {out}")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="06_novel_attack_generalization"):
            mlflow.log_params({
                "n_samples_per_class": n_samples,
                "seed":                seed,
                "n_attack_classes":    len(ATTACK_GRAMMARS),
            })
            metrics: dict[str, float] = {}
            detection_rates = []
            for attack_name, info in results.items():
                dr = info["detection_rate"]
                detection_rates.append(dr)
                metrics[f"detection_{attack_name}"]  = float(dr)
                metrics[f"evasion_{attack_name}"]    = float(info["evasion_rate"])
                metrics[f"ci_low_{attack_name}"]     = float(info["ci_95_low"])
                metrics[f"ci_high_{attack_name}"]    = float(info["ci_95_high"])
            if detection_rates:
                metrics["mean_detection_rate"] = float(sum(detection_rates) / len(detection_rates))
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(out))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",    default="config/pipeline.yaml")
    p.add_argument("--n-samples", type=int, default=500,
                   help="Fuzzing variations per attack class (≥500 recommended)")
    p.add_argument("--seed",      type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())