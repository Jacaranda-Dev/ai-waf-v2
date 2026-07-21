"""
ai_waf_v2/augment/fillers.py
----------------------------
High-cardinality, label-neutral filler generators for synthetic payloads.

Grammars (and any other generator) emit **named placeholders** like ``§HOST§``
instead of hardcoded literals; ``fill_placeholders`` substitutes each from a
shared, procedural generator. The *same* generators are used for benign traffic,
so a filled token — a hostname, port, id, number — carries no signal about the
label. The model is therefore forced to key on attack *structure* (``UNION
SELECT``, ``../``, ``<script>``, shell metacharacters) rather than on incidental
constants like ``evil.com`` or ``4444`` that would be trivially evadable and would
not generalise.

This is the payload-level analogue of ``HttpMetadataDistribution`` (which shares
header/UA/method distributions across attack and benign framing).

Placeholder syntax::

    §NAME§           fresh value at every occurrence
    §NAME#TAG§       all occurrences of the same NAME#TAG within one payload
                     resolve to a single value (coreference — e.g. an SSRF that
                     rebinds to the same host, or a reverse shell that reconnects)

Only *incidental* slots become placeholders. Tokens that genuinely define the
attack class (SQL keywords, ``../``, ``/etc/passwd``, event-handler names, template
delimiters, RCE gadgets) stay as literals in the grammar so the model still learns
them.
"""

from __future__ import annotations

import random
import re
import string
from collections.abc import Callable

_LOWER = string.ascii_lowercase
_ALNUM = string.ascii_lowercase + string.digits

_TLDS = ("com", "net", "org", "io", "co", "dev", "app", "cloud", "xyz", "info",
         "biz", "site", "tech", "us", "uk", "de", "fr", "in", "ru", "cn")

# Ordinary web-ish words reused across hosts, paths and params so those tokens
# look identical in attack and benign records.
_WORDS = ("api", "cdn", "static", "assets", "img", "media", "data", "svc", "node",
          "edge", "cache", "proxy", "gw", "auth", "login", "user", "account",
          "shop", "store", "blog", "portal", "dashboard", "report", "file",
          "download", "upload", "doc", "view", "search", "query", "item",
          "product", "order", "list", "profile", "settings", "config", "session")


def _label(rng: random.Random, lo: int = 2, hi: int = 10) -> str:
    return "".join(rng.choice(_ALNUM) for _ in range(rng.randint(lo, hi)))


def rand_host(rng: random.Random) -> str:
    depth = rng.choices((1, 2, 3), weights=(3, 4, 2))[0]
    labels = [rng.choice(_WORDS) if rng.random() < 0.5 else _label(rng, 2, 10)
              for _ in range(depth)]
    return ".".join(labels) + f".{rng.choice(_WORDS)}-{_label(rng, 2, 5)}.{rng.choice(_TLDS)}"


def rand_ip(rng: random.Random) -> str:
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def rand_port(rng: random.Random) -> str:
    return str(rng.randint(1, 65535))


def rand_path(rng: random.Random) -> str:
    n = rng.randint(1, 4)
    segs = [rng.choice(_WORDS) if rng.random() < 0.6 else _label(rng, 3, 8) for _ in range(n)]
    return "/" + "/".join(segs)


def rand_param(rng: random.Random) -> str:
    return rng.choice(_WORDS) if rng.random() < 0.5 else _label(rng, 3, 8)


def rand_ident(rng: random.Random) -> str:
    return rng.choice(_LOWER) + "".join(rng.choice(_ALNUM + "_") for _ in range(rng.randint(2, 10)))


def rand_str(rng: random.Random) -> str:
    return rng.choice(_WORDS) if rng.random() < 0.5 else _label(rng, 3, 10)


def rand_int(rng: random.Random) -> str:
    return str(rng.randint(1, 99999))


def rand_col(rng: random.Random) -> str:
    common = ("id", "uid", "name", "email", "username", "password", "token", "role",
              "status", "created_at", "user_id", "account_id", "hash", "secret", "key")
    return rng.choice(common) if rng.random() < 0.7 else rand_ident(rng)


def rand_word(rng: random.Random) -> str:
    return rng.choice(_WORDS) if rng.random() < 0.5 else _label(rng, 1, 6)


def rand_js_body(rng: random.Random) -> str:
    """A JS expression whose *shape* is scripty but whose identifiers/args vary."""
    fn = rng.choice(("alert", "confirm", "prompt", "print", "eval", "fetch",
                     "setTimeout", "queueMicrotask"))
    arg = rng.choice((
        str(rng.randint(0, 9999)),
        "document.cookie", "document.domain", "location",
        f"'{_label(rng, 3, 8)}'",
        f"atob('{_label(rng, 6, 14)}')",
        f"'//{rand_host(rng)}'",
    ))
    return f"{fn}({arg})"


# name → generator.  Referenced from grammars as "§NAME§".
FILLERS: dict[str, Callable[[random.Random], str]] = {
    "HOST": rand_host,
    "IP": rand_ip,
    "PORT": rand_port,
    "PATH": rand_path,
    "PARAM": rand_param,
    "IDENT": rand_ident,
    "STR": rand_str,
    "INT": rand_int,
    "COL": rand_col,
    "WORD": rand_word,
    "JS_BODY": rand_js_body,
}

PLACEHOLDER_RE = re.compile(r"§([A-Z_][A-Z0-9_]*)(?:#([A-Za-z0-9_]+))?§")


def fill_placeholders(text: str, rng: random.Random, strict: bool = False) -> str:
    """
    Replace every ``§NAME§`` / ``§NAME#TAG§`` in ``text`` with a generated value.

    Coreference: within a single call, all placeholders sharing the same
    ``NAME#TAG`` resolve to one value. Untagged placeholders draw independently.

    Unknown names are left untouched (``strict=False``, the pipeline default, so a
    stray token can never crash synthesis) or raise ``ValueError`` (``strict=True``,
    used by tests to catch grammar typos).
    """
    if "§" not in text:
        return text
    cache: dict[tuple[str, str], str] = {}

    def _sub(m: re.Match) -> str:
        name, tag = m.group(1), m.group(2)
        gen = FILLERS.get(name)
        if gen is None:
            if strict:
                raise ValueError(f"unknown filler placeholder §{name}§")
            return m.group(0)
        if tag is not None:
            key = (name, tag)
            if key not in cache:
                cache[key] = gen(rng)
            return cache[key]
        return gen(rng)

    return PLACEHOLDER_RE.sub(_sub, text)


def placeholder_names(text: str) -> set[str]:
    """Return the set of filler names referenced in ``text`` (for validation)."""
    return {m.group(1) for m in PLACEHOLDER_RE.finditer(text)}


# ── Textbook-literal scrubbing (for LLM output, which ignores placeholders) ─────

_TEXTBOOK_HOSTS = ("evil.com", "example.com", "example.org", "attacker.com",
                   "attacker.io", "malicious.com", "victim.com", "test.com",
                   "badguy.com", "hacker.com", "evil.example", "localhost.evil")
_TEXTBOOK_RE = re.compile("|".join(re.escape(t) for t in _TEXTBOOK_HOSTS), re.IGNORECASE)


def scrub_textbook_hosts(text: str, rng: random.Random) -> str:
    """Replace well-known placeholder hostnames an LLM tends to emit with high-cardinality ones."""
    return _TEXTBOOK_RE.sub(lambda _m: rand_host(rng), text)
