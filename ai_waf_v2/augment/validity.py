"""
ai_waf_v2/augment/validity.py
-----------------------------
Per-class structural validity checks for synthetic attack payloads.

Used by the Stage 3.4 quality gate to keep ``attack_class`` labels honest: a
payload labelled ``sqli`` should still look like SQL injection, ``xss`` should
still parse as markup, etc. This catches malformed synthesis output — most
importantly truncated or unbalanced payloads a recursive grammar can emit — that
would otherwise poison the training signal for a class.

These are deliberately lightweight (regex + bracket-balance, no parser
dependencies). ``is_valid_for_class`` returns ``True`` for any class without a
registered validator, so it never rejects records it doesn't understand.
"""

from __future__ import annotations

import re
from collections.abc import Callable

_SQL_TOKEN = re.compile(
    r"(union\b|select\b|sleep\s*\(|benchmark\s*\(|waitfor\b|"
    r"\bor\b|\band\b|=|--|#|/\*)",
    re.IGNORECASE,
)
_XSS_VECTOR = re.compile(
    r"(<\s*[a-z]|on[a-z]+\s*=|javascript\s*:|data\s*:\s*text/html)",
    re.IGNORECASE,
)
_LFI_TOKEN = re.compile(
    r"(\.\./|\.\.\\|%2e%2e|%252e|/etc/|/proc/|php://|data://|\\windows\\)",
    re.IGNORECASE,
)
_SSRF_TOKEN = re.compile(
    r"(https?://|file://|dict://|gopher://|ftp://|169\.254\.169\.254|"
    r"metadata\.google\.internal|localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]|"
    r"192\.168\.|10\.0\.|\d{8,})",
    re.IGNORECASE,
)
_CMDI_TOKEN = re.compile(r"([;|&`]|\$\(|\bbash\b|\bnc\b|\n)")
_CRLF_TOKEN = re.compile(r"(%0d%0a|%0d|%0a|\r\n|\r|\n)", re.IGNORECASE)
_TEMPLATE_TOKEN = re.compile(r"(\{\{|\}\}|\$\{|#\{|<%|%>|\*\{|@\()")


def _balanced(text: str, opener: str, closer: str) -> bool:
    return text.count(opener) == text.count(closer)


def _valid_sqli(text: str) -> bool:
    # A SQL-ish token must survive, and parentheses must balance (recursion can
    # nest them; truncation leaves them unbalanced).
    return bool(_SQL_TOKEN.search(text)) and _balanced(text, "(", ")")


def _valid_xss(text: str) -> bool:
    # A markup vector must survive. Realistic breakouts (e.g. '">') are
    # deliberately bracket-imbalanced, so instead of a strict balance we reject
    # the truncation signature: a trailing, unclosed "<tag..." — i.e. the last
    # '<' comes after the last '>' (a dangling opening tag with no close).
    if not _XSS_VECTOR.search(text):
        return False
    return text.rfind("<") <= text.rfind(">")


def _valid_xxe(text: str) -> bool:
    low = text.lower()
    return "<!doctype" in low and "<!entity" in low


def _valid_ssti(text: str) -> bool:
    return bool(_TEMPLATE_TOKEN.search(text))


VALIDATORS: dict[str, Callable[[str], bool]] = {
    "sqli": _valid_sqli,
    "xss": _valid_xss,
    "lfi": lambda t: bool(_LFI_TOKEN.search(t)),
    "path_traversal": lambda t: bool(_LFI_TOKEN.search(t)),
    "ssrf": lambda t: bool(_SSRF_TOKEN.search(t)),
    "cmdi": lambda t: bool(_CMDI_TOKEN.search(t)),
    "header_injection": lambda t: bool(_CRLF_TOKEN.search(t)),
    "xxe": _valid_xxe,
    "ssti": _valid_ssti,
}


def is_valid_for_class(attack_class: str, text: str) -> bool:
    """
    Return whether ``text`` is structurally consistent with ``attack_class``.

    Classes without a registered validator (including ``benign``) always pass, so
    this check only ever rejects payloads it positively recognises as malformed.
    """
    validator = VALIDATORS.get(attack_class)
    if validator is None:
        return True
    return validator(text)
