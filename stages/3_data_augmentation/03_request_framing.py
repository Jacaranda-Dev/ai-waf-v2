"""
stages/3_data_augmentation/03_request_framing.py
-------------------------------------------------
Module C — Unified HTTP Request Framing Engine

Merges: 08_api_llm_framing, 09_benign_rest_traffic

Architecture:
  - Decouples payload generation from HTTP envelope construction
  - Single Jinja2-based template engine for ALL request types (attack + benign)
    → Ensures grammar-generated and LLM-generated payloads share the same
      header/path distribution, preventing the model from learning "synthetic" fingerprints
  - Consumes raw payload strings from Module A (01_attack_synthesis.py)
  - Reads benign_distribution.json written by Module B (02_benign_enrichment.py)
    to align HttpMetadataDistribution with PCAP-observed UA/auth/referer rates
  - Also generates benign traffic via programmatic REST patterns + cloud LLM
  - Cloud LLM used ONLY for HTTP framing context, never for raw attack payloads

Run:
    python stages/3_data_augmentation/03_request_framing.py \
        --config config/pipeline.yaml [--provider anthropic|google]
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid
from pathlib import Path

import pyarrow.parquet as pq
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, MofNCompleteColumn

from ai_waf_v2.data.schema import HttpRecord, records_to_table
from ai_waf_v2.utils.config import load_config
from ai_waf_v2.utils.llm import call_llm
from ai_waf_v2.utils.logging import configure_root, get_logger
from ai_waf_v2.utils.pipeline import require_inputs, check_output
from ai_waf_v2.utils.timing import StepTimer

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

    Long string fields (User-Agent, Referer, Accept, Accept-Language) are
    generated combinatorially rather than sampled from small hardcoded lists,
    producing hundreds of realistic variants per field.
    """

    # ── User-Agent components ────────────────────────────────────────────────

    _UA_WINDOWS  = ["Windows NT 10.0; Win64; x64", "Windows NT 10.0; WOW64",
                    "Windows NT 6.1; Win64; x64", "Windows NT 6.3; Win64; x64"]
    _UA_MAC      = ["Macintosh; Intel Mac OS X 10_15_7", "Macintosh; Intel Mac OS X 13_6",
                    "Macintosh; Intel Mac OS X 14_4", "Macintosh; Intel Mac OS X 12_7"]
    _UA_LINUX    = ["X11; Linux x86_64", "X11; Linux i686",
                    "X11; Ubuntu; Linux x86_64", "X11; Linux aarch64"]
    _UA_ANDROID  = ["Linux; Android 13; SM-G991B", "Linux; Android 14; Pixel 8",
                    "Linux; Android 13; Pixel 7", "Linux; Android 12; SM-A525F"]
    _UA_IOS      = ["iPhone; CPU iPhone OS 16_6 like Mac OS X",
                    "iPhone; CPU iPhone OS 17_4 like Mac OS X",
                    "iPad; CPU OS 16_6 like Mac OS X"]

    _CHROME_VERSIONS = ["110.0.5481.177", "114.0.5735.199", "118.0.5993.88",
                        "120.0.6099.130", "122.0.6261.128", "124.0.6367.82",
                        "125.0.6422.60"]
    _FIREFOX_VERSIONS = ["109.0", "115.0", "118.0", "121.0", "124.0", "125.0"]
    _SAFARI_WEBKIT    = ["605.1.15"]
    _SAFARI_VERSIONS  = ["16.6", "17.0", "17.2", "17.4"]

    _TOOL_AGENTS = [
        "python-requests/2.28.2", "python-requests/2.31.0",
        "curl/7.88.1", "curl/8.4.0",
        "axios/1.6.8", "axios/1.4.0",
        "okhttp/4.12.0", "okhttp/4.9.3",
        "Go-http-client/1.1", "Go-http-client/2.0",
        "PostmanRuntime/7.37.0", "PostmanRuntime/7.32.3",
        "Wget/1.21.4", "HTTPie/3.2.2",
        "Java/11.0.20", "Ruby/3.2.0",
    ]

    # ── Referer components ───────────────────────────────────────────────────

    _REFERER_SEARCH = [
        "https://www.google.com/search?q={q}",
        "https://www.bing.com/search?q={q}",
        "https://duckduckgo.com/?q={q}",
        "https://search.yahoo.com/search?p={q}",
    ]
    _REFERER_APP = [
        "https://app.example.com/dashboard",
        "https://app.example.com/search",
        "https://app.example.com/profile",
        "https://app.example.com/settings",
        "https://portal.example.com/home",
        "https://admin.example.com/users",
        "https://shop.example.com/cart",
    ]
    _REFERER_SEARCH_TERMS = [
        "product+review", "how+to", "buy+online", "best+price",
        "login+page", "api+docs", "support+ticket",
    ]

    # ── Accept components ────────────────────────────────────────────────────

    _ACCEPT_BROWSER  = [
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "text/html, application/xhtml+xml, application/xml;q=0.9, */*;q=0.8",
    ]
    _ACCEPT_API = [
        "application/json",
        "application/json, text/plain, */*",
        "application/json;charset=UTF-8",
        "application/json, text/javascript, */*; q=0.01",
    ]
    _ACCEPT_ANY = ["*/*", "text/plain, */*"]

    # ── Accept-Language components ────────────────────────────────────────────

    _LANG_PRIMARY   = ["en-US", "en-GB", "fr-FR", "de-DE", "es-ES", "pt-BR",
                       "it-IT", "ja-JP", "zh-CN", "ko-KR", "nl-NL", "ru-RU"]
    _LANG_SECONDARY = ["en", "fr", "de", "es", "pt", "it", "ja", "zh", "ko"]
    _LANG_Q_VALUES  = ["0.9", "0.8", "0.7", "0.5"]

    # ── Fixed small sets ─────────────────────────────────────────────────────

    HOST_VARIANTS = [
        "api.example.com", "app.example.com", "www.example.com",
        "staging.example.com", "example.com", "dev.example.com",
        "api.example.io", "backend.example.com",
    ]

    _AUTH_NONEMPTY = [
        lambda r: f"Bearer eyJhbGciOiJIUzI1NiJ9.{uuid.uuid4().hex[:24]}.sig",
        lambda r: f"Token {uuid.uuid4().hex}",
        lambda r: f"Basic {uuid.uuid4().hex[:16]}",
        lambda r: f"Bearer {uuid.uuid4().hex}",
    ]

    CONTENT_TYPES = [
        "application/x-www-form-urlencoded",
        "application/json",
        "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        "text/plain",
    ]

    # ── Instance state (overridable via configure()) ──────────────────────────

    def __init__(self) -> None:
        self._pcap_uas:    list[str]   = []
        self._pcap_ua_wts: list[float] = []
        self._auth_prob:    float      = 0.8   # fraction of requests with auth
        self._referer_prob: float      = 0.7   # fraction of requests with Referer

    def configure(self, profile: dict) -> None:
        """Apply PCAP-fitted distributions, overriding internal defaults."""
        uas = profile.get("user_agents", {})
        if uas:
            self._pcap_uas    = list(uas.keys())
            self._pcap_ua_wts = list(uas.values())
        if (v := profile.get("has_auth")) is not None:
            self._auth_prob = float(v)
        if (v := profile.get("has_referer")) is not None:
            self._referer_prob = float(v)

    # ── Generators ───────────────────────────────────────────────────────────

    def _sample_user_agent(self, rng: random.Random) -> str:
        if self._pcap_uas:
            return rng.choices(self._pcap_uas, weights=self._pcap_ua_wts, k=1)[0]
        kind = rng.choices(["chrome", "firefox", "safari", "mobile_chrome", "tool"],
                           weights=[35, 20, 15, 20, 10])[0]
        cv = rng.choice(self._CHROME_VERSIONS)
        fv = rng.choice(self._FIREFOX_VERSIONS)
        sv = rng.choice(self._SAFARI_VERSIONS)
        if kind == "chrome":
            os_ = rng.choice(self._UA_WINDOWS + self._UA_MAC + self._UA_LINUX)
            return (f"Mozilla/5.0 ({os_}) AppleWebKit/537.36 "
                    f"(KHTML, like Gecko) Chrome/{cv} Safari/537.36")
        if kind == "firefox":
            os_ = rng.choice(self._UA_WINDOWS + self._UA_MAC + self._UA_LINUX)
            return (f"Mozilla/5.0 ({os_}; rv:{fv}) Gecko/20100101 Firefox/{fv}")
        if kind == "safari":
            os_ = rng.choice(self._UA_MAC)
            wv = self._SAFARI_WEBKIT[0]
            return (f"Mozilla/5.0 ({os_}) AppleWebKit/{wv} "
                    f"(KHTML, like Gecko) Version/{sv} Safari/{wv}")
        if kind == "mobile_chrome":
            if rng.random() < 0.6:
                os_ = rng.choice(self._UA_ANDROID)
                return (f"Mozilla/5.0 ({os_}) AppleWebKit/537.36 "
                        f"(KHTML, like Gecko) Chrome/{cv} Mobile Safari/537.36")
            os_ = rng.choice(self._UA_IOS)
            wv = self._SAFARI_WEBKIT[0]
            return (f"Mozilla/5.0 ({os_}) AppleWebKit/{wv} "
                    f"(KHTML, like Gecko) Version/{sv} Mobile/15E148 Safari/604.1")
        return rng.choice(self._TOOL_AGENTS)

    def _sample_referer(self, rng: random.Random) -> str:
        if rng.random() < 0.36:
            tmpl = rng.choice(self._REFERER_SEARCH)
            term = rng.choice(self._REFERER_SEARCH_TERMS)
            return tmpl.format(q=term)
        return rng.choice(self._REFERER_APP)

    def _sample_accept(self, rng: random.Random, has_body: bool) -> str:
        if has_body:
            return rng.choice(self._ACCEPT_API + self._ACCEPT_ANY)
        return rng.choice(self._ACCEPT_BROWSER + self._ACCEPT_API)

    def _sample_language(self, rng: random.Random) -> str:
        primary = rng.choice(self._LANG_PRIMARY)
        lang    = primary.split("-")[0]
        if lang in self._LANG_SECONDARY and rng.random() < 0.7:
            q = rng.choice(self._LANG_Q_VALUES)
            return f"{primary},{lang};q={q}"
        return primary

    def sample_headers(
        self,
        rng:      random.Random,
        method:   str,
        has_body: bool = False,
        extra:    dict | None = None,
    ) -> dict:
        h = {
            "Host":            rng.choice(self.HOST_VARIANTS),
            "User-Agent":      self._sample_user_agent(rng),
            "Accept":          self._sample_accept(rng, has_body),
            "Accept-Language": self._sample_language(rng),
            "Connection":      rng.choice(["keep-alive", "close"]),
        }

        if rng.random() < self._referer_prob:
            h["Referer"] = self._sample_referer(rng)

        if rng.random() < self._auth_prob:
            h["Authorization"] = rng.choice(self._AUTH_NONEMPTY)(rng)

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
    # `endpoints` intentionally absent — paths are sampled from _ENDPOINT_REGISTRY
    # (shared with benign) so the model cannot fingerprint on URL alone.
    # Only injection mechanics live here: params, methods, header_inject, injection_sites.
    "sqli": dict(
        params=["id", "q", "search", "user_id", "filter", "sort", "order", "category", "page", "token"],
        methods=["GET", "POST"],
        header_inject=["Cookie", "X-Forwarded-For"],
        path_templates=["/api/v1/users/{id}", "/api/v1/products/{id}",
                        "/api/v1/orders/{id}", "/api/v1/files/{id}"],
        injection_sites=[("query_param", 35), ("post_body", 20), ("path_segment", 20),
                         ("cookie",      15), ("json_nested", 10)],
    ),
    "xss": dict(
        params=["comment", "name", "message", "q", "text", "body", "title",
                "description", "subject", "content", "username", "email"],
        methods=["GET", "POST"],
        header_inject=["Referer", "User-Agent"],
        path_templates=["/api/v1/users/{id}", "/api/v1/messages/{id}", "/api/v1/comments/{id}"],
        injection_sites=[("query_param", 35), ("post_body", 25), ("cookie",      15),
                         ("json_nested", 15), ("path_segment", 10)],
    ),
    "lfi": dict(
        params=["file", "page", "path", "doc", "include", "template", "view", "load"],
        methods=["GET"],
        header_inject=[],
        path_templates=["/api/v1/files/{id}", "/api/v1/reports/{id}", "/api/v1/messages/{id}"],
        injection_sites=[("query_param", 45), ("path_segment", 35), ("multipart", 20)],
    ),
    "ssrf": dict(
        params=["url", "uri", "dest", "redirect", "callback", "webhook",
                "proxy", "next", "target", "fetch", "request"],
        methods=["GET", "POST"],
        header_inject=[],
        path_templates=[],
        injection_sites=[("query_param", 45), ("post_body",   20),
                         ("json_nested", 25), ("host_header", 10)],
    ),
    "cmdi": dict(
        params=["cmd", "host", "ping", "exec", "run", "input", "query", "ip", "shell"],
        methods=["GET", "POST"],
        header_inject=[],
        path_templates=[],
        injection_sites=[("query_param", 40), ("post_body",   30),
                         ("json_nested", 20), ("cookie",      10)],
    ),
    "path_traversal": dict(
        params=["path", "file", "dir", "filename", "filepath", "document", "resource"],
        methods=["GET"],
        header_inject=[],
        path_templates=["/api/v1/files/{id}", "/api/v1/reports/{id}", "/api/v1/messages/{id}"],
        injection_sites=[("query_param", 40), ("path_segment", 40), ("multipart", 20)],
    ),
    "header_injection": dict(
        params=["redirect", "url", "next", "location", "return", "returnUrl", "callback"],
        methods=["GET"],
        header_inject=["Location", "X-Forwarded-Host"],
        path_templates=[],
        injection_sites=[("query_param", 55), ("header", 30), ("host_header", 15)],
    ),
    "xxe": dict(
        params=[],
        methods=["POST"],
        header_inject=[],
        path_templates=[],
        injection_sites=[("raw_body", 100)],
    ),
    "ssti": dict(
        params=["name", "template", "subject", "greeting", "message", "title", "content"],
        methods=["GET", "POST"],
        header_inject=[],
        path_templates=["/api/v1/messages/{id}", "/api/v1/notifications/{id}", "/api/v1/users/{id}"],
        injection_sites=[("query_param", 40), ("post_body",    20),
                         ("path_segment", 30), ("json_nested", 10)],
    ),
    # Benign records are routed through the same pipeline as attacks.
    # The "payload" is an innocent value; the request structure is identical.
    "benign": dict(
        params=["q", "search", "name", "email", "text", "subject", "page",
                "limit", "sort", "category", "status", "type", "format"],
        methods=["GET", "POST", "PUT", "PATCH"],
        header_inject=[],
        path_templates=["/api/v1/users/{id}", "/api/v1/products/{id}",
                        "/api/v1/orders/{id}", "/api/v1/messages/{id}"],
        injection_sites=[("query_param", 45), ("post_body",    30),
                         ("json_nested", 15), ("path_segment", 10)],
    ),
}


# ── Shared param value generators ─────────────────────────────────────────────
# Lambdas reference module-level lists defined later; lookup is at call-time.

_PARAM_VALUE_MAP = {
    # Pagination / sorting
    "page":          lambda r: str(r.randint(1, 50)),
    "limit":         lambda r: r.choice(["10", "20", "25", "50", "100"]),
    "per_page":      lambda r: r.choice(["10", "20", "25", "50"]),
    "offset":        lambda r: str(r.randint(0, 500)),
    "cursor":        lambda r: r.choice(["eyJpZCI6MTAwfQ==", "eyJpZCI6MjAwfQ==",
                                         "eyJpZCI6MzAwfQ==", "eyJza2lwIjo1MH0="]),
    "sort":          lambda r: r.choice(_SORT_FIELDS),
    "order":         lambda r: r.choice(["asc", "desc"]),
    "order_by":      lambda r: r.choice(_SORT_FIELDS),
    "direction":     lambda r: r.choice(["asc", "desc"]),
    # Filtering
    "status":        lambda r: r.choice(_STATUS_VALUES),
    "role":          lambda r: r.choice(["user", "admin", "moderator", "viewer", "editor", "guest"]),
    "category":      lambda r: r.choice(_CATEGORIES),
    "type":          lambda r: r.choice(["pdf", "image", "video", "document", "spreadsheet", "archive"]),
    "tag":           lambda r: r.choice(["featured", "sale", "new", "popular", "limited", "trending"]),
    "label":         lambda r: r.choice(["urgent", "low-priority", "review", "approved", "rejected"]),
    "priority":      lambda r: r.choice(["low", "medium", "high", "critical"]),
    "in_stock":      lambda r: r.choice(["true", "false", "1", "0"]),
    "active":        lambda r: r.choice(["true", "false", "1", "0"]),
    "filter":        lambda r: r.choice(["all", "mine", "team", "recent", "starred"]),
    "include":       lambda r: r.choice(["tags", "meta", "author", "related", "comments"]),
    "expand":        lambda r: r.choice(["user", "product", "order", "category", "tags"]),
    "fields":        lambda r: r.choice(["id,name,status", "id,email,role", "id,name,price,stock",
                                         "id,created_at,updated_at", "name,description,category"]),
    # Search
    "q":             lambda r: r.choice(_BENIGN_QUERIES).replace("+", " "),
    "query":         lambda r: r.choice(_BENIGN_QUERIES).replace("+", " "),
    "search":        lambda r: r.choice(_BENIGN_QUERIES).replace("+", " "),
    "keyword":       lambda r: r.choice(_BENIGN_QUERIES).replace("+", " "),
    "term":          lambda r: r.choice(_BENIGN_QUERIES).replace("+", " "),
    # Dates
    "from":          lambda r: r.choice(_DATE_RANGE_PAIRS)[0],
    "start_date":    lambda r: r.choice(_DATE_RANGE_PAIRS)[0],
    "to":            lambda r: r.choice(_DATE_RANGE_PAIRS)[1],
    "end_date":      lambda r: r.choice(_DATE_RANGE_PAIRS)[1],
    "date":          lambda r: r.choice(_DATE_RANGE_PAIRS)[0],
    "created_after": lambda r: r.choice(_DATE_RANGE_PAIRS)[0],
    "created_before":lambda r: r.choice(_DATE_RANGE_PAIRS)[1],
    # Pricing / financial
    "min_price":     lambda r: str(r.randint(5, 200)),
    "max_price":     lambda r: str(r.randint(200, 2000)),
    "price":         lambda r: str(round(r.uniform(5.0, 999.0), 2)),
    "amount":        lambda r: str(round(r.uniform(1.0, 500.0), 2)),
    "currency":      lambda r: r.choice(["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]),
    "plan":          lambda r: r.choice(["free", "starter", "pro", "enterprise", "basic", "premium"]),
    "billing_cycle": lambda r: r.choice(["monthly", "yearly", "quarterly"]),
    # Identifiers
    "product_id":    lambda r: str(r.randint(1, 9999)),
    "parent_id":     lambda r: str(r.randint(1, 999)),
    "ids":           lambda r: ",".join(str(r.randint(1, 99)) for _ in range(r.randint(1, 5))),
    "slug":          lambda r: r.choice(["best-laptop-2024", "summer-sale-items",
                                         "new-arrivals-spring", "featured-products",
                                         "top-rated-electronics", "weekly-deals"]),
    # User / auth fields
    "email":         lambda r: (
                         f"{r.choice(['alice', 'bob', 'carol', 'dave', 'eve', 'frank', 'grace', 'hank'])}"
                         f"{r.randint(1, 999)}"
                         f"@{r.choice(_EMAIL_DOMAINS)}"
                     ),
    "username":      lambda r: f"{r.choice(_FIRST_NAMES).lower()}{r.randint(10, 99)}",
    "password":      lambda r: r.choice(["S3cur3P@ss!", "P@ssw0rd!", "Tr0ub4d0r&", "C0rrectHorse!"]),
    "new_password":  lambda r: r.choice(["N3wP@ss!", "Upd@ted123!", "Ch@nged456!"]),
    "token":         lambda r: uuid.uuid4().hex,
    "refresh_token": lambda r: uuid.uuid4().hex,
    "api_key":       lambda r: f"sk_{uuid.uuid4().hex[:32]}",
    # Person / profile
    "name":          lambda r: r.choice([
                         f"{r.choice(_FIRST_NAMES)} {r.choice(_LAST_NAMES)}",
                         r.choice(_FIRST_NAMES),
                         r.choice(_COMPANIES),
                     ]),
    "first_name":    lambda r: r.choice(_FIRST_NAMES),
    "last_name":     lambda r: r.choice(_LAST_NAMES),
    "bio":           lambda r: r.choice([
                         f"{r.choice(_OCCUPATIONS)} based in {r.choice(_CITIES)}.",
                         f"Passionate about {r.choice(['open source', 'design', 'data', 'products'])}.",
                         f"Working at {r.choice(_COMPANIES)}.",
                     ]),
    "public":        lambda r: r.choice(["true", "false", "1", "0"]),
    # Content fields
    "title":         lambda r: r.choice([
                         "Great product, exactly what I needed",
                         "Fast shipping and excellent quality",
                         "Could be better for the price",
                         "Absolutely love this — five stars",
                         "Decent but not exceptional",
                         "Minor issues, overall satisfied",
                         "Perfect gift for the holidays",
                         "Would recommend to a friend",
                     ]),
    "text":          lambda r: r.choice([
                         "Great product, highly recommend to anyone looking for quality.",
                         "Works exactly as described. Fast shipping too.",
                         "Could be better quality for the price, but it does the job.",
                         "Five stars — absolutely love it. Will buy again.",
                         "Decent but not exceptional. Does what it says.",
                         "Minor issues with the packaging but the product itself is great.",
                         "Perfect gift. Arrived on time and well packaged.",
                         "Satisfied with my purchase. Customer service was helpful.",
                     ]),
    "body":          lambda r: r.choice([
                         "Hi, I have a question about my recent order.",
                         "Hello, could you help me track my shipment?",
                         "I would like to update my billing information.",
                         "Please cancel my subscription effective immediately.",
                         "I am having trouble logging in to my account.",
                     ]),
    "subject":       lambda r: r.choice(["Hello", "Question about my order", "Order inquiry",
                                         "Re: support ticket", "Follow-up", "Account help",
                                         "Billing question", "Shipping update"]),
    "description":   lambda r: r.choice([
                         "Excellent build quality and well-documented features.",
                         "Lightweight and durable, perfect for everyday use.",
                         "Compact design with all the features you need.",
                         "High performance at a competitive price point.",
                         "Easy to set up and works out of the box.",
                     ]),
    "message":       lambda r: r.choice([
                         "Your request has been processed successfully.",
                         "Please review the attached document.",
                         "Meeting scheduled for next week.",
                         "Action required: confirm your email address.",
                     ]),
    "reason":        lambda r: r.choice(["duplicate", "spam", "inappropriate", "other",
                                         "changed-mind", "found-better-price", "not-as-described"]),
    "action":        lambda r: r.choice(["approve", "reject", "archive", "restore",
                                         "publish", "unpublish", "export"]),
    # Cart / order
    "qty":           lambda r: str(r.randint(1, 10)),
    "quantity":      lambda r: str(r.randint(1, 10)),
    "variant":       lambda r: r.choice(["red-M", "blue-L", "black-XL", "white-S",
                                         "grey-M", "navy-L", "green-S", "yellow-XL"]),
    "coupon":        lambda r: r.choice(["SAVE10", "SUMMER20", "WELCOME15", "LOYALTY5",
                                         "FLASH30", "VIP25", ""]),
    "stock":         lambda r: str(r.randint(0, 500)),
    # Address fields
    "street":        lambda r: r.choice(["123 Main St", "456 Oak Ave", "789 Pine Rd",
                                         "321 Elm Blvd", "654 Maple Dr", "987 Cedar Ln"]),
    "city":          lambda r: r.choice(_CITIES),
    "country":       lambda r: r.choice(["US", "GB", "CA", "AU", "DE", "FR", "JP", "BR"]),
    "zip":           lambda r: r.choice(["10001", "90210", "60601", "77001", "85001",
                                         "EC1A 1BB", "M1 1AE", "W1A 0AX"]),
    # Files
    "filename":      lambda r: r.choice(["report-2024-q1.pdf", "invoice_001.pdf",
                                         "photo.jpg", "data_export.csv",
                                         "backup_2024.zip", "presentation.pptx"]),
    "path":          lambda r: r.choice(["/docs/readme.md", "/reports/q1-summary.pdf",
                                         "/images/logo.png", "/exports/data.csv",
                                         "/templates/invoice.html"]),
    "content_type":  lambda r: r.choice(["application/json", "text/plain",
                                         "application/pdf", "image/png", "text/csv"]),
    "size":          lambda r: str(r.randint(1024, 10485760)),
    # Misc
    "rating":        lambda r: str(r.randint(1, 5)),
    "url":           lambda r: r.choice([
                         f"https://hooks.example.com/notify/{uuid.uuid4().hex[:8]}",
                         f"https://webhook.example.org/event/{uuid.uuid4().hex[:6]}",
                         f"https://app.example.io/callback/{uuid.uuid4().hex[:8]}",
                     ]),
    "events":        lambda r: r.choice([
                         '["order.created","user.updated"]',
                         '["payment.success","order.shipped"]',
                         '["user.signup","user.deleted"]',
                         '["product.updated","stock.low"]',
                     ]),
    "secret":        lambda r: uuid.uuid4().hex[:16],
    "color":         lambda r: r.choice(["#ff0000", "#00cc44", "#0066ff", "#ffaa00",
                                         "#9900cc", "#ff6699", "#333333", "#e8e8e8"]),
    "notifications": lambda r: r.choice(["true", "false"]),
    "language":      lambda r: r.choice(["en", "fr", "de", "es", "pt", "it", "ja", "zh"]),
    "theme":         lambda r: r.choice(["light", "dark", "system", "high-contrast"]),
    "timezone":      lambda r: r.choice(["UTC", "America/New_York", "America/Los_Angeles",
                                         "Europe/London", "Europe/Paris", "Asia/Tokyo"]),
    "lang":          lambda r: r.choice(["en", "fr", "de", "es", "ja", "pt", "zh"]),
    "locale":        lambda r: r.choice(["en-US", "en-GB", "fr-FR", "de-DE",
                                         "es-ES", "pt-BR", "ja-JP", "zh-CN"]),
    "level":         lambda r: r.choice(["info", "warn", "error", "debug", "trace"]),
    "read":          lambda r: r.choice(["true", "false"]),
    "format":        lambda r: r.choice(["json", "csv", "xlsx", "pdf", "xml"]),
    "event_type":    lambda r: r.choice(["click", "purchase", "signup", "view", "search",
                                         "download", "share", "login", "logout"]),
    "ids":           lambda r: ",".join(str(r.randint(1, 99)) for _ in range(r.randint(1, 5))),
    "v":             lambda r: r.choice(["1", "2", "3"]),
    "ts":            lambda r: str(r.randint(1700000000, 1800000000)),
    "key":           lambda r: r.choice(["feature_x", "beta_ui", "dark_mode",
                                         "max_results", "timeout_ms", "retry_limit"]),
    "value":         lambda r: r.choice(["true", "false", "42", "enabled", "v2"]),
    "version":       lambda r: r.choice(["1.0.0", "2.1.3", "3.0.0-beta", "1.5.2"]),
    "duration":      lambda r: r.choice(["7d", "30d", "90d", "1y", "3600", "86400"]),
}


def _innocent_value(param: str, rng: random.Random) -> str:
    fn = _PARAM_VALUE_MAP.get(param)
    if fn:
        return fn(rng)
    # Generic fallback: sample a plausible HTTP param value across common types
    kind = rng.choice(["int", "float", "bool", "uuid", "slug", "date", "short_id", "empty"])
    if kind == "int":       return str(rng.randint(0, 99999))
    if kind == "float":     return str(round(rng.uniform(0.1, 9999.9), 2))
    if kind == "bool":      return rng.choice(["true", "false", "1", "0"])
    if kind == "uuid":      return str(uuid.uuid4())
    if kind == "slug":      return rng.choice(["my-item", "top-picks", "new-release",
                                               "best-value", "staff-pick", "limited-edition"])
    if kind == "date":      return rng.choice(_DATE_RANGE_PAIRS)[0]
    if kind == "short_id":  return uuid.uuid4().hex[:8]
    return ""  # empty


def _fill_params(
    params: list[str],
    rng: random.Random,
    inject_param: str | None = None,
    inject_value: str | None = None,
) -> dict[str, str]:
    """Fill param→value dict. All values are innocent except inject_param (if given)."""
    n        = rng.randint(max(1, len(params) // 2), len(params)) if params else 0
    selected = rng.sample(params, k=n) if 0 < n <= len(params) else list(params)
    result   = {p: _innocent_value(p, rng) for p in selected}
    if inject_param is not None:
        result[inject_param] = inject_value or ""
    return result


def _params_to_qs(params_dict: dict[str, str], rng: random.Random) -> str:
    items = list(params_dict.items())
    rng.shuffle(items)
    return "&".join(f"{k}={v}" for k, v in items)


# ── Shared endpoint sampler ───────────────────────────────────────────────────

def _sample_endpoint(methods: list[str], rng: random.Random) -> tuple[str, str, list[str]]:
    """Sample (method, path, params) from _ENDPOINT_REGISTRY filtered by allowed methods.

    Both attack and benign framing draw from the same pool so the model cannot
    distinguish label from URL or request structure alone.
    """
    pool = [(m, p, ps) for m, p, ps in _ENDPOINT_REGISTRY if m in methods]
    if not pool:
        pool = list(_ENDPOINT_REGISTRY)
    method, path_tmpl, params = rng.choice(pool)
    return method, path_tmpl.replace("{id}", str(rng.randint(1, 9999))), params


def _pick_inject_param(attack_params: list[str], endpoint_params: list[str],
                       rng: random.Random) -> str:
    """Choose injection param: prefer overlap with the endpoint's natural params."""
    if not attack_params:
        return rng.choice(endpoint_params) if endpoint_params else "q"
    overlap = [p for p in attack_params if p in endpoint_params]
    return rng.choice(overlap) if overlap else rng.choice(attack_params)


# ── Per-site injection helpers ────────────────────────────────────────────────
# Each returns a dict: {method, path, qs, body, extra}
# payload is the value placed in the injection slot (malicious for attacks,
# innocent for benign — callers are identical either way).

def _frame_query_param(payload: str, meta: dict, rng: random.Random) -> dict:
    method, path, params = _sample_endpoint(meta["methods"], rng)
    param  = _pick_inject_param(meta["params"], params, rng)
    filled = _fill_params(params, rng, param, payload)
    return dict(method=method, path=path, qs=_params_to_qs(filled, rng), body="", extra={})


def _frame_post_body(payload: str, meta: dict, rng: random.Random) -> dict:
    _, path, params = _sample_endpoint(["POST"], rng)
    param  = _pick_inject_param(meta["params"], params, rng)
    filled = _fill_params(params, rng, param, payload)
    return dict(method="POST", path=path, qs="",
                body=json.dumps(filled), extra={"Content-Type": "application/json"})


def _frame_path_segment(payload: str, meta: dict, rng: random.Random) -> dict:
    tmpl               = rng.choice(meta.get("path_templates") or ["/api/v1/items/{id}"])
    _, _, params       = _sample_endpoint(["GET"], rng)
    filled             = _fill_params(params, rng)
    return dict(method="GET", path=tmpl.replace("{id}", payload),
                qs=_params_to_qs(filled, rng), body="", extra={})


def _frame_cookie(payload: str, meta: dict, rng: random.Random) -> dict:
    method, path, params = _sample_endpoint(meta["methods"], rng)
    param    = _pick_inject_param(meta["params"], params, rng)
    innocent = [
        f"session={uuid.uuid4().hex[:16]}",
        f"csrf={uuid.uuid4().hex[:12]}",
        f"_ga=GA1.1.{rng.randint(100000000, 999999999)}.1700000000",
    ]
    cookie_parts = innocent + [f"{param}={payload}"]
    rng.shuffle(cookie_parts)
    filled = _fill_params(params, rng)
    return dict(method=method, path=path,
                qs=_params_to_qs(filled, rng) if method == "GET" else "",
                body="" if method == "GET" else json.dumps(filled),
                extra={"Cookie": "; ".join(cookie_parts)})


def _frame_json_nested(payload: str, meta: dict, rng: random.Random) -> dict:
    _, path, params = _sample_endpoint(["POST"], rng)
    param  = _pick_inject_param(meta["params"], params, rng)
    filled = _fill_params(params, rng)
    wrappers = [
        {param: payload, **filled},
        {"data":    {param: payload, **filled}},
        {"request": {param: payload, "limit": rng.randint(10, 100)}},
        {"filter":  {param: payload, "active": True}},
        {"params":  {param: payload, **filled}},
    ]
    return dict(method="POST", path=path, qs="",
                body=json.dumps(rng.choice(wrappers)),
                extra={"Content-Type": "application/json"})


def _frame_multipart(payload: str, meta: dict, rng: random.Random) -> dict:
    _, path, _ = _sample_endpoint(["POST"], rng)
    boundary   = f"----WebKitFormBoundary{uuid.uuid4().hex[:16]}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{payload}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n\r\n"
        f"--{boundary}--"
    )
    return dict(method="POST", path=path, qs="", body=body,
                extra={"Content-Type": f"multipart/form-data; boundary={boundary}"})


def _frame_host_header(payload: str, meta: dict, rng: random.Random) -> dict:
    method, path, params = _sample_endpoint(meta["methods"], rng)
    filled = _fill_params(params, rng)
    qs     = _params_to_qs(filled, rng) if method == "GET" else ""
    body   = json.dumps(filled) if method in ("POST", "PUT", "PATCH") else ""
    return dict(method=method, path=path, qs=qs, body=body, extra={"Host": payload})


def _frame_header(payload: str, meta: dict, rng: random.Random) -> dict:
    targets          = meta.get("header_inject") or []
    header           = rng.choice(targets) if targets else "X-Custom-Header"
    _, path, params  = _sample_endpoint(["GET"], rng)
    filled           = _fill_params(params, rng)
    return dict(method="GET", path=path, qs=_params_to_qs(filled, rng),
                body="", extra={header: payload})


def _frame_raw_body(payload: str, meta: dict, rng: random.Random) -> dict:
    _, path, _ = _sample_endpoint(["POST"], rng)
    return dict(method="POST", path=path, qs="", body=payload,
                extra={"Content-Type": "application/xml"})


_SITE_DISPATCH = {
    "query_param":  _frame_query_param,
    "post_body":    _frame_post_body,
    "path_segment": _frame_path_segment,
    "cookie":       _frame_cookie,
    "json_nested":  _frame_json_nested,
    "multipart":    _frame_multipart,
    "host_header":  _frame_host_header,
    "header":       _frame_header,
    "raw_body":     _frame_raw_body,
}


def frame_record(
    payload:      str,
    attack_class: str,
    label:        int,
    source:       str,
    rng:          random.Random,
) -> HttpRecord | None:
    """Frame a payload string into a realistic HTTP request.

    Works for both attack (label=1) and benign (label=0) records — the only
    difference is the value placed in the injection slot.  Everything else
    (endpoint, params, headers, body format) is generated identically.
    """
    meta = _ATTACK_META.get(attack_class, _ATTACK_META["benign"])
    sites, weights = zip(*meta["injection_sites"])
    site           = rng.choices(sites, weights=weights)[0]
    result         = _SITE_DISPATCH[site](payload, meta, rng)
    headers        = DIST.sample_headers(
        rng, result["method"], has_body=bool(result["body"]), extra=result["extra"]
    )
    try:
        return HttpRecord(
            id=str(uuid.uuid4()),
            method=result["method"],
            path=result["path"],
            query_string=result["qs"],
            headers=json.dumps(headers),
            body=result["body"],
            label=label,
            attack_class=attack_class,
            source=source,
        ).build_raw()
    except Exception as e:
        log.debug(f"Framing failed ({attack_class}): {e}")
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
        r = frame_record(payload, attack_class, label=1, source=row.get("source", "aug_reframed"), rng=rng)
        if r:
            records_out.append(r)

    log.info(f"Reframed {len(records_out):,} attack records with shared metadata distribution")
    return records_out


# ─────────────────────────────────────────────────────────────────────────────
# Benign request generator  (programmatic REST patterns)
# ─────────────────────────────────────────────────────────────────────────────

# ── Unified endpoint registry ─────────────────────────────────────────────────
# Single source of truth for paths used by BOTH attack and benign framing.
# Tuple: (method, path_template, natural_params)
# natural_params drives _fill_params — the innocent context around any payload.
_ENDPOINT_REGISTRY: list[tuple[str, str, list[str]]] = [
    # Auth & session
    ("POST",   "/api/v1/auth/login",           ["email", "password"]),
    ("POST",   "/api/v1/auth/logout",          []),
    ("POST",   "/api/v1/auth/refresh",         ["refresh_token"]),
    ("POST",   "/api/v1/auth/register",        ["name", "email", "password", "role"]),
    ("POST",   "/api/v1/auth/password/reset",  ["email"]),
    ("POST",   "/api/v1/auth/password/change", ["token", "new_password"]),
    ("POST",   "/api/v1/auth/verify-email",    ["token"]),
    # Profile
    ("GET",    "/api/v1/profile",              []),
    ("PATCH",  "/api/v1/profile",              ["name", "bio", "public", "email"]),
    ("GET",    "/api/v1/profile/preferences",  ["lang", "theme", "timezone"]),
    # Users
    ("GET",    "/api/v1/users",                ["page", "limit", "sort", "order", "status", "role", "q"]),
    ("GET",    "/api/v1/users/{id}",           []),
    ("PUT",    "/api/v1/users/{id}",           ["name", "email", "role", "bio", "status"]),
    ("PATCH",  "/api/v1/users/{id}",           ["name", "email", "role"]),
    ("DELETE", "/api/v1/users/{id}",           []),
    ("GET",    "/api/v1/users/{id}/activity",  ["page", "limit", "from", "to"]),
    # Products / catalog
    ("GET",    "/api/v1/products",             ["page", "limit", "sort", "order", "category",
                                                "q", "min_price", "max_price", "in_stock"]),
    ("GET",    "/api/v1/products/{id}",        []),
    ("POST",   "/api/v1/products",             ["name", "price", "category", "description", "stock"]),
    ("PATCH",  "/api/v1/products/{id}",        ["name", "price", "stock", "category"]),
    ("DELETE", "/api/v1/products/{id}",        []),
    ("GET",    "/api/v1/categories",           []),
    ("GET",    "/api/v1/categories/{id}",      []),
    ("GET",    "/api/v1/categories/{id}/products", ["page", "limit", "sort"]),
    ("GET",    "/api/v2/catalog/items",        ["page", "limit", "q", "category", "sort"]),
    ("GET",    "/api/v2/catalog/items/{id}",   []),
    # Search
    ("GET",    "/api/v1/search",               ["q", "page", "limit", "sort", "category", "type"]),
    ("GET",    "/search",                      ["q", "page", "limit"]),
    ("GET",    "/api/v2/search/products",      ["q", "page", "limit", "category"]),
    ("GET",    "/api/v2/search/users",         ["q", "page", "limit", "role"]),
    # Orders / cart
    ("POST",   "/api/v1/orders",              ["product_id", "qty", "coupon"]),
    ("GET",    "/api/v1/orders",              ["page", "limit", "status", "sort", "from", "to"]),
    ("GET",    "/api/v1/orders/{id}",         []),
    ("PATCH",  "/api/v1/orders/{id}",         ["status"]),
    ("DELETE", "/api/v1/orders/{id}",         []),
    ("POST",   "/api/v1/cart/items",          ["product_id", "qty", "variant"]),
    ("GET",    "/api/v1/cart",                []),
    ("PATCH",  "/api/v1/cart/items/{id}",     ["qty"]),
    ("DELETE", "/api/v1/cart/items/{id}",     []),
    # Reviews & comments
    ("GET",    "/api/v1/reviews",             ["page", "limit", "sort", "rating"]),
    ("POST",   "/api/v1/reviews",             ["text", "rating", "title"]),
    ("GET",    "/api/v1/products/{id}/reviews", ["page", "limit", "sort", "rating"]),
    ("GET",    "/api/v1/comments",            ["page", "limit", "sort"]),
    ("POST",   "/api/v1/comments",            ["text", "parent_id"]),
    ("DELETE", "/api/v1/comments/{id}",       []),
    # Notifications & messages
    ("GET",    "/api/v1/notifications",       ["page", "limit", "read", "type"]),
    ("PATCH",  "/api/v1/notifications/{id}",  ["read"]),
    ("POST",   "/api/v1/notifications/mark-read", ["ids"]),
    ("GET",    "/api/v1/messages",            ["page", "limit", "sort", "read"]),
    ("POST",   "/api/v1/messages",            ["subject", "body"]),
    ("GET",    "/api/v1/messages/{id}",       []),
    ("DELETE", "/api/v1/messages/{id}",       []),
    # Files / uploads
    ("GET",    "/api/v1/files",               ["page", "limit", "sort", "type"]),
    ("GET",    "/api/v1/files/{id}",          []),
    ("DELETE", "/api/v1/files/{id}",          []),
    # Reports / analytics
    ("GET",    "/api/v1/reports",             ["page", "limit", "from", "to", "sort"]),
    ("GET",    "/api/v1/reports/{id}",        []),
    ("GET",    "/api/v1/analytics/events",    ["from", "to", "event_type", "page", "limit"]),
    ("GET",    "/api/v1/analytics/summary",   ["from", "to"]),
    ("GET",    "/api/v1/analytics/users",     ["from", "to", "page", "limit"]),
    # Tags / labels
    ("GET",    "/api/v1/tags",                []),
    ("POST",   "/api/v1/tags",                ["name", "color"]),
    ("GET",    "/api/v1/labels",              []),
    # Admin
    ("GET",    "/admin/users",                ["page", "limit", "sort", "status", "role"]),
    ("GET",    "/admin/dashboard",            []),
    ("GET",    "/admin/logs",                 ["from", "to", "level", "page"]),
    ("GET",    "/admin/settings",             []),
    ("PATCH",  "/admin/settings",             ["theme", "notifications", "language", "timezone"]),
    # Health / meta
    ("GET",    "/healthz",                    []),
    ("GET",    "/readyz",                     []),
    ("GET",    "/api/v1/status",              []),
    ("GET",    "/api/v2/config",              []),
    ("GET",    "/metrics",                    []),
    ("GET",    "/api/v1/version",             []),
    # Webhooks / integrations
    ("GET",    "/api/v1/webhooks",            []),
    ("POST",   "/api/v1/webhooks",            ["url", "events", "secret"]),
    ("DELETE", "/api/v1/webhooks/{id}",       []),
    # Payments / billing
    ("POST",   "/api/v1/payments",            ["amount", "currency", "plan"]),
    ("GET",    "/api/v1/payments",            ["page", "limit", "status", "from", "to"]),
    ("GET",    "/api/v1/payments/{id}",       []),
    ("GET",    "/api/v1/invoices",            ["page", "limit", "status", "from", "to"]),
    ("GET",    "/api/v1/invoices/{id}",       []),
    ("GET",    "/api/v1/subscriptions",       ["page", "limit", "status"]),
    ("POST",   "/api/v1/subscriptions",       ["plan", "billing_cycle", "coupon"]),
    ("PATCH",  "/api/v1/subscriptions/{id}",  ["plan", "billing_cycle"]),
    ("DELETE", "/api/v1/subscriptions/{id}",  []),
    # Addresses
    ("GET",    "/api/v1/addresses",           []),
    ("POST",   "/api/v1/addresses",           ["street", "city", "country", "zip"]),
    ("PUT",    "/api/v1/addresses/{id}",      ["street", "city", "country", "zip"]),
    ("DELETE", "/api/v1/addresses/{id}",      []),
    # Sessions / devices
    ("GET",    "/api/v1/sessions",            ["page", "limit"]),
    ("DELETE", "/api/v1/sessions/{id}",       []),
    ("GET",    "/api/v1/devices",             ["page", "limit", "type"]),
    ("DELETE", "/api/v1/devices/{id}",        []),
    # Audit / events
    ("GET",    "/api/v1/events",              ["page", "limit", "from", "to", "event_type"]),
    ("GET",    "/api/v1/audit",               ["page", "limit", "from", "to", "action"]),
    # Exports
    ("POST",   "/api/v1/exports",             ["format", "type", "from", "to"]),
    ("GET",    "/api/v1/exports/{id}",        []),
    # Permissions / roles
    ("GET",    "/api/v1/permissions",         []),
    ("GET",    "/api/v1/roles",               ["page", "limit"]),
    ("POST",   "/api/v1/roles",               ["name", "description"]),
    ("DELETE", "/api/v1/roles/{id}",          []),
    # User sub-resources
    ("GET",    "/api/v1/users/{id}/permissions", []),
    ("GET",    "/api/v1/users/{id}/sessions",    ["page", "limit"]),
    ("GET",    "/api/v1/users/{id}/orders",      ["page", "limit", "status"]),
    # Product sub-resources
    ("GET",    "/api/v1/products/{id}/images",   []),
    ("GET",    "/api/v1/products/{id}/related",  ["limit"]),
    ("GET",    "/api/v1/products/{id}/reviews",  ["page", "limit", "sort", "rating"]),
    # Feature flags / config
    ("GET",    "/api/v1/features",            ["filter"]),
    ("GET",    "/api/v1/config",              []),
    ("PUT",    "/api/v1/config/{key}",        ["value"]),
    # Different API namespaces
    ("GET",    "/v2/users",                   ["page", "limit", "sort", "filter"]),
    ("GET",    "/v2/products",               ["page", "limit", "q", "category"]),
    ("POST",   "/v2/orders",                  ["product_id", "qty", "coupon"]),
    ("GET",    "/v2/search",                  ["q", "type", "page", "limit"]),
    ("GET",    "/rest/v1/catalog",            ["page", "limit", "q", "category"]),
    ("GET",    "/rest/v1/users",              ["page", "limit", "sort", "status"]),
    ("POST",   "/rest/v1/auth/token",         ["email", "password"]),
    ("GET",    "/api/v3/analytics",           ["from", "to", "event_type", "limit"]),
    ("GET",    "/api/v3/users",               ["page", "limit", "sort", "role"]),
    # GraphQL
    ("POST",   "/graphql",                    ["query", "variables"]),
    ("POST",   "/api/graphql",                ["query", "variables"]),
    # Internal / ops
    ("GET",    "/internal/health",            []),
    ("GET",    "/actuator/health",            []),
    ("GET",    "/actuator/metrics",           []),
    ("GET",    "/actuator/info",              []),
    # More admin
    ("GET",    "/admin/analytics",            ["from", "to", "type"]),
    ("GET",    "/admin/reports",              ["page", "limit", "from", "to"]),
    ("POST",   "/admin/users/{id}/ban",       ["reason", "duration"]),
    ("POST",   "/admin/users/{id}/reset",     []),
    ("GET",    "/admin/audit",                ["page", "limit", "from", "to", "action"]),
    # Bulk operations
    ("POST",   "/api/v1/users/bulk",          ["action", "ids"]),
    ("POST",   "/api/v1/products/bulk",       ["action", "ids"]),
    ("POST",   "/api/v1/orders/bulk",         ["action", "ids", "status"]),
]

_BENIGN_QUERIES = [
    # Shopping / e-commerce
    "blue shirt", "summer sale", "size medium", "best sellers", "new arrivals",
    "red shoes", "wireless headphones", "running shoes men", "coffee maker",
    "winter jacket", "smartphone case", "office chair", "gaming mouse",
    "yoga mat", "kitchen knives", "bluetooth speaker", "standing desk",
    "protein powder", "laptop bag", "mechanical keyboard", "organic food",
    "led desk lamp", "noise cancelling headphones", "air purifier",
    # Natural language with SQL/HTML-like tokens (benign edge cases)
    "select size guide", "order status update", "group discount policy",
    "union membership benefits", "drop shipping service", "insert coin arcade",
    "script writing course", "table lamp modern design", "where to buy",
    "having trouble logging in", "cast iron cookware review",
    "alert notification settings", "how to track my order",
    # Other domains
    "customer service contact", "delivery tracking number",
    "product warranty information", "compare models 2024",
    "user guide download", "return policy details", "subscription upgrade",
    "account settings help", "two factor authentication setup",
]
_CATEGORIES = [
    "electronics", "clothing", "furniture", "books", "sports", "toys",
    "home-garden", "beauty", "automotive", "grocery", "health", "office",
    "music", "tools", "pet-supplies", "baby", "jewelry", "software",
    "travel", "food-beverage", "gaming", "art-crafts", "outdoor",
    "fitness", "photography", "kitchen", "lighting", "storage",
]
_DATE_RANGE_PAIRS = [
    ("2024-01-01", "2024-03-31"), ("2024-04-01", "2024-06-30"),
    ("2024-07-01", "2024-09-30"), ("2024-10-01", "2024-12-31"),
    ("2025-01-01", "2025-03-31"), ("2025-04-01", "2025-06-30"),
    ("2023-01-01", "2023-06-30"), ("2023-07-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"), ("2023-01-01", "2023-12-31"),
]
_SORT_FIELDS   = ["created_at", "updated_at", "id", "name", "price", "rating",
                  "total", "views", "popularity", "relevance", "score",
                  "date", "amount", "size", "position", "rank", "modified_at"]
_STATUS_VALUES = ["active", "inactive", "pending", "archived", "draft",
                  "published", "cancelled", "completed", "processing",
                  "failed", "paused", "suspended", "verified", "unverified"]
_FIRST_NAMES   = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank",
                  "Ivy", "Jack", "Kate", "Liam", "Mia", "Nora", "Oscar", "Pam",
                  "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy",
                  "Xander", "Yara", "Zoe", "Noah", "Emma", "Oliver", "Sophia"]
_LAST_NAMES    = ["Smith", "Jones", "Lee", "Brown", "Garcia", "Wilson",
                  "Taylor", "Anderson", "Martinez", "Johnson", "Williams",
                  "Davis", "Miller", "Moore", "Jackson", "Harris", "Clark"]
_CITIES        = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                  "London", "Paris", "Berlin", "Tokyo", "Sydney",
                  "Toronto", "Amsterdam", "Singapore", "São Paulo", "Mumbai",
                  "Seoul", "Barcelona", "Vienna", "Melbourne", "Dublin"]
_EMAIL_DOMAINS = ["example.com", "test.org", "demo.net", "sample.io",
                  "acme.com", "webmail.org", "mailtest.net", "devnull.io"]
_COMPANIES     = ["Acme Corp", "Globex", "Initech", "Umbrella Ltd",
                  "Stark Industries", "Wayne Enterprises", "Soylent Corp",
                  "Hooli", "Pied Piper", "Dunder Mifflin"]
_OCCUPATIONS   = ["Software engineer", "Product manager", "Data scientist",
                  "Designer", "DevOps engineer", "Frontend developer",
                  "Backend developer", "QA engineer", "Security researcher",
                  "ML engineer"]


def _make_benign_record(rng: random.Random) -> HttpRecord:
    """Generate one benign record.

    Unlike attack records, there is no injection slot — every param is filled
    with an innocent type-appropriate value via _fill_params.  The endpoint,
    method, param names, and body format are drawn from the same
    _ENDPOINT_REGISTRY used by attack framing.
    """
    method, path_tmpl, params = rng.choice(_ENDPOINT_REGISTRY)
    path     = path_tmpl.replace("{id}", str(rng.randint(1, 9999)))
    has_body = method in ("POST", "PUT", "PATCH")
    filled   = _fill_params(params, rng)  # all innocent, no injection

    if has_body:
        qs   = ""
        body = json.dumps(filled) if filled else ""
        ct   = "application/json" if filled else None
    else:
        qs   = _params_to_qs(filled, rng)
        body = ""
        ct   = None

    headers = DIST.sample_headers(rng, method, has_body=has_body)
    if ct:
        headers["Content-Type"] = ct

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


def _parse_llm_json(text: str) -> list[dict] | None:
    """Parse an LLM response that should be a JSON array of HTTP request dicts."""
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
    try:
        r = json.loads(text)
        return r if isinstance(r, list) else ([r] if isinstance(r, dict) else None)
    except json.JSONDecodeError:
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
            source="aug_llm_benign",
        ).build_raw()
    except Exception:
        return None


def generate_llm_benign(
    provider:        str,
    model:           str,
    max_tok:         int,
    n_benign:        int,
    n_edge:          int,
    rng:             random.Random,
    temperature:     float = 0.7,
    model_path:      str   = "",
    ollama_base_url: str   = "http://localhost:11434",
    request_timeout: int   = 30,
) -> tuple[list[HttpRecord], list[HttpRecord]]:
    """Generate benign (+ edge-case benign) HTTP records via an LLM.

    Returns
    -------
    standard : list[HttpRecord]
        Standard benign records (up to n_benign).
    edge : list[HttpRecord]
        Edge-case benign records (up to n_edge).
    """
    BATCH = 10
    records: list[HttpRecord] = []

    log.info(f"LLM benign generation: target={n_benign:,}, edge_case={n_edge:,} via {provider}")

    def _llm(prompt: str) -> list[dict] | None:
        text = call_llm(
            provider        = provider,
            system          = _BENIGN_SYSTEM,
            user            = prompt,
            model           = model,
            max_tokens      = max_tok,
            temperature     = temperature,
            model_path      = model_path,
            ollama_base_url = ollama_base_url,
            request_timeout = request_timeout,
        )
        return _parse_llm_json(text) if text else None

    MAX_CONSECUTIVE_FAILURES = 5

    with Progress(SpinnerColumn(), "[progress.description]{task.description}",
                  BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as prog:

        # Standard benign
        t_benign = prog.add_task(f"LLM benign [{provider}]", total=n_benign)
        consecutive_failures = 0
        while len(records) < n_benign:
            prompt = rng.choice(_BENIGN_PROMPTS).format(n=BATCH)
            result = _llm(prompt)
            if result:
                before = len(records)
                for d in result:
                    r = _llm_dict_to_record(d)
                    if r:
                        records.append(r)
                added = len(records) - before
                if added:
                    consecutive_failures = 0
                    prog.advance(t_benign, added)
                else:
                    consecutive_failures += 1
                    log.warning(f"LLM benign: response yielded no records ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
            else:
                consecutive_failures += 1
                log.warning(f"LLM benign: call_llm returned None ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                time.sleep(1.0)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log.warning(f"LLM benign: aborting after {MAX_CONSECUTIVE_FAILURES} consecutive failures — {len(records)}/{n_benign} records collected")
                break
            time.sleep(0.2)

        # Edge-case benign
        edge_records: list[HttpRecord] = []
        edge_prompts = [
            f"Generate {BATCH} HTTP GET requests where search queries contain SELECT, DROP, UNION or SLEEP in natural English sentences — NOT SQL injections.",
            f"Generate {BATCH} HTTP POST requests where JSON body contains HTML-like content that is NOT XSS, e.g. blog posts with <em> or <strong> tags.",
            f"Generate {BATCH} HTTP GET requests to file endpoints with relative paths like ../docs/readme.md — legitimate paths, NOT path traversals.",
        ]
        t_edge = prog.add_task("LLM edge-case benign", total=n_edge)
        consecutive_failures = 0
        while len(edge_records) < n_edge:
            prompt = rng.choice(edge_prompts)
            result = _llm(prompt)
            if result:
                before = len(edge_records)
                for d in result:
                    r = _llm_dict_to_record(d)
                    if r:
                        object.__setattr__(r, "attack_class", "benign_edge_case")
                        object.__setattr__(r, "source", "aug_llm_edge")
                        edge_records.append(r)
                added = len(edge_records) - before
                if added:
                    consecutive_failures = 0
                    prog.advance(t_edge, added)
                else:
                    consecutive_failures += 1
                    log.warning(f"LLM edge-case: response yielded no records ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
            else:
                consecutive_failures += 1
                log.warning(f"LLM edge-case: call_llm returned None ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                time.sleep(1.0)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log.warning(f"LLM edge-case: aborting after {MAX_CONSECUTIVE_FAILURES} consecutive failures — {len(edge_records)}/{n_edge} records collected")
                break
            time.sleep(0.3)

    standard = records[:n_benign]
    edge     = edge_records[:n_edge]
    log.info(f"LLM benign: {len(standard):,} standard + {len(edge):,} edge-case")
    return standard, edge


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    configure_root()
    cfg     = load_config(args.config)
    aug_cfg = cfg.augmentation
    rng     = random.Random(cfg.project.seed)

    require_inputs({
        "data/augmented/synthesis/synthesized_attacks.parquet": "make data_augment_synthesis",
    })
    if check_output(
        Path(cfg.paths.data_augmented) / "framed" / "framed_records.parquet",
        args.force, "Stage 3.3 request framing"
    ):
        return

    # Load PCAP-fitted distributions from Module B if available
    dist_profile = Path(cfg.paths.reports) / "metrics" / "benign_distribution.json"
    if dist_profile.exists():
        try:
            DIST.configure(json.loads(dist_profile.read_text()))
            log.info(f"Loaded benign distribution profile from {dist_profile}")
        except Exception as exc:
            log.warning(f"Could not load benign distribution profile: {exc}")
    else:
        log.info("No benign_distribution.json found — using internal header defaults")

    out_dir = Path(cfg.paths.data_augmented) / "framed"
    out_dir.mkdir(parents=True, exist_ok=True)

    timer = StepTimer()
    all_records: list[HttpRecord] = []

    # ── 1. Re-frame Module A attack payloads ──────────────────────────────
    synth_path = Path(cfg.paths.data_augmented) / "synthesis" / "synthesized_attacks.parquet"
    with timer.step("reframe_attacks"):
        attack_records = reframe_synthesized(synth_path, rng)
    all_records.extend(attack_records)

    # ── 2. Programmatic benign REST traffic ───────────────────────────────
    n_rest = getattr(getattr(aug_cfg, "benign", None), "rest_samples", 10_000)
    with timer.step("benign_rest"):
        with Progress(SpinnerColumn(), "[progress.description]{task.description}",
                      BarColumn(), MofNCompleteColumn(), TimeElapsedColumn()) as prog:
            task = prog.add_task("Benign REST records", total=n_rest)
            benign_rest = []
            for _ in range(n_rest):
                benign_rest.append(_make_benign_record(rng))
                prog.advance(task)
    all_records.extend(benign_rest)
    log.info(f"Generated {len(benign_rest):,} programmatic benign REST records")

    # ── 3. Optional: LLM benign framing ──────────────────────────────────
    llm_cfg  = aug_cfg.llm
    provider = args.provider or llm_cfg.provider

    llm_standard: list[HttpRecord] = []
    llm_edge:     list[HttpRecord] = []
    if provider:
        with timer.step("llm_benign"):
            llm_standard, llm_edge = generate_llm_benign(
                provider        = provider,
                model           = llm_cfg.model,
                max_tok         = llm_cfg.max_tokens,
                n_benign        = llm_cfg.samples_benign,
                n_edge          = llm_cfg.samples_edge_case,
                rng             = rng,
                temperature     = llm_cfg.temperature,
                model_path      = llm_cfg.model_path,
                ollama_base_url = llm_cfg.ollama_base_url,
                request_timeout = llm_cfg.request_timeout,
            )
        all_records.extend(llm_standard)
        all_records.extend(llm_edge)
    else:
        log.info("LLM framing skipped (no provider configured)")

    # ── 4. Write ──────────────────────────────────────────────────────────
    out_path = out_dir / "framed_records.parquet"
    with timer.step("parquet_write"):
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
        "n_attack_reframed":  len(attack_records),
        "n_benign_rest":      len(benign_rest),
        "n_llm_standard":     len(llm_standard),
        "n_llm_edge_case":    len(llm_edge),
        "total":              len(all_records),
        "llm_provider":       provider or "none",
        "llm_model":          llm_cfg.model or "none",
        "timings_s":          timer.timings,
    }, indent=2))

    try:
        import mlflow
        from ai_waf_v2.utils.mlflow_utils import init_experiment, log_metrics_dict
        init_experiment(cfg)
        with mlflow.start_run(run_name="03_request_framing"):
            mlflow.log_params({
                "llm_provider":   provider or "none",
                "llm_model":      llm_cfg.model or "none",
                "n_rest_samples": n_rest,
            })
            log_metrics_dict({
                "n_attack_reframed": float(len(attack_records)),
                "n_benign_rest":     float(len(benign_rest)),
                "n_llm_standard":    float(len(llm_standard)),
                "n_llm_edge_case":   float(len(llm_edge)),
                "total_records":     float(len(all_records)),
                "n_attack":          float(n_attack),
                "n_benign":          float(n_benign),
            })
            timer.log_mlflow()
            mlflow.log_artifact(str(stats_path))
    except Exception as exc:
        log.warning(f"MLflow logging skipped: {exc}", exc_info=True)

    if args.save:
        from ai_waf_v2.utils.hub import push_folder
        push_folder(
            cfg,
            folder=Path(cfg.paths.data_augmented) / "framed",
            repo_key="dataset_framed",
            repo_type="dataset",
            commit_message=f"Stage 3.3 framing: {len(all_records):,} records (attack={n_attack:,}, benign={n_benign:,})",
            revision=args.revision,
            dry_run=args.dry_run,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="config/pipeline.yaml")
    p.add_argument("--provider", default=None, choices=["anthropic", "google", "local", "ollama"])
    p.add_argument("--force",    action="store_true", help="Re-run even if outputs already exist")
    p.add_argument("--save",     action="store_true", help="Push outputs to HuggingFace Hub")
    p.add_argument("--revision", default="main",      help="HuggingFace revision/tag (default: main)")
    p.add_argument("--dry-run",  action="store_true", help="Preview hub push without uploading")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())