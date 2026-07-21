"""
ai_waf_v2/eval/leakage.py
-------------------------
Token↔label leakage diagnostics — does the dataset (or a model) key on incidental
constants instead of attack structure?

Two complementary tools:

1. ``token_leakage(texts, labels)`` — a model-free report. For every token it
   computes P(malicious | token), lift over the base rate, PMI, and the mutual
   information between token-presence and the label. Tokens that are both frequent
   and near-deterministic predictors are shortcut candidates; each is flagged as
   *incidental* (filler-shaped: a host, number, id) or structural. If a hostname or
   port tops this list, fillers are leaking; if SQL keywords / ``../`` / ``<script>``
   top it, the model is being taught the right thing.

2. ``counterfactual_auc_delta(predict_fn, ...)`` — swap the incidental values
   (``swap_fillers``) and re-score. A model that relies on structure barely moves
   (small delta); one that memorised constants collapses (large delta). Works with
   any ``predict_fn(list[str]) -> list[float]``, so it needs no particular model.

Pure standard library — no torch / numpy — so it runs on synthetic data in tests.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from ai_waf_v2.augment.fillers import rand_host, rand_int, rand_ip

# HTTP-aware boundaries (same spirit as HttpTokenizer's pre-tokenisation) so the
# report is interpretable at roughly the granularity the model sees.
_SPLIT_RE = re.compile(r"[\s?&=/.;:(){}\[\]<>#'\"%,|\\+*`]+")

_TLDS = ("com", "net", "org", "io", "co", "dev", "app", "cloud", "xyz", "info",
         "biz", "site", "tech", "us", "uk", "de", "fr", "in", "ru", "cn", "me", "gov")

Tokenizer = Callable[[str], "list[str]"]


def pretokenize(text: str, casefold: bool = True) -> list[str]:
    if casefold:
        text = text.lower()
    return [t for t in _SPLIT_RE.split(text) if t]


def looks_incidental(token: str) -> bool:
    """Heuristic: does the token look like a filler value (host label, number, id)
    rather than a structural keyword? A hint for the report, not a hard gate."""
    if token.isdigit():
        return True
    if len(token) >= 8 and re.fullmatch(r"[0-9a-f]+", token):  # hash / uuid chunk
        return True
    if len(token) >= 4 and re.search(r"[a-z]", token) and re.search(r"\d", token):
        return True  # mixed alnum → likely a random label like "mu9x3f"
    return False


@dataclass
class TokenStat:
    token: str
    df: int          # documents containing the token
    df_pos: int      # of those, malicious
    p_malicious: float
    lift: float      # p_malicious − base_rate
    pmi_pos: float   # log2 P(y=1 | token) / P(y=1)
    mi: float        # mutual information of token-presence vs label (bits)
    incidental: bool


@dataclass
class LeakageReport:
    n_docs: int
    n_pos: int
    base_rate: float
    n_tokens: int
    min_df: int
    max_mi: float
    n_strong: int              # frequent + near-deterministic predictors
    n_strong_incidental: int   # of those, filler-shaped → the real red flags
    top: list[TokenStat] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["top"] = [asdict(s) for s in self.top]
        return d


def _binary_mi(df: int, df_pos: int, n: int, n_pos: int) -> float:
    """Mutual information (bits) of token-presence (0/1) with the label (0/1)."""
    n_neg = n - n_pos
    a = df_pos            # present, malicious
    b = df - df_pos       # present, benign
    c = n_pos - df_pos    # absent, malicious
    d = n_neg - b         # absent, benign
    mi = 0.0
    for count, row, col in ((a, df, n_pos), (b, df, n_neg),
                            (c, n - df, n_pos), (d, n - df, n_neg)):
        if count <= 0 or row <= 0 or col <= 0:
            continue
        p_xy = count / n
        mi += p_xy * math.log2(p_xy / ((row / n) * (col / n)))
    return max(mi, 0.0)


def token_leakage(
    texts: Sequence[str],
    labels: Sequence[int],
    min_df: int = 5,
    top_k: int = 50,
    det_eps: float = 0.02,
    tokenizer: Tokenizer = pretokenize,
) -> LeakageReport:
    """Rank tokens by how strongly their presence predicts the label."""
    n = len(texts)
    n_pos = sum(1 for y in labels if y == 1)
    base = n_pos / n if n else 0.0

    df: Counter = Counter()
    df_pos: Counter = Counter()
    for text, y in zip(texts, labels):
        for t in set(tokenizer(text)):
            df[t] += 1
            if y == 1:
                df_pos[t] += 1

    stats: list[TokenStat] = []
    for t, d in df.items():
        if d < min_df:
            continue
        dp = df_pos[t]
        p_mal = dp / d
        p_t = d / n
        p_t_given_pos = (dp / n_pos) if n_pos else 0.0
        pmi = math.log2(p_t_given_pos / p_t) if p_t_given_pos > 0 else float("-inf")
        stats.append(TokenStat(
            token=t, df=d, df_pos=dp,
            p_malicious=round(p_mal, 4),
            lift=round(p_mal - base, 4),
            pmi_pos=round(pmi, 4) if math.isfinite(pmi) else -99.0,
            mi=round(_binary_mi(d, dp, n, n_pos), 6),
            incidental=looks_incidental(t),
        ))

    stats.sort(key=lambda s: s.mi, reverse=True)
    strong = [s for s in stats if s.p_malicious <= det_eps or s.p_malicious >= 1 - det_eps]
    return LeakageReport(
        n_docs=n, n_pos=n_pos, base_rate=round(base, 4),
        n_tokens=len(stats), min_df=min_df,
        max_mi=stats[0].mi if stats else 0.0,
        n_strong=len(strong),
        n_strong_incidental=sum(1 for s in strong if s.incidental),
        top=stats[:top_k],
    )


# ── Counterfactual filler swap ────────────────────────────────────────────────

_IPV4_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_HOST_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:" + "|".join(_TLDS) + r")\b", re.IGNORECASE)
_INT_RE = re.compile(r"\b\d{2,}\b")


def swap_fillers(text: str, rng) -> str:
    """
    Re-randomise incidental values (IPs, hostnames ending in a known TLD, numbers)
    while leaving structural tokens intact — the counterfactual transform. A model
    keyed on structure is unaffected; one keyed on a memorised host/number is not.
    """
    text = _IPV4_RE.sub(lambda _m: rand_ip(rng), text)
    text = _HOST_RE.sub(lambda _m: rand_host(rng), text)
    text = _INT_RE.sub(lambda _m: rand_int(rng), text)
    return text


def auc_roc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Rank-based AUC-ROC (Mann–Whitney U), tie-safe. 0.5 when scores are constant."""
    n_pos = sum(1 for y in labels if y == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1  # average 1-based rank across the tie group
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def counterfactual_auc_delta(
    predict_fn: Callable[[Sequence[str]], Sequence[float]],
    texts: Sequence[str],
    labels: Sequence[int],
    rng,
    transform: Callable[[str, object], str] = swap_fillers,
) -> dict:
    """
    AUC before vs. after swapping incidental values. ``delta`` near 0 means the
    predictor relies on structure (robust); a large positive ``delta`` means it
    relied on the swapped constants (leakage).
    """
    auc0 = auc_roc(predict_fn(list(texts)), labels)
    swapped = [transform(t, rng) for t in texts]
    auc1 = auc_roc(predict_fn(swapped), labels)
    return {"auc": round(auc0, 6), "auc_swapped": round(auc1, 6),
            "delta": round(auc0 - auc1, 6)}
