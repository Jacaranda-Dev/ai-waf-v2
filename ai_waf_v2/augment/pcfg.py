"""
ai_waf_v2/augment/pcfg.py
-------------------------
Recursive probabilistic context-free grammar (PCFG) engine for synthetic attack
payload generation (Stage 3.1). Grammars are *data* — loaded from YAML — so the
attack structure lives in ``config/pcfg_grammars.yaml`` rather than in code.

A production is ``(weight, [symbol, ...])``. Symbols (as YAML tokens in parens):

    T(str)   ("t:<literal>")   terminal literal
    N(str)   ("n:<NAME>")      nonterminal reference   → recursion happens here
    F(name)  ("f:<func>")      callable terminal from PCFG_FUNCS, fn(rng, ctx)->str
    OPENQ    ("openq")         pick a quote char, remember it in ctx, emit it
    CLOSEQ   ("closeq")        emit the quote remembered by the most recent OPENQ

``OPENQ``/``CLOSEQ`` form a small attribute grammar so opened quotes are closed
with the *same* character — a constraint a plain CFG cannot express.

Termination is guaranteed: once past ``max_depth``, expansion is restricted to
each nonterminal's minimal-cost productions (shortest derivation to terminals),
computed once by a fixpoint over the grammar. Unlike a flat prefix+payload+suffix
template, the engine reaches nested / variable-width structures (chained boolean
clauses, N-column ``UNION`` selects, nested tags) that downstream encoding and
obfuscation transforms then mutate further.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

Symbol = tuple      # ("t", str) | ("n", str) | ("f", str) | ("openq",) | ("closeq",)
Production = tuple  # (weight: float, body: list[Symbol])


def T(literal: str) -> Symbol:
    return ("t", literal)


def N(name: str) -> Symbol:
    return ("n", name)


def F(func_name: str) -> Symbol:
    return ("f", func_name)  # resolved against PCFG_FUNCS at expansion time


OPENQ: Symbol = ("openq",)
CLOSEQ: Symbol = ("closeq",)


# Named callable terminals. Referenced from YAML as "f:<name>" so grammars stay
# serialisable. Reserve `f:` for small *structural* vocabularies the model should
# learn (event-handler names, tag names); route incidental, high-cardinality
# values (hosts, ports, ids, numbers, JS bodies) through §NAME§ filler
# placeholders instead — see ai_waf_v2.augment.fillers.
PCFG_FUNCS: dict[str, Callable[[random.Random, dict], str]] = {
    "xss_evt": lambda r, _c: r.choice(
        ("onerror", "onload", "onfocus", "onmouseover", "ontoggle", "onpointerover")),
    "xss_tag": lambda r, _c: r.choice(("div", "span", "b", "p", "section", "details")),
}


def _grammar_min_cost(rules: dict[str, list[Production]]) -> dict[str, float]:
    """Fixpoint: minimal derivation cost (nonterminal expansions) to terminate each NT."""
    inf = float("inf")
    cost: dict[str, float] = {nt: inf for nt in rules}
    changed = True
    while changed:
        changed = False
        for nt, prods in rules.items():
            best = min(
                (1 + max((cost[s[1]] for s in body if s[0] == "n"), default=0)
                 for _, body in prods),
                default=inf,
            )
            if best < cost[nt]:
                cost[nt] = best
                changed = True
    return cost


class Pcfg:
    """A probabilistic recursive context-free grammar with a matched-quote attribute."""

    def __init__(
        self,
        name: str,
        rules: dict[str, list[Production]],
        start: str = "START",
        max_depth: int = 6,
        max_len: int = 256,
    ):
        if start not in rules:
            raise ValueError(f"PCFG '{name}': start symbol '{start}' has no rule")
        self.name = name
        self.rules = rules
        self.start = start
        self.max_depth = max_depth
        self.max_len = max_len
        cost = _grammar_min_cost(rules)
        unreachable = [nt for nt, c in cost.items() if c == float("inf")]
        if unreachable:
            raise ValueError(
                f"PCFG '{name}': nonterminals with no terminating derivation: {unreachable}")
        # Cheapest (shortest-to-terminate) productions per NT, used once past max_depth.
        self._cheap: dict[str, list[Production]] = {}
        for nt, prods in rules.items():
            costs = [1 + max((cost[s[1]] for s in body if s[0] == "n"), default=0)
                     for _, body in prods]
            lo = min(costs)
            self._cheap[nt] = [p for p, c in zip(prods, costs) if c == lo]

    def sample(self, rng: random.Random) -> str:
        return self._expand(self.start, rng, {}, 0)

    def _expand(self, nt: str, rng: random.Random, ctx: dict, depth: int) -> str:
        prods = self._cheap[nt] if depth >= self.max_depth else self.rules[nt]
        _, body = rng.choices(prods, weights=[w for w, _ in prods], k=1)[0]
        out: list[str] = []
        for sym in body:
            tag = sym[0]
            if tag == "t":
                out.append(sym[1])
            elif tag == "n":
                out.append(self._expand(sym[1], rng, ctx, depth + 1))
            elif tag == "f":
                out.append(PCFG_FUNCS[sym[1]](rng, ctx))
            elif tag == "openq":
                ctx["quote"] = rng.choice(("'", '"'))
                out.append(ctx["quote"])
            elif tag == "closeq":
                out.append(ctx.get("quote", "'"))
        return "".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# YAML loader  (grammar-as-data)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_symbol(tok: str) -> Symbol:
    if tok == "openq":
        return OPENQ
    if tok == "closeq":
        return CLOSEQ
    kind, sep, val = tok.partition(":")
    if not sep:
        raise ValueError(f"PCFG symbol token missing ':' prefix: {tok!r}")
    if kind == "t":
        return T(val)
    if kind == "n":
        return N(val)
    if kind == "f":
        if val not in PCFG_FUNCS:
            raise ValueError(f"PCFG function '{val}' not registered in PCFG_FUNCS")
        return F(val)
    raise ValueError(f"Unrecognised PCFG symbol kind {kind!r} in token {tok!r}")


def _parse_rules(raw_rules: dict) -> dict[str, list[Production]]:
    rules: dict[str, list[Production]] = {}
    for nt, prods in raw_rules.items():
        parsed: list[Production] = []
        for weight, body in prods:
            parsed.append((float(weight), [_parse_symbol(t) for t in body]))
        rules[nt] = parsed
    return rules


def load_pcfg_grammars(path: str | Path) -> dict[str, Pcfg]:
    """
    Load PCFG grammars from a YAML file into ``{attack_class: Pcfg}``.

    Returns an empty dict if the file does not exist, so callers can transparently
    fall back to flat templates whenever no grammar is present. Each grammar spec:

        pcfg_grammars:
          sqli:
            start: START        # optional (default START)
            max_depth: 6        # optional
            max_len: 256        # optional
            rules:
              START:
                - [0.6, ["n:INJ", "t: ", "n:COMMENT"]]
                - [0.4, ["n:INJ"]]
    """
    import yaml

    path = Path(path)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    grammars = data.get("pcfg_grammars", data)
    out: dict[str, Pcfg] = {}
    for name, spec in grammars.items():
        out[name] = Pcfg(
            name=name,
            rules=_parse_rules(spec["rules"]),
            start=spec.get("start", "START"),
            max_depth=int(spec.get("max_depth", 6)),
            max_len=int(spec.get("max_len", 256)),
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sampler  (PCFG where available, flat-template fallback otherwise)
# ─────────────────────────────────────────────────────────────────────────────

class PcfgSampler:
    """
    Produces up to ``n`` unique payloads for an attack class.

    - class present in ``registry``  → sample the recursive PCFG (deduplicated)
    - class absent from ``registry`` → delegate entirely to ``fallback``
    - PCFG can't produce ``n`` uniques → top up the remainder from ``fallback``

    ``fallback`` has signature ``(attack_class, n, rng) -> list[str]`` — in the
    pipeline it is the flat ``GrammarGenerator`` templates, which therefore remain
    the backup whenever no PCFG grammar is present.
    """

    def __init__(
        self,
        registry: dict[str, Pcfg],
        fallback: Callable[[str, int, random.Random], list[str]],
        oversample: int = 8,
    ):
        self.registry = registry
        self.fallback = fallback
        self.oversample = oversample

    def generate(self, attack_class: str, n: int, rng: random.Random) -> list[str]:
        pcfg = self.registry.get(attack_class)
        if pcfg is None:
            return self.fallback(attack_class, n, rng)

        out: list[str] = []
        seen: set[str] = set()
        for _ in range(max(n * self.oversample, n + 16)):
            if len(out) >= n:
                break
            s = pcfg.sample(rng)
            if s and len(s) <= pcfg.max_len and s not in seen:
                seen.add(s)
                out.append(s)
        if len(out) < n:  # PCFG ran dry → top up from the flat templates
            out.extend(self.fallback(attack_class, n - len(out), rng))
        return out[:n]
