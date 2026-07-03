"""
ai_waf_v2.augment — synthetic attack payload generation helpers.

Reusable, dependency-light building blocks for Stage 3 (data augmentation) that
are imported by the (non-importable, digit-prefixed) stage scripts and covered by
the test suite:

- `pcfg`     — recursive probabilistic context-free grammar engine + YAML loader
- `validity` — per-class structural validity checks for synthetic payloads
"""

from ai_waf_v2.augment.fillers import (
    FILLERS,
    fill_placeholders,
    placeholder_names,
    scrub_textbook_hosts,
)
from ai_waf_v2.augment.pcfg import (
    CLOSEQ,
    OPENQ,
    PCFG_FUNCS,
    F,
    N,
    Pcfg,
    PcfgSampler,
    T,
    load_pcfg_grammars,
)
from ai_waf_v2.augment.validity import VALIDATORS, is_valid_for_class

__all__ = [
    "Pcfg",
    "PcfgSampler",
    "load_pcfg_grammars",
    "PCFG_FUNCS",
    "T",
    "N",
    "F",
    "OPENQ",
    "CLOSEQ",
    "FILLERS",
    "fill_placeholders",
    "placeholder_names",
    "scrub_textbook_hosts",
    "is_valid_for_class",
    "VALIDATORS",
]
