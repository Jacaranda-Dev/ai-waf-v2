"""
ai_waf_v2.eval.adversarial
----------------------
Adversarial robustness evaluation for WAF models.

Tests model performance against:
1. Encoding obfuscations (URL encode, double-encode, hex, unicode, base64 in params)
2. Structural evasion (comment insertion, case variation, whitespace manipulation)
3. Payload fragmentation (split injection across multiple parameters)
4. Novel attack patterns from held-out evasion wordlists

Usage
-----
    from ai_waf_v2.eval.adversarial import AdversarialEvaluator

    evaluator = AdversarialEvaluator(model, tokenizer, device="cuda")
    results = evaluator.run(malicious_records, tamper_scripts=["space2comment"])
    evaluator.save_report(results, "reports/metrics/adversarial.json")
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch

from ai_waf_v2.data.schema import HttpRecord
from ai_waf_v2.eval.metrics import compute_metrics

if TYPE_CHECKING:
    from ai_waf_v2.models.head import WafClassifier


# ─────────────────────────────────────────────────────────
# Tamper / obfuscation functions
# ─────────────────────────────────────────────────────────

def tamper_space2comment(payload: str) -> str:
    """Replace spaces with /**/ SQL comment to bypass simple space detection."""
    return payload.replace(" ", "/**/")


def tamper_randomcase(payload: str) -> str:
    """Randomly alternate case of alphabetic characters."""
    import random
    return "".join(
        c.upper() if random.random() > 0.5 else c.lower()
        for c in payload
    )


def tamper_url_encode(payload: str) -> str:
    """URL-encode every character in the payload."""
    return urllib.parse.quote(payload, safe="")


def tamper_double_url_encode(payload: str) -> str:
    """Double URL-encode the payload."""
    return urllib.parse.quote(urllib.parse.quote(payload, safe=""), safe="")


def tamper_hex_encode(payload: str) -> str:
    """Hex-encode alphabetic characters (e.g. a → 0x61 SQL hex literal)."""
    result = []
    for c in payload:
        if c.isalpha():
            result.append(f"0x{ord(c):02x}")
        else:
            result.append(c)
    return "".join(result)


def tamper_between_comments(payload: str) -> str:
    """Insert SQL comments between each character."""
    return "/**/".join(payload)


def tamper_base64_param(payload: str) -> str:
    """Base64-encode the payload (simulates base64-wrapped injection)."""
    import base64
    return base64.b64encode(payload.encode()).decode()


def tamper_unicode_escape(payload: str) -> str:
    """Unicode-escape alphabetic characters."""
    result = []
    for c in payload:
        if c.isalpha():
            result.append(f"\\u{ord(c):04x}")
        else:
            result.append(c)
    return "".join(result)


TAMPER_REGISTRY: dict[str, Callable[[str], str]] = {
    "space2comment":      tamper_space2comment,
    "randomcase":         tamper_randomcase,
    "url_encode":         tamper_url_encode,
    "double_url_encode":  tamper_double_url_encode,
    "hex_encode":         tamper_hex_encode,
    "between_comments":   tamper_between_comments,
    "base64_param":       tamper_base64_param,
    "unicode_escape":     tamper_unicode_escape,
}


# ─────────────────────────────────────────────────────────
# Adversarial Evaluator
# ─────────────────────────────────────────────────────────

@dataclass
class AdversarialResult:
    tamper_name:     str
    n_samples:       int
    detection_rate:  float    # recall on transformed malicious samples
    evasion_rate:    float    # 1 - detection_rate
    metrics:         dict     = field(default_factory=dict)
    examples:        list[dict] = field(default_factory=list)  # worst failures


class AdversarialEvaluator:
    """
    Evaluate model robustness against obfuscated attack payloads.

    Parameters
    ----------
    model       : WafClassifier — any model with a predict() method
    tokenizer   : HuggingFace Tokenizer
    device      : "cuda" | "cpu"
    seq_len     : int — max sequence length
    batch_size  : int — inference batch size
    threshold   : float — classification threshold for malicious class
    n_failure_examples : int — number of evasion failures to record per tamper
    """

    def __init__(
        self,
        model:              "WafClassifier",
        tokenizer:          object,
        device:             str  = "cuda",
        seq_len:            int  = 256,
        batch_size:         int  = 64,
        threshold:          float = 0.5,
        n_failure_examples: int  = 10,
    ) -> None:
        self.model              = model
        self.tokenizer          = tokenizer
        self.device             = torch.device(device)
        self.seq_len            = seq_len
        self.batch_size         = batch_size
        self.threshold          = threshold
        self.n_failure_examples = n_failure_examples

        self.model.to(self.device).eval()

    def _apply_tamper(
        self,
        records: list[HttpRecord],
        tamper_fn: Callable[[str], str],
    ) -> list[HttpRecord]:
        """
        Apply a tamper function to the query_string and body of each record,
        then rebuild the raw HTTP string.
        """
        tampered = []
        for r in records:
            d = r.model_dump()
            d["query_string"] = tamper_fn(r.query_string)
            if r.body:
                d["body"] = tamper_fn(r.body)
            new_r = HttpRecord(**d).build_raw()
            tampered.append(new_r)
        return tampered

    def _tokenize_batch(
        self,
        records: list[HttpRecord],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a list of records and return (input_ids, attention_mask)."""
        texts = [r.raw for r in records]
        input_ids_list, mask_list = [], []

        for text in texts:
            enc     = self.tokenizer.encode(text)
            ids     = enc.ids[: self.seq_len]
            mask    = enc.attention_mask[: self.seq_len]
            pad_len = self.seq_len - len(ids)
            pad_id  = self.tokenizer.token_to_id("[PAD]") or 0
            ids     = ids  + [pad_id] * pad_len
            mask    = mask + [0]      * pad_len
            input_ids_list.append(ids)
            mask_list.append(mask)

        return (
            torch.tensor(input_ids_list, dtype=torch.long),
            torch.tensor(mask_list,      dtype=torch.long),
        )

    def _predict_records(
        self,
        records: list[HttpRecord],
        labels:  list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run inference on a list of records in batches."""
        all_preds, all_probs = [], []

        for i in range(0, len(records), self.batch_size):
            batch_records = records[i : i + self.batch_size]
            input_ids, attention_mask = self._tokenize_batch(batch_records)
            input_ids      = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                preds, probs = self.model.predict(
                    input_ids, attention_mask, threshold=self.threshold
                )
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

        preds_t  = torch.cat(all_preds)
        probs_t  = torch.cat(all_probs)
        labels_t = torch.tensor(labels, dtype=torch.long)
        return preds_t, probs_t, labels_t

    def run(
        self,
        malicious_records: list[HttpRecord],
        tamper_scripts:    list[str] | None = None,
        evasion_wordlists: list[str] | None = None,
    ) -> list[AdversarialResult]:
        """
        Run adversarial evaluation.

        Parameters
        ----------
        malicious_records : list of HttpRecord with label=1
        tamper_scripts    : list of tamper function names (from TAMPER_REGISTRY)
        evasion_wordlists : list of paths to wordlist files (one payload per line)

        Returns
        -------
        list of AdversarialResult, one per tamper/wordlist
        """
        if tamper_scripts is None:
            tamper_scripts = list(TAMPER_REGISTRY.keys())

        labels = [1] * len(malicious_records)
        results: list[AdversarialResult] = []

        for name in tamper_scripts:
            if name not in TAMPER_REGISTRY:
                continue

            tamper_fn = TAMPER_REGISTRY[name]
            tampered  = self._apply_tamper(malicious_records, tamper_fn)
            preds, probs, lbls = self._predict_records(tampered, labels)

            metrics = compute_metrics(preds, probs, lbls)
            detection_rate = metrics["recall"]
            evasion_rate   = 1.0 - detection_rate

            # Collect failure examples (malicious predicted as benign)
            failures = []
            for i, (p, r) in enumerate(zip(preds.tolist(), tampered)):
                if p == 0 and len(failures) < self.n_failure_examples:
                    failures.append({
                        "original_raw": malicious_records[i].raw[:200],
                        "tampered_raw": r.raw[:200],
                        "prob_malicious": round(probs[i].item(), 4),
                    })

            results.append(AdversarialResult(
                tamper_name=name,
                n_samples=len(malicious_records),
                detection_rate=round(detection_rate, 4),
                evasion_rate=round(evasion_rate, 4),
                metrics=metrics,
                examples=failures,
            ))

        # Wordlist-based evasion
        if evasion_wordlists:
            for wl_path in evasion_wordlists:
                wl_result = self._eval_wordlist(wl_path)
                if wl_result:
                    results.append(wl_result)

        return results

    def _eval_wordlist(self, wordlist_path: str) -> AdversarialResult | None:
        """Evaluate model on raw payloads from a wordlist file."""
        path = Path(wordlist_path)
        if not path.exists():
            return None

        payloads = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        if not payloads:
            return None

        # Wrap each payload in a minimal GET request
        records = []
        for payload in payloads:
            r = HttpRecord(
                method="GET",
                path="/test",
                query_string=f"id={payload}",
                label=1,
                attack_class="evasion_wordlist",
                source=path.name,
            ).build_raw()
            records.append(r)

        labels = [1] * len(records)
        preds, probs, lbls = self._predict_records(records, labels)
        metrics = compute_metrics(preds, probs, lbls)

        return AdversarialResult(
            tamper_name=f"wordlist:{path.stem}",
            n_samples=len(records),
            detection_rate=round(metrics["recall"], 4),
            evasion_rate=round(1.0 - metrics["recall"], 4),
            metrics=metrics,
        )

    @staticmethod
    def save_report(results: list[AdversarialResult], path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "tamper_name":    r.tamper_name,
                "n_samples":      r.n_samples,
                "detection_rate": r.detection_rate,
                "evasion_rate":   r.evasion_rate,
                "metrics":        r.metrics,
                "failure_examples": r.examples[:5],
            }
            for r in results
        ]
        with path.open("w") as f:
            json.dump(data, f, indent=2)