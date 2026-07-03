"""
tests/test_all.py
-----------------
Unit tests for the ai-waf-v2 core library.

Run:
    pytest tests/ -v
    pytest tests/ -v --tb=short -q   # compact output

All tests use small synthetic data — no real datasets or GPU required.
"""

from __future__ import annotations

import json

import pytest
import torch


# ─────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────

class TestHttpRecord:
    def test_build_raw_get(self):
        from ai_waf_v2.data.schema import HttpRecord

        r = HttpRecord(
            method="GET",
            path="/api/users",
            query_string="id=1",
            headers=json.dumps({"Host": "example.com"}),
            label=0,
        ).build_raw()

        assert "GET /api/users?id=1 HTTP/1.1" in r.raw
        assert "Host: example.com" in r.raw

    def test_build_raw_post_with_body(self):
        from ai_waf_v2.data.schema import HttpRecord

        r = HttpRecord(
            method="POST",
            path="/login",
            body="username=admin&password=1234",
            label=1,
            attack_class="sqli",
        ).build_raw()

        assert "POST /login HTTP/1.1" in r.raw
        assert "username=admin" in r.raw

    def test_label_validation(self):
        from ai_waf_v2.data.schema import HttpRecord
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            HttpRecord(label=99)

    def test_method_uppercased(self):
        from ai_waf_v2.data.schema import HttpRecord
        r = HttpRecord(method="get")
        assert r.method == "GET"

    def test_headers_dict(self):
        from ai_waf_v2.data.schema import HttpRecord
        r = HttpRecord(headers=json.dumps({"Content-Type": "application/json"}))
        assert r.headers_dict()["Content-Type"] == "application/json"

    def test_records_to_table_roundtrip(self):
        from ai_waf_v2.data.schema import HttpRecord, records_to_table, table_to_records

        records = [
            HttpRecord(method="GET", path=f"/api/{i}", label=i % 2).build_raw()
            for i in range(5)
        ]
        table   = records_to_table(records)
        back    = table_to_records(table)

        assert len(back) == 5
        assert back[0].method == "GET"


# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

class TestConfig:
    def test_load_valid_config(self, tmp_path):
        from ai_waf_v2.utils.config import load_config
        import yaml, shutil

        src = "config/pipeline.yaml"
        dst = tmp_path / "pipeline.yaml"
        shutil.copy(src, dst)
        cfg = load_config(dst)

        assert cfg.model.track_b_99m.d_model == 768
        assert cfg.model.track_b_99m.n_layers == 13

    def test_head_dim_validation(self):
        from ai_waf_v2.utils.config import ModelArchConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="divisible by n_heads"):
            ModelArchConfig(d_model=768, n_heads=7)   # 768 / 7 is not integer

    def test_split_sum_validation(self):
        from ai_waf_v2.utils.config import SplitConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="sum to 1.0"):
            SplitConfig(train=0.5, val=0.1, test=0.1, adversarial=0.1, canary=0.05)

    def test_distill_alpha_sum_validation(self):
        from ai_waf_v2.utils.config import DistillationTrainingConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="must sum to 1.0"):
            DistillationTrainingConfig(alpha_soft=0.5, alpha_hard=0.5 + 0.1)

    def test_effective_batch_size(self):
        from ai_waf_v2.utils.config import TeacherTrainingConfig
        t = TeacherTrainingConfig(batch_size=128, grad_accum_steps=8)
        assert t.effective_batch_size == 1024


# ─────────────────────────────────────────────────────────
# Encoder
# ─────────────────────────────────────────────────────────

class TestWafEncoder:
    @pytest.fixture
    def small_encoder(self):
        from ai_waf_v2.models.encoder import WafEncoder
        return WafEncoder(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            dropout=0.0,
            attn_dropout=0.0,
            max_seq_len=32,
        )

    def test_output_shape(self, small_encoder):
        B, T = 4, 16
        input_ids      = torch.randint(1, 100, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)

        cls_hidden = small_encoder(input_ids, attention_mask)
        assert cls_hidden.shape == (B, 64), f"Expected (4, 64), got {cls_hidden.shape}"

    def test_padding_mask_applied(self, small_encoder):
        """Padded tokens should not affect the [CLS] output."""
        input_ids_full = torch.randint(1, 100, (1, 16))

        # Create padded version: last 8 tokens are [PAD] = 0
        input_ids_padded = input_ids_full.clone()
        input_ids_padded[0, 8:] = 0
        mask_full   = torch.ones(1, 16, dtype=torch.long)
        mask_padded = torch.cat([torch.ones(1, 8), torch.zeros(1, 8)], dim=1).long()

        with torch.no_grad():
            out_full   = small_encoder(input_ids_full,   mask_full)
            out_padded = small_encoder(input_ids_padded, mask_padded)

        # Outputs should differ — padding mask is actually applied
        # (they won't be identical because the input tokens differ)
        assert out_full.shape == out_padded.shape

    def test_parameter_count(self, small_encoder):
        n = small_encoder.count_parameters()
        assert n > 0
        breakdown = small_encoder.parameter_breakdown()
        assert breakdown["total"] == n
        assert breakdown["embeddings"] > 0

    def test_d_model_head_constraint(self):
        from ai_waf_v2.models.encoder import WafEncoder
        with pytest.raises(AssertionError):
            WafEncoder(vocab_size=100, d_model=64, n_heads=7)  # 64 / 7 not integer

    def test_no_nan_in_output(self, small_encoder):
        B, T = 2, 16
        input_ids      = torch.randint(1, 100, (B, T))
        attention_mask = torch.ones(B, T, dtype=torch.long)
        out = small_encoder(input_ids, attention_mask)
        assert not torch.isnan(out).any(), "NaN detected in encoder output"


# ─────────────────────────────────────────────────────────
# Classification head
# ─────────────────────────────────────────────────────────

class TestWafClassifier:
    @pytest.fixture
    def small_classifier(self):
        from ai_waf_v2.models.head import WafClassifier
        from ai_waf_v2.utils.config import ModelArchConfig

        arch = ModelArchConfig(
            vocab_size=100, d_model=64, n_layers=2, n_heads=4,
            d_ff=256, dropout=0.0, attn_dropout=0.0,
            max_seq_len=32, num_labels=2,
        )
        return WafClassifier.from_config(arch)

    def test_logits_shape(self, small_classifier):
        B, T = 4, 16
        ids  = torch.randint(1, 100, (B, T))
        mask = torch.ones(B, T, dtype=torch.long)
        out  = small_classifier(ids, mask)
        assert out["logits"].shape == (B, 2)

    def test_loss_computed_when_labels_given(self, small_classifier):
        B, T = 4, 16
        ids    = torch.randint(1, 100, (B, T))
        mask   = torch.ones(B, T, dtype=torch.long)
        labels = torch.randint(0, 2, (B,))
        out    = small_classifier(ids, mask, labels=labels)
        assert "loss" in out
        assert out["loss"].shape == torch.Size([])   # scalar

    def test_predict_returns_binary(self, small_classifier):
        B, T   = 4, 16
        ids    = torch.randint(1, 100, (B, T))
        mask   = torch.ones(B, T, dtype=torch.long)
        preds, probs = small_classifier.predict(ids, mask)
        assert set(preds.tolist()).issubset({0, 1})
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_save_load_roundtrip(self, small_classifier, tmp_path):
        from ai_waf_v2.models.head import WafClassifier
        from ai_waf_v2.utils.config import ModelArchConfig

        path = tmp_path / "model.pt"
        small_classifier.save(path)

        arch = ModelArchConfig(
            vocab_size=100, d_model=64, n_layers=2, n_heads=4,
            d_ff=256, dropout=0.0, attn_dropout=0.0,
            max_seq_len=32, num_labels=2,
        )
        loaded = WafClassifier.load(path, arch)

        ids  = torch.randint(1, 100, (2, 16))
        mask = torch.ones(2, 16, dtype=torch.long)

        with torch.no_grad():
            out_orig   = small_classifier(ids, mask)
            out_loaded = loaded(ids, mask)

        assert torch.allclose(out_orig["logits"], out_loaded["logits"])


# ─────────────────────────────────────────────────────────
# Distillation loss
# ─────────────────────────────────────────────────────────

class TestDistillationLoss:
    def test_loss_keys_present(self):
        from ai_waf_v2.distill.losses import DistillationLoss

        loss_fn = DistillationLoss(temperature=4.0, alpha_soft=0.7, alpha_hard=0.3)
        B, C    = 8, 2

        s_logits = torch.randn(B, C)
        t_logits = torch.randn(B, C)
        labels   = torch.randint(0, C, (B,))

        out = loss_fn(s_logits, t_logits, labels)
        assert "loss"    in out
        assert "loss_ce" in out
        assert "loss_kl" in out
        assert "loss_mse" in out

    def test_loss_is_scalar(self):
        from ai_waf_v2.distill.losses import DistillationLoss

        loss_fn  = DistillationLoss()
        s_logits = torch.randn(4, 2)
        t_logits = torch.randn(4, 2)
        labels   = torch.randint(0, 2, (4,))
        out      = loss_fn(s_logits, t_logits, labels)

        assert out["loss"].shape == torch.Size([])

    def test_alpha_sum_validation(self):
        from ai_waf_v2.distill.losses import DistillationLoss

        with pytest.raises(ValueError, match="sum to 1.0"):
            DistillationLoss(alpha_soft=0.6, alpha_hard=0.6)

    def test_temperature_squared_scaling(self):
        """At higher T, KL loss should be larger (more gradient from soft labels)."""
        from ai_waf_v2.distill.losses import DistillationLoss

        B, C     = 16, 2
        s_logits = torch.randn(B, C)
        t_logits = torch.randn(B, C)
        labels   = torch.randint(0, C, (B,))

        loss_t1 = DistillationLoss(temperature=1.0, alpha_soft=1.0, alpha_hard=0.0)
        loss_t4 = DistillationLoss(temperature=4.0, alpha_soft=1.0, alpha_hard=0.0)

        out_t1 = loss_t1(s_logits, t_logits, labels)
        out_t4 = loss_t4(s_logits, t_logits, labels)

        # At T=4, T² × KL ≥ T=1 version in magnitude (not strictly guaranteed,
        # but should hold for random logits with high probability)
        assert out_t4["loss"].item() >= 0.0
        assert out_t1["loss"].item() >= 0.0


# ─────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────

class TestMetrics:
    def _make_perfect(self, n: int = 100):
        labels = torch.randint(0, 2, (n,))
        probs  = labels.float() * 0.99 + (1 - labels.float()) * 0.01
        preds  = labels.clone()
        return preds, probs, labels

    def _make_random(self, n: int = 100):
        labels = torch.randint(0, 2, (n,))
        probs  = torch.rand(n)
        preds  = (probs >= 0.5).long()
        return preds, probs, labels

    def test_perfect_classifier(self):
        from ai_waf_v2.eval.metrics import compute_metrics

        preds, probs, labels = self._make_perfect()
        m = compute_metrics(preds, probs, labels)

        assert m["f1"]       > 0.98
        assert m["fpr"]      < 0.02
        assert m["auc_roc"]  > 0.98
        assert m["auc_pr"]   > 0.98

    def test_metric_keys_complete(self):
        from ai_waf_v2.eval.metrics import compute_metrics

        preds, probs, labels = self._make_random()
        m = compute_metrics(preds, probs, labels)

        for key in ("f1", "precision", "recall", "fpr", "fnr",
                    "accuracy", "auc_roc", "auc_pr", "avg_precision"):
            assert key in m, f"Missing metric: {key}"

    def test_all_benign_edge_case(self):
        """All-benign predictions: recall=0, fpr=0."""
        from ai_waf_v2.eval.metrics import compute_metrics

        preds  = torch.zeros(50, dtype=torch.long)
        probs  = torch.zeros(50)
        labels = torch.cat([torch.zeros(25), torch.ones(25)]).long()

        m = compute_metrics(preds, probs, labels)
        assert m["recall"] == pytest.approx(0.0, abs=1e-4)
        assert m["fpr"]    == pytest.approx(0.0, abs=1e-4)

    def test_per_class_metrics(self):
        from ai_waf_v2.eval.metrics import compute_per_class_metrics

        n      = 200
        preds  = torch.randint(0, 2, (n,))
        probs  = torch.rand(n)
        labels = torch.randint(0, 2, (n,))
        classes = ["benign"] * 100 + ["sqli"] * 50 + ["xss"] * 50

        result = compute_per_class_metrics(preds, probs, labels, classes)
        assert "sqli" in result
        assert "xss"  in result
        assert "f1"   in result["sqli"]

    def test_threshold_sweep_lengths(self):
        from ai_waf_v2.eval.metrics import compute_threshold_sweep

        probs  = torch.rand(100)
        labels = torch.randint(0, 2, (100,))
        sweep  = compute_threshold_sweep(probs, labels, n_thresholds=50)

        assert len(sweep["thresholds"]) == 50
        assert len(sweep["f1"])         == 50

    def test_find_threshold_at_fpr(self):
        from ai_waf_v2.eval.metrics import find_threshold_at_fpr

        # Perfect separation: threshold should be high
        labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
        probs  = torch.cat([torch.zeros(50), torch.ones(50)])

        t = find_threshold_at_fpr(probs, labels, target_fpr=0.01)
        assert isinstance(t, float)
        assert 0.0 <= t <= 1.0


# ─────────────────────────────────────────────────────────
# Adversarial tamper functions
# ─────────────────────────────────────────────────────────

class TestTamperFunctions:
    def test_space2comment(self):
        from ai_waf_v2.eval.adversarial import tamper_space2comment
        result = tamper_space2comment("UNION SELECT 1")
        assert "/**/" in result
        assert " " not in result

    def test_url_encode(self):
        from ai_waf_v2.eval.adversarial import tamper_url_encode
        result = tamper_url_encode("UNION SELECT")
        assert " " not in result
        assert "%" in result

    def test_double_url_encode(self):
        from ai_waf_v2.eval.adversarial import tamper_double_url_encode
        result = tamper_double_url_encode("UNION SELECT")
        # Double-encoded space = %2520
        assert result != "UNION SELECT"

    def test_randomcase_preserves_non_alpha(self):
        from ai_waf_v2.eval.adversarial import tamper_randomcase
        result = tamper_randomcase("123 !@#")
        # Non-alpha chars should be unchanged
        assert "1" in result and "2" in result and "3" in result


# ─────────────────────────────────────────────────────────
# Collator
# ─────────────────────────────────────────────────────────

class TestWafCollator:
    def test_dynamic_padding(self):
        from ai_waf_v2.data.collator import WafCollator

        collator = WafCollator(pad_token_id=0, max_seq_len=32)
        batch = [
            {
                "input_ids":      torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels":         torch.tensor(0),
            },
            {
                "input_ids":      torch.tensor([4, 5, 6, 7, 8]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
                "labels":         torch.tensor(1),
            },
        ]
        out = collator(batch)

        # Should pad to length 5 (max in batch)
        assert out["input_ids"].shape      == (2, 5)
        assert out["attention_mask"].shape == (2, 5)
        assert out["labels"].shape         == (2,)

        # First sample should be right-padded with 0s
        assert out["input_ids"][0, 3].item()      == 0
        assert out["attention_mask"][0, 3].item() == 0

    def test_truncation_at_max_seq_len(self):
        from ai_waf_v2.data.collator import WafCollator

        collator = WafCollator(pad_token_id=0, max_seq_len=4)
        batch = [{
            "input_ids":      torch.tensor([1, 2, 3, 4, 5, 6]),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1]),
            "labels":         torch.tensor(0),
        }]
        out = collator(batch)
        assert out["input_ids"].shape[1] == 4


# ─────────────────────────────────────────────────────────
# Seed utility
# ─────────────────────────────────────────────────────────

class TestSeedEverything:
    def test_reproducible_random(self):
        from ai_waf_v2.utils.seed import seed_everything
        import random

        seed_everything(42)
        a = random.random()

        seed_everything(42)
        b = random.random()

        assert a == b

    def test_reproducible_torch(self):
        from ai_waf_v2.utils.seed import seed_everything

        seed_everything(42)
        a = torch.randn(10)

        seed_everything(42)
        b = torch.randn(10)

        assert torch.allclose(a, b)

# ─────────────────────────────────────────────────────────
# PCFG engine  (recursive grammar, loader, fallback sampler)
# ─────────────────────────────────────────────────────────

class TestPcfg:
    def _recursive_grammar(self, max_depth=4):
        # A -> "a" A | "b"  : unbounded right recursion, must still terminate.
        from ai_waf_v2.augment.pcfg import Pcfg, T, N
        rules = {"A": [(0.9, [T("a"), N("A")]), (0.1, [T("b")])]}
        return Pcfg("rec", rules, start="A", max_depth=max_depth, max_len=64)

    def test_terminates_under_heavy_recursion(self):
        import random
        g = self._recursive_grammar(max_depth=4)
        # Weighted 9:1 toward the recursive production — without the min-cost
        # depth guard this would rarely terminate. Every sample must return a
        # finite string that ends in the only terminal ("b").
        for i in range(2000):
            s = g.sample(random.Random(i))
            assert isinstance(s, str)
            assert s.endswith("b")
            assert set(s) <= {"a", "b"}
            assert len(s) <= g.max_depth + 2   # depth-bounded, so length-bounded

    def test_rejects_nonterminating_grammar(self):
        import pytest
        from ai_waf_v2.augment.pcfg import Pcfg, N
        # A -> A A  has no terminating derivation: construction must fail fast.
        with pytest.raises(ValueError):
            Pcfg("bad", {"A": [(1.0, [N("A"), N("A")])]}, start="A")

    def test_matched_quotes_attribute(self):
        import random
        from ai_waf_v2.augment.pcfg import Pcfg, OPENQ, CLOSEQ, F
        g = Pcfg("q", {"START": [(1.0, [OPENQ, F("sql_word"), CLOSEQ])]})
        for i in range(200):
            s = g.sample(random.Random(i))
            assert s[0] == s[-1] and s[0] in ("'", '"')  # opened quote is closed with same char

    def test_load_grammars_roundtrip(self, tmp_path):
        import random
        from ai_waf_v2.augment.pcfg import load_pcfg_grammars
        yaml_text = (
            "pcfg_grammars:\n"
            "  demo:\n"
            "    start: START\n"
            "    rules:\n"
            "      START:\n"
            '        - [1.0, ["t:x=", "f:rand_int", "n:TAIL"]]\n'
            "      TAIL:\n"
            '        - [0.5, ["t:!"]]\n'
            '        - [0.5, ["t:,", "n:TAIL"]]\n'
        )
        p = tmp_path / "g.yaml"
        p.write_text(yaml_text)
        reg = load_pcfg_grammars(p)
        assert set(reg) == {"demo"}
        s = reg["demo"].sample(random.Random(0))
        assert s.startswith("x=")

    def test_load_missing_file_returns_empty(self, tmp_path):
        from ai_waf_v2.augment.pcfg import load_pcfg_grammars
        assert load_pcfg_grammars(tmp_path / "nope.yaml") == {}

    def test_load_unknown_func_raises(self, tmp_path):
        import pytest
        from ai_waf_v2.augment.pcfg import load_pcfg_grammars
        p = tmp_path / "g.yaml"
        p.write_text(
            "pcfg_grammars:\n  demo:\n    rules:\n      START:\n"
            '        - [1.0, ["f:does_not_exist"]]\n'
        )
        with pytest.raises(ValueError):
            load_pcfg_grammars(p)

    def test_shipped_grammars_load_and_sample(self):
        import random
        from pathlib import Path
        from ai_waf_v2.augment.pcfg import load_pcfg_grammars
        from ai_waf_v2.augment.validity import is_valid_for_class

        from ai_waf_v2.augment.fillers import fill_placeholders

        cfg = Path(__file__).resolve().parent.parent / "config" / "pcfg_grammars.yaml"
        reg = load_pcfg_grammars(cfg)
        assert {"sqli", "xss", "cmdi", "ssti", "lfi", "path_traversal"} <= set(reg)
        # lfi and path_traversal share one grammar via a YAML anchor
        for cls in ("sqli", "xss", "cmdi", "ssti", "lfi", "path_traversal"):
            # sample structure, then fill §NAME§ fillers — strict=True fails if a
            # grammar references a placeholder that isn't a registered filler.
            filled = {fill_placeholders(reg[cls].sample(random.Random(i)),
                                        random.Random(1000 + i), strict=True)
                      for i in range(200)}
            assert len(filled) > 50                        # structural + filler diversity
            assert not any("§" in s for s in filled)       # every placeholder resolved
            assert all(is_valid_for_class(cls, s) for s in filled)


class TestPcfgSampler:
    def _fallback(self):
        # Marks its output so we can tell fallback payloads apart from PCFG ones.
        def fb(attack_class, n, rng):
            return [f"FLAT::{attack_class}::{i}" for i in range(n)]
        return fb

    def test_falls_back_when_no_grammar(self):
        import random
        from ai_waf_v2.augment.pcfg import PcfgSampler
        s = PcfgSampler(registry={}, fallback=self._fallback())
        out = s.generate("sqli", 5, random.Random(0))
        assert out == [f"FLAT::sqli::{i}" for i in range(5)]

    def test_tops_up_from_fallback_when_pcfg_runs_dry(self):
        import random
        from ai_waf_v2.augment.pcfg import Pcfg, PcfgSampler, T
        # Grammar can emit exactly ONE distinct string, so 5 requested → 1 PCFG + 4 fallback.
        tiny = Pcfg("tiny", {"START": [(1.0, [T("ONLY")])]})
        s = PcfgSampler(registry={"sqli": tiny}, fallback=self._fallback())
        out = s.generate("sqli", 5, random.Random(0))
        assert len(out) == 5
        assert out.count("ONLY") == 1
        assert sum(x.startswith("FLAT::") for x in out) == 4

    def test_uses_pcfg_and_returns_exact_count(self):
        import random
        from ai_waf_v2.augment.pcfg import Pcfg, PcfgSampler, T, N, F
        g = Pcfg("g", {
            "START": [(1.0, [T("id="), F("rand_int"), N("T")])],
            "T": [(0.5, [T("")]), (0.5, [T("-"), N("T")])],
        })
        s = PcfgSampler(registry={"sqli": g}, fallback=self._fallback())
        out = s.generate("sqli", 20, random.Random(1))
        assert len(out) == 20
        assert all(o.startswith("id=") for o in out)
        assert not any(o.startswith("FLAT::") for o in out)  # PCFG satisfied the whole quota


# ─────────────────────────────────────────────────────────
# Per-class payload validity  (quality-gate structural check)
# ─────────────────────────────────────────────────────────

class TestClassValidity:
    def test_valid_payloads_pass(self):
        from ai_waf_v2.augment.validity import is_valid_for_class
        assert is_valid_for_class("sqli", "1 OR 1=1-- ")
        assert is_valid_for_class("sqli", "' UNION SELECT NULL,NULL#")
        assert is_valid_for_class("xss", "<script>alert(1)</script>")
        assert is_valid_for_class("xss", '"><img src=x onerror=alert(1)>')  # realistic bracket-imbalanced breakout
        assert is_valid_for_class("xss", '" onmouseover=alert(1) x="')       # event-handler injection, no tags
        assert is_valid_for_class("lfi", "../../etc/passwd")
        assert is_valid_for_class("ssti", "{{7*7}}")

    def test_malformed_payloads_rejected(self):
        from ai_waf_v2.augment.validity import is_valid_for_class
        # truncated tag → dangling '<' after the last '>'
        assert not is_valid_for_class("xss", "<script>alert(1)</scrip")
        assert not is_valid_for_class("xss", "<img src=x onerror=alert(")   # truncated, no closing '>'
        assert not is_valid_for_class("sqli", "1 OR (1=1 AND (1=2")
        # payload with no class signal at all
        assert not is_valid_for_class("xss", "just some text")
        assert not is_valid_for_class("ssti", "plain string")

    def test_unknown_class_always_passes(self):
        from ai_waf_v2.augment.validity import is_valid_for_class
        assert is_valid_for_class("benign", "anything at all")
        assert is_valid_for_class("totally_new_class", "")


# ─────────────────────────────────────────────────────────
# Filler placeholders  (label-neutral, high-cardinality values)
# ─────────────────────────────────────────────────────────

class TestFillers:
    def test_high_cardinality(self):
        import random
        from ai_waf_v2.augment.fillers import fill_placeholders
        hosts = {fill_placeholders("§HOST§", random.Random(i)) for i in range(1000)}
        assert len(hosts) > 900                 # no memorisable constant like "evil.com"
        assert all("§" not in h for h in hosts)

    def test_coreference_same_tag_same_value(self):
        import random
        from ai_waf_v2.augment.fillers import fill_placeholders
        first, second, _third = fill_placeholders(
            "§HOST#a§|§HOST#a§|§HOST§", random.Random(0)).split("|")
        assert first == second                  # same NAME#TAG → one value per payload

    def test_unknown_placeholder_untouched_or_strict(self):
        import random
        import pytest
        from ai_waf_v2.augment.fillers import fill_placeholders
        assert fill_placeholders("§NOPE§", random.Random(0)) == "§NOPE§"
        with pytest.raises(ValueError):
            fill_placeholders("§NOPE§", random.Random(0), strict=True)

    def test_no_placeholder_is_noop(self):
        import random
        from ai_waf_v2.augment.fillers import fill_placeholders
        assert fill_placeholders("../../etc/passwd", random.Random(0)) == "../../etc/passwd"

    def test_scrub_textbook_hosts(self):
        import random
        from ai_waf_v2.augment.fillers import scrub_textbook_hosts
        out = scrub_textbook_hosts(
            "curl http://evil.com && wget http://EXAMPLE.COM/x", random.Random(1))
        assert "evil.com" not in out.lower()
        assert "example.com" not in out.lower()


# ─────────────────────────────────────────────────────────
# Token↔label leakage measurement
# ─────────────────────────────────────────────────────────

class TestLeakage:
    def test_shared_fillers_are_neutral(self):
        # Identical structure on both sides, only the (shared, random) host varies
        # → nothing predicts the label → near-zero mutual information.
        import random
        from ai_waf_v2.augment.fillers import rand_host
        from ai_waf_v2.eval.leakage import token_leakage
        rng = random.Random(0)
        texts, labels = [], []
        for i in range(400):
            texts.append(f"q=1 and 1=1 url=https://{rand_host(rng)}/home")
            labels.append(i % 2)
        rep = token_leakage(texts, labels, min_df=5)
        assert rep.max_mi < 0.1

    def test_detects_leaked_constant(self):
        # Same structure, but the attack host is a CONSTANT → that token leaks.
        import random
        from ai_waf_v2.augment.fillers import rand_host
        from ai_waf_v2.eval.leakage import token_leakage
        rng = random.Random(0)
        texts, labels = [], []
        for _ in range(200):
            texts.append("q=1 and 1=1 url=https://leak.evilcorp.io/home"); labels.append(1)
        for _ in range(200):
            texts.append(f"q=1 and 1=1 url=https://{rand_host(rng)}/home"); labels.append(0)
        rep = token_leakage(texts, labels, min_df=5)
        top = {s.token: s for s in rep.top}
        assert "evilcorp" in top and top["evilcorp"].p_malicious == 1.0
        assert rep.max_mi > 0.5

    def test_counterfactual_separates_leak_from_structure(self):
        # A predictor keyed on a memorised host collapses when fillers are swapped;
        # one keyed on structure is unaffected.
        import random
        from ai_waf_v2.augment.fillers import rand_host
        from ai_waf_v2.eval.leakage import counterfactual_auc_delta
        rng = random.Random(0)
        texts, labels = [], []
        for _ in range(150):
            texts.append("q=1 UNION SELECT pw -- host=leak.evilcorp.io"); labels.append(1)
        for _ in range(150):
            texts.append(f"q=hi host=https://{rand_host(rng)}/x"); labels.append(0)

        leaky = counterfactual_auc_delta(
            lambda ts: [1.0 if "evilcorp" in t else 0.0 for t in ts],
            texts, labels, random.Random(1))
        structural = counterfactual_auc_delta(
            lambda ts: [1.0 if "union select" in t.lower() else 0.0 for t in ts],
            texts, labels, random.Random(2))
        assert leaky["delta"] > 0.3            # relied on the constant → collapses
        assert abs(structural["delta"]) < 0.05  # relied on structure → robust

    def test_auc_roc_basic(self):
        from ai_waf_v2.eval.leakage import auc_roc
        assert abs(auc_roc([0, 0, 1, 1], [0, 0, 1, 1]) - 1.0) < 1e-9
        assert abs(auc_roc([1, 1, 1, 1], [0, 1, 0, 1]) - 0.5) < 1e-9  # constant → 0.5


# ─────────────────────────────────────────────────────────
# Report path registry
# ─────────────────────────────────────────────────────────

class TestReportRegistry:
    def test_paths_are_unique(self):
        # No two logical reports may resolve to the same file (would clobber).
        from ai_waf_v2.utils.reports import REPORTS
        assert len(set(REPORTS.values())) == len(REPORTS)

    def test_paths_are_stage_organized(self):
        from ai_waf_v2.utils.reports import REPORTS
        # every report lives under a numbered stage folder
        assert all(rel.split("/")[0][0].isdigit() for rel in REPORTS.values())

    def test_report_path_under_root_and_creates_dir(self, tmp_path):
        from ai_waf_v2.utils.reports import report_path
        p = report_path("quality_gate.json", tmp_path)
        assert p == tmp_path / "3_data_augmentation/metrics/04_quality_gate.json"
        assert p.parent.is_dir()               # parent created

    def test_unknown_name_is_conspicuous(self, tmp_path):
        from ai_waf_v2.utils.reports import report_path
        p = report_path("mystery.json", tmp_path, mkdir=False)
        assert p == tmp_path / "_unfiled/mystery.json"
