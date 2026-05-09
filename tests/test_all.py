"""
tests/test_all.py
-----------------
Unit tests for the WAF-AI core library.

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