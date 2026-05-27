"""
ai_waf_v2.utils.config
------------------
Load and validate config/pipeline.yaml using Pydantic v2 models.
All stage scripts obtain settings via:

    from ai_waf_v2.utils.config import load_config
    cfg = load_config("config/pipeline.yaml")
    print(cfg.model.track_b_99m.d_model)   # 768
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, model_validator, ConfigDict


# ─────────────────────────────────────────────────────────
# Sub-models (match config/pipeline.yaml structure exactly)
# ─────────────────────────────────────────────────────────

class ProjectConfig(BaseModel):
    name:    str = "ai-waf-v2"
    version: str = "0.1.0"
    seed:    int = 42


class PathsConfig(BaseModel):
    data_raw:        Path = Path("data/raw")
    data_normalized: Path = Path("data/normalized")
    data_augmented:  Path = Path("data/augmented")
    data_filtered:   Path = Path("data/filtered")
    data_splits:     Path = Path("data/splits")
    models:          Path = Path("models")
    tokenizers:      Path = Path("tokenizers")
    reports:         Path = Path("reports")
    mlruns:          Path = Path("mlruns")


class SloConfig(BaseModel):
    latency_inline_p99_ms:  float = 5.0
    latency_offline_p99_ms: float = 50.0
    throughput_min_rps:     int   = 10_000
    max_false_positive_rate: float = 0.001


class DatasetEntry(BaseModel):
    # Forbid unknown fields — if YAML has a key not declared here,
    # Pydantic raises ValidationError immediately instead of silently dropping it.
    model_config = ConfigDict(extra="forbid")

    # ── Required ──────────────────────────────────────────
    name:         str

    # ── Acquisition ───────────────────────────────────────
    converter_id:         Optional[str]       = None
    kaggle_handle:        Optional[str]       = None
    url:                  str                 = ""
    mirrors:              list[str]           = Field(default_factory=list)
    archive_sha256:       Optional[str]       = None
    manual_instructions:  Optional[str]       = None

    # ── Metadata ──────────────────────────────────────────
    description:  Optional[str] = None
    license:      Optional[str] = None
    citation:     Optional[str] = None

    # ── Output ────────────────────────────────────────────
    output_filename: Optional[str] = None
    label_col:       str           = "label"


class SplitConfig(BaseModel):
    train:       float = 0.70
    val:         float = 0.15
    test:        float = 0.10
    adversarial: float = 0.03
    canary:      float = 0.02
    stratify_by: list[str] = Field(default_factory=lambda: ["label", "attack_class"])

    @model_validator(mode="after")
    def _check_sum(self) -> "SplitConfig":
        total = self.train + self.val + self.test + self.adversarial + self.canary
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
        return self


class DedupConfig(BaseModel):
    minhash_threshold:  float = 0.85
    semantic_threshold: float = 0.90
    exact_hash:         bool  = True


class DataSchemaConfig(BaseModel):
    label_map:      dict[str, int] = Field(default_factory=lambda: {"benign": 0, "malicious": 1})
    attack_classes: list[str] = Field(default_factory=list)


class DataConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True) 
    
    datasets:    list[DatasetEntry] = Field(default_factory=list) 
    data_schema: DataSchemaConfig   = Field(default_factory=DataSchemaConfig, alias="schema")  
    split:       SplitConfig        = Field(default_factory=SplitConfig) 
    dedup:       DedupConfig        = Field(default_factory=DedupConfig)  



class AugRulesConfig(BaseModel):
    mutations_per_sample: int       = 20
    encodings:            list[str] = Field(default_factory=list)


class AugGrammarConfig(BaseModel):
    targets:            list[str] = Field(default_factory=list)
    samples_per_class:  int       = 5000


class AugLocalLLMConfig(BaseModel):
    model_path:             str   = ""
    model_type:             str   = "causal"
    batch_size:             int   = 8
    max_new_tokens:         int   = 256
    temperature:            float = 0.9
    samples_per_gap_class:  int   = 1000


class AugApiLLMConfig(BaseModel):
    provider:       str = "anthropic"
    model:          str = "claude-sonnet-4-20250514"
    max_tokens:     int = 512
    samples_benign: int = 10_000
    samples_edge_case: int = 2000


class AugBenignConfig(BaseModel):
    rest_samples:   int  = 20_000
    replay_enabled: bool = False
    replay_path:    str  = ""


class AugFilterConfig(BaseModel):
    max_unk_ratio:            float = 0.15
    min_token_length:         int   = 4
    max_token_length:         int   = 256
    label_consistency_check:  bool  = True


class AugmentationConfig(BaseModel):
    rules:      AugRulesConfig    = Field(default_factory=AugRulesConfig)
    grammar:    AugGrammarConfig  = Field(default_factory=AugGrammarConfig)
    local_llm:  AugLocalLLMConfig = Field(default_factory=AugLocalLLMConfig)
    api_llm:    AugApiLLMConfig   = Field(default_factory=AugApiLLMConfig)
    benign:     AugBenignConfig   = Field(default_factory=AugBenignConfig)
    filtering:  AugFilterConfig   = Field(default_factory=AugFilterConfig)


class TokenizerTrackAConfig(BaseModel):
    base_model:  str       = "bert-base-uncased"
    output_dir:  Path      = Path("tokenizers/track_a")
    http_tokens: list[str] = Field(default_factory=list)


class TokenizerTrackBConfig(BaseModel):
    vocab_size:          int   = 8000
    character_coverage:  float = 0.9999
    model_type:          str   = "bpe"
    output_dir:          Path  = Path("tokenizers/track_b")
    pad_token:           str   = "[PAD]"
    unk_token:           str   = "[UNK]"
    cls_token:           str   = "[CLS]"
    sep_token:           str   = "[SEP]"
    mask_token:          str   = "[MASK]"


class TokenizerConfig(BaseModel):
    seq_len: int = 256
    track_a: TokenizerTrackAConfig = Field(default_factory=TokenizerTrackAConfig)
    track_b: TokenizerTrackBConfig = Field(default_factory=TokenizerTrackBConfig)


class ModelArchConfig(BaseModel):
    d_model:     int   = 768
    n_layers:    int   = 13
    n_heads:     int   = 12
    d_ff:        int   = 3072
    dropout:     float = 0.1
    attn_dropout: float = 0.1
    max_seq_len: int   = 256
    vocab_size:  int   = 8000
    num_labels:  int   = 2
    pad_token_id: int  = 0
    output_dir:  Path  = Path("models/track_b/99m")

    @model_validator(mode="after")
    def _check_head_dim(self) -> "ModelArchConfig":
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads}). "
                f"head_dim would be {self.d_model / self.n_heads:.1f}"
            )
        return self


class StudentModelConfig(ModelArchConfig):
    d_model:     int  = 256
    n_layers:    int  = 6
    n_heads:     int  = 4
    d_ff:        int  = 1024
    output_dir:  Path = Path("models/student")
    quantization: str = "int8"


class TrackAModelConfig(BaseModel):
    base_model:  str  = "microsoft/deberta-v3-base"
    num_labels:  int  = 2
    dropout:     float = 0.1
    output_dir:  Path = Path("models/track_a/large")


class ModelConfig(BaseModel):
    track_a_large: TrackAModelConfig = Field(default_factory=TrackAModelConfig)
    track_a_small: TrackAModelConfig = Field(
        default_factory=lambda: TrackAModelConfig(
            base_model="prajjwal1/bert-tiny",
            output_dir=Path("models/track_a/small"),
        )
    )
    track_b_99m: ModelArchConfig     = Field(default_factory=ModelArchConfig)
    student:     StudentModelConfig  = Field(default_factory=StudentModelConfig)


class TeacherTrainingConfig(BaseModel):
    batch_size:               int   = 256
    grad_accum_steps:         int   = 4
    max_steps:                int   = 50_000
    warmup_ratio:             float = 0.06
    peak_lr:                  float = 1e-4
    min_lr_ratio:             float = 0.1
    weight_decay:             float = 0.01
    adam_beta1:               float = 0.9
    adam_beta2:               float = 0.999
    adam_eps:                 float = 1e-8
    grad_clip_norm:           float = 1.0
    label_smoothing:          float = 0.1
    precision:                str   = "bf16"
    eval_every_steps:         int   = 500
    save_every_steps:         int   = 1000
    early_stopping_patience:  int   = 10
    early_stopping_metric:    str   = "auc_pr"
    operating_fpr_target:     float = 0.001

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.grad_accum_steps

    @property
    def warmup_steps(self) -> int:
        return int(self.max_steps * self.warmup_ratio)

    @property
    def min_lr(self) -> float:
        return self.peak_lr * self.min_lr_ratio


class DistillationTrainingConfig(BaseModel):
    batch_size:         int   = 128
    grad_accum_steps:   int   = 8
    max_steps:          int   = 40_000
    warmup_ratio:       float = 0.06
    peak_lr:            float = 2e-4
    weight_decay:       float = 0.01
    grad_clip_norm:     float = 1.0
    precision:          str   = "bf16"
    alpha_soft:         float = 0.7
    alpha_hard:         float = 0.3
    temperature:        float = 4.0
    hidden_mse_weight:  float = 0.0

    @model_validator(mode="after")
    def _check_alpha_sum(self) -> "DistillationTrainingConfig":
        if abs(self.alpha_soft + self.alpha_hard - 1.0) > 1e-4:
            raise ValueError(
                f"alpha_soft + alpha_hard must equal 1.0, "
                f"got {self.alpha_soft + self.alpha_hard}"
            )
        return self


class TrainingConfig(BaseModel):
    teacher:      TeacherTrainingConfig      = Field(default_factory=TeacherTrainingConfig)
    distillation: DistillationTrainingConfig = Field(default_factory=DistillationTrainingConfig)


class AdversarialConfig(BaseModel):
    tamper_scripts:   list[str] = Field(default_factory=list)
    evasion_wordlists: list[str] = Field(default_factory=list)


class AblationConfig(BaseModel):
    tokenizer:       bool = True
    augmentation:    bool = True
    model_size:      bool = True
    label_smoothing: bool = True


class EvaluationConfig(BaseModel):
    batch_sizes:       list[int] = Field(default_factory=lambda: [1, 8, 32, 64, 256])
    devices:           list[str] = Field(default_factory=lambda: ["gpu", "cpu"])
    n_latency_warmup:  int       = 50
    n_latency_runs:    int       = 500
    metrics:           list[str] = Field(default_factory=list)
    adversarial:       AdversarialConfig = Field(default_factory=AdversarialConfig)
    ablation:          AblationConfig    = Field(default_factory=AblationConfig)


class MLflowConfig(BaseModel):
    tracking_uri:    str            = "sqlite:///mlruns/mlflow.db"
    experiment_name: str            = "waf-ai"
    tags:            dict[str, str] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────
# Root config model
# ─────────────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    project:       ProjectConfig     = Field(default_factory=ProjectConfig)
    paths:         PathsConfig       = Field(default_factory=PathsConfig)
    slo:           SloConfig         = Field(default_factory=SloConfig)
    data:          DataConfig        = Field(default_factory=DataConfig)
    augmentation:  AugmentationConfig = Field(default_factory=AugmentationConfig)
    tokenizer:     TokenizerConfig   = Field(default_factory=TokenizerConfig)
    model:         ModelConfig       = Field(default_factory=ModelConfig)
    training:      TrainingConfig    = Field(default_factory=TrainingConfig)
    evaluation:    EvaluationConfig  = Field(default_factory=EvaluationConfig)
    mlflow:        MLflowConfig      = Field(default_factory=MLflowConfig)


# ─────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────

def _expand_env(obj: Any) -> Any:
    """Recursively expand ${ENV_VAR} strings in config values."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(item) for item in obj]
    return obj


def load_config(path: str | Path = "config/pipeline.yaml") -> PipelineConfig:
    """
    Load, env-expand, and validate the pipeline config.

    Parameters
    ----------
    path : str | Path
        Path to the YAML config file.

    Returns
    -------
    PipelineConfig
        Fully validated config object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    pydantic.ValidationError
        If any field fails validation (e.g. split ratios don't sum to 1,
        d_model not divisible by n_heads).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with path.open() as f:
        raw: dict = yaml.safe_load(f)

    raw = _expand_env(raw)
    return PipelineConfig.model_validate(raw)


# ─────────────────────────────────────────────────────────
# CLI helper: python -m ai_waf_v2.utils.config config/pipeline.yaml
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/pipeline.yaml"
    cfg = load_config(cfg_path)
    # pretty-print key settings
    print(json.dumps({
        "project": cfg.project.name,
        "seed": cfg.project.seed,
        "teacher": {
            "d_model":            cfg.model.track_b_99m.d_model,
            "n_layers":           cfg.model.track_b_99m.n_layers,
            "n_heads":            cfg.model.track_b_99m.n_heads,
            "effective_batch":    cfg.training.teacher.effective_batch_size,
            "warmup_steps":       cfg.training.teacher.warmup_steps,
        },
        "slo": {
            "inline_p99_ms": cfg.slo.latency_inline_p99_ms,
            "max_fpr":       cfg.slo.max_false_positive_rate,
        },
    }, indent=2))