#!/usr/bin/env bash
# ai-waf-v2: Project scaffold
# Creates all directories and empty Python/config files for the pipeline.
# Run once from the project root: bash scaffold.sh

set -euo pipefail

PROJECT="${1:-.}"   # pass a path argument or default to current directory
cd "$PROJECT"

echo "Scaffolding ai-waf-v2 project in: $(pwd)"

# ─────────────────────────────────────────────
# DIRECTORIES
# ─────────────────────────────────────────────
dirs=(
    # data lake
    data/raw
    data/normalized
    data/augmented/rules
    data/augmented/grammar
    data/augmented/local_llm
    data/augmented/api_llm
    data/augmented/benign
    data/filtered
    data/splits

    # models
    models/baselines
    models/track_a
    models/track_b
    models/student

    # tokenizers
    tokenizers/track_a
    tokenizers/track_b

    # reports
    reports/metrics
    reports/figures
    reports/latency

    # mlflow
    mlruns

    # config
    config

    # pipeline stages
    stages/baselines
    stages/data
    stages/augment
    stages/tokenizer
    stages/train
    stages/distill
    stages/eval

    # shared library
    ai-waf-v2/data
    ai-waf-v2/models
    ai-waf-v2/tokenizer
    ai-waf-v2/distill
    ai-waf-v2/eval
    ai-waf-v2/utils

    # tests
    tests

    # notebooks
    notebooks
)

for d in "${dirs[@]}"; do
    mkdir -p "$d"
done
echo "  ✓ directories created"

# ─────────────────────────────────────────────
# HELPER: touch a file only if it does not exist
# ─────────────────────────────────────────────
t() { [[ -f "$1" ]] || touch "$1"; }

# ─────────────────────────────────────────────
# TOP-LEVEL PROJECT FILES
# ─────────────────────────────────────────────
t .env
t .gitignore
t README.md
t pyproject.toml
t Makefile
t config/pipeline.yaml
t config/logging.yaml

# ─────────────────────────────────────────────
# STAGE 0 — BASELINES
# ─────────────────────────────────────────────
stage0=(
    stages/baselines/00_define_slos.py
    stages/baselines/01_modsecurity_crs.py
    stages/baselines/02_tfidf_xgboost.py
    stages/baselines/03_tfidf_lightgbm.py
    stages/baselines/04_taxonomy_inventory.py
)
for f in "${stage0[@]}"; do t "$f"; done
echo "  ✓ stage 0 — baselines"

# ─────────────────────────────────────────────
# STAGE 1 — DATA COLLECTION & CURATION
# ─────────────────────────────────────────────
stage1=(
    stages/data/01_collect_datasets.py
    stages/data/02_normalize_schema.py
    stages/data/03_cross_dataset_dedup.py
    stages/data/04_dataset_analysis.py
    stages/data/05_datasheet.py
    stages/data/06_taxonomy_coverage.py
    stages/data/07_length_distribution.py
    stages/data/08_stratified_split.py
)
for f in "${stage1[@]}"; do t "$f"; done
echo "  ✓ stage 1 — data collection"

# ─────────────────────────────────────────────
# STAGE 2 — AUGMENTATION
# ─────────────────────────────────────────────
stage2=(
    stages/augment/01_encoding_mutations.py
    stages/augment/02_tamper_scripts.py
    stages/augment/03_grammar_sqli.py
    stages/augment/04_grammar_xss.py
    stages/augment/05_grammar_lfi_rfi.py
    stages/augment/06_grammar_ssrf_cmdi.py
    stages/augment/07_local_llm_payloads.py
    stages/augment/08_api_llm_framing.py
    stages/augment/09_benign_rest_traffic.py
    stages/augment/10_benign_replay_traces.py
    stages/augment/11_format_validation.py
    stages/augment/12_tokenizer_coverage_check.py
    stages/augment/13_semantic_dedup.py
    stages/augment/14_label_consistency.py
    stages/augment/15_augmentation_probe.py
)
for f in "${stage2[@]}"; do t "$f"; done
echo "  ✓ stage 2 — augmentation"

# ─────────────────────────────────────────────
# STAGE 3 — TOKENIZATION
# ─────────────────────────────────────────────
stage3=(
    stages/tokenizer/01_augment_pretrained_vocab.py
    stages/tokenizer/02_measure_oov_track_a.py
    stages/tokenizer/03_train_custom_bpe.py
    stages/tokenizer/04_measure_oov_track_b.py
    stages/tokenizer/05_compare_tokenizers.py
)
for f in "${stage3[@]}"; do t "$f"; done
echo "  ✓ stage 3 — tokenization"

# ─────────────────────────────────────────────
# STAGE 4 — TRAINING
# ─────────────────────────────────────────────
stage4=(
    stages/train/01_track_a_large.py
    stages/train/02_track_a_small.py
    stages/train/03_track_b_99m.py
    stages/train/04_threshold_calibration.py
    stages/train/05_canary_eval.py
)
for f in "${stage4[@]}"; do t "$f"; done
echo "  ✓ stage 4 — training"

# ─────────────────────────────────────────────
# STAGE 5 — DISTILLATION
# ─────────────────────────────────────────────
stage5=(
    stages/distill/01_student_arch.py
    stages/distill/02_distill_train.py
    stages/distill/03_post_training_quant.py
    stages/distill/04_qat_comparison.py
    stages/distill/05_export_onnx.py
    stages/distill/06_export_trt.py
    stages/distill/07_onnxruntime_bench.py
)
for f in "${stage5[@]}"; do t "$f"; done
echo "  ✓ stage 5 — distillation"

# ─────────────────────────────────────────────
# STAGE 6/7 — EVALUATION & INTERPRETABILITY
# ─────────────────────────────────────────────
stage6=(
    stages/eval/01_detection_metrics.py
    stages/eval/02_latency_bench.py
    stages/eval/03_memory_footprint.py
    stages/eval/04_evasion_payloads.py
    stages/eval/05_obfuscation_robustness.py
    stages/eval/06_novel_attack_generalization.py
    stages/eval/07_tokenizer_ablation.py
    stages/eval/08_augmentation_ablation.py
    stages/eval/09_model_size_scaling.py
    stages/eval/10_label_smoothing_ablation.py
    stages/eval/11_attention_visualization.py
    stages/eval/12_shap_analysis.py
    stages/eval/13_error_analysis.py
    stages/eval/14_comparison_table.py
    stages/eval/15_generate_report.py
)
for f in "${stage6[@]}"; do t "$f"; done
echo "  ✓ stage 6/7 — evaluation & interpretability"

# ─────────────────────────────────────────────
# SHARED LIBRARY (ai-waf-v2 package)
# ─────────────────────────────────────────────
lib=(
    ai-waf-v2/__init__.py
    ai-waf-v2/data/__init__.py
    ai-waf-v2/data/schema.py
    ai-waf-v2/data/dataset.py
    ai-waf-v2/data/collator.py
    ai-waf-v2/models/__init__.py
    ai-waf-v2/models/encoder.py
    ai-waf-v2/models/head.py
    ai-waf-v2/models/student.py
    ai-waf-v2/tokenizer/__init__.py
    ai-waf-v2/tokenizer/http_tokenizer.py
    ai-waf-v2/tokenizer/vocab_utils.py
    ai-waf-v2/distill/__init__.py
    ai-waf-v2/distill/losses.py
    ai-waf-v2/distill/trainer.py
    ai-waf-v2/eval/__init__.py
    ai-waf-v2/eval/metrics.py
    ai-waf-v2/eval/latency.py
    ai-waf-v2/eval/adversarial.py
    ai-waf-v2/utils/__init__.py
    ai-waf-v2/utils/config.py
    ai-waf-v2/utils/logging.py
    ai-waf-v2/utils/mlflow_utils.py
    ai-waf-v2/utils/seed.py
)
for f in "${lib[@]}"; do t "$f"; done
echo "  ✓ ai-waf-v2 package"

# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────
tests=(
    tests/__init__.py
    tests/test_tokenizer.py
    tests/test_dataset.py
    tests/test_model.py
    tests/test_distill_loss.py
    tests/test_metrics.py
    tests/test_augmentation.py
)
for f in "${tests[@]}"; do t "$f"; done
echo "  ✓ tests"

# ─────────────────────────────────────────────
# .gitignore defaults
# ─────────────────────────────────────────────
cat > .gitignore << 'EOF'
# python
.venv/
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
env/
bin/
lib/
include/


# data (track structure, not content)
data/raw/**
data/normalized/**
data/augmented/**/**
data/filtered/**
data/splits/**
!data/**/.gitkeep

# models
models/**/*.pt
models/**/*.parquet
models/**/*.ckpt
models/**/*.weights
models/**/*.onnx
models/**/*.engine
models/**/*.bin
!models/**/.gitkeep

# tokenizers (large files)
tokenizers/**/*.model
tokenizers/**/*.vocab
!tokenizers/**/.gitkeep

# mlflow
mlruns/

# reports (generated)
reports/metrics/*.json
reports/figures/*.png
reports/latency/*.json

# Logs and Experiment Tracking
runs/
mlruns/
logs/
.hydra/
outputs/


# Credentials and Local Configs
.env
.env.local
configs/local_*.yaml


# ide
#VS Code specific
.vscode/
*.code-workspace
.idea/

# OS specific files
*.DS_Store
Thumbs.db
EOF
echo "  ✓ .gitignore"

# ─────────────────────────────────────────────
# .gitkeep placeholders so empty dirs are tracked
# ─────────────────────────────────────────────
gitkeep_dirs=(
    data/raw
    data/normalized
    data/augmented/rules
    data/augmented/grammar
    data/augmented/local_llm
    data/augmented/api_llm
    data/augmented/benign
    data/filtered
    data/splits
    models/baselines
    models/track_a
    models/track_b
    models/student
    tokenizers/track_a
    tokenizers/track_b
    reports/metrics
    reports/figures
    reports/latency
    mlruns
    notebooks
)
for d in "${gitkeep_dirs[@]}"; do
    touch "$d/.gitkeep"
done
echo "  ✓ .gitkeep placeholders"

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
echo ""
echo "Done. Project structure:"
echo ""
find . -not -path './.git/*' \
       -not -path './.venv/*' \
       -not -name '*.pyc' \
  | sort \
  | sed 's|[^/]*/|  |g' \
  | head -120
echo ""
echo "Next: cp .env.example .env  →  fill in API keys  →  make setup"