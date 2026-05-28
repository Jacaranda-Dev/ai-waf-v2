# ai-waf-v2: Transformer-based WAF Classifier
# Flexible multi-stage pipeline with A/B tokenizer tracks,
# baseline comparison, distillation, and adversarial evaluation
# Usage: make <target> [TRACK=a|b] [SIZE=large|99m|student]

PYTHON     := .venv/bin/python
MLFLOW     := .venv/bin/mlflow
CFG        := config/pipeline.yaml
TRACK      ?= b
SIZE       ?= 99m
# Default installation mode
MODE 	   ?= full

include .env
export $(shell sed 's/=.*//' .env)

.PHONY: all setup \
        baselines baselines_post_aug \
        data_collect data_analyze data_augment_rules data_augment_grammar \
        data_augment_local_llm data_augment_api_llm data_augment_benign \
        data_filter data_validate data_split \
        tokenize_a tokenize_b tokenize_eval \
        train_a_large train_a_small train_b_99m \
        distill_train distill_quant distill_export \
        eval_detection eval_latency eval_adversarial eval_ablation eval_interp \
        compare_all report \
        ui clean

# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────
# Logic to determine which dependency groups to install
ifeq ($(MODE),research)
    INSTALL_TARGET := .[research]
    KERNEL_DESC := "Python (ai-waf-v2 Research)"
else
    INSTALL_TARGET := .[dev,research,full]
    KERNEL_DESC := "Python (ai-waf-v2 Full Stack)"
endif

setup:
	@echo "=== Initializing environment in [$(MODE)] mode ==="
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e "$(INSTALL_TARGET)"
	# Register the kernel automatically
	$(PYTHON) -m ipykernel install --user --name=ai-waf-v2 --display-name $(KERNEL_DESC)
	@echo "Setup complete. Virtual environment mode: $(MODE)"

# ─────────────────────────────────────────────
# STATIC ANALYSIS & LINTING
# ─────────────────────────────────────────────
.PHONY: lint typecheck quality

lint:
	@echo "=== Running Ruff (Linting & Formatting) ==="
	$(PYTHON) -m ruff check . --fix
	$(PYTHON) -m ruff format .

typecheck:
	@echo "=== Running Mypy (Type Checking) ==="
	$(PYTHON) -m mypy stages/

quality: lint typecheck

# ─────────────────────────────────────────────
# TESTING ENGINE
# ─────────────────────────────────────────────
.PHONY: test test_core test_data test_distill

# Run the master suite (all tests)
test:
	@echo "=== Running Master Test Suite ==="
	$(PYTHON) -m pytest tests/test_all.py

# Logic and Architecture tests
test_core:
	@echo "=== Testing Model Architecture & Tokenizers ==="
	$(PYTHON) -m pytest tests/test_model.py tests/test_tokenizer.py

# Data Integrity tests
test_data:
	@echo "=== Testing Data Pipeline & Augmentation ==="
	$(PYTHON) -m pytest tests/test_dataset.py tests/test_augmentation.py

# Performance & Loss tests
test_distill:
	@echo "=== Testing Distillation Logic & Metrics ==="
	$(PYTHON) -m pytest tests/test_distill_loss.py tests/test_metrics.py



# ─────────────────────────────────────────────
# STAGE 1 — DATA ACQUISITION & CURATION
# ─────────────────────────────────────────────
data_collect:
	@echo "=== Stage 1.1–1.2:  Data Acquisition & Curation ==="
	$(PYTHON) stages/1_data_acquisition_and_curation/01_acquire_and_normalize.py           --config $(CFG)
	$(PYTHON) stages/1_data_acquisition_and_curation/02_cross_dataset_dedup.py          --config $(CFG)

data_analyze: data_collect
	@echo "=== Stage 1.3-1.4: Coverage Analysis ==="
	$(PYTHON) stages/1_data_acquisition_and_curation/03_generate_corpus_report.py        --config $(CFG)

# ─────────────────────────────────────────────
# STAGE 2 — BASELINES  (pre-augmentation)
# Runs on the raw curated splits to establish motivation baselines.
# Splits are saved to data/splits_pre_aug/ so they survive Stage 3's overwrite.
# ─────────────────────────────────────────────
baselines:
	@echo "=== Stage 2: Baselines (pre-augmentation) ==="
	$(PYTHON) stages/2_baselines/00_stratified_split.py          --config $(CFG)
	@mkdir -p data/splits_pre_aug
	@cp data/splits/*.parquet data/splits_pre_aug/
	$(PYTHON) stages/2_baselines/01_define_slos.py               --config $(CFG)
	$(PYTHON) stages/2_baselines/02_modsecurity_crs.py           --config $(CFG)
	$(PYTHON) stages/2_baselines/03_classical_ml_baseline.py     --config $(CFG) \
	    --split-dir data/splits_pre_aug --out-file baselines.json --phase pre_aug
	@echo "Pre-augmentation baseline metrics → reports/metrics/baselines.json"

# ─────────────────────────────────────────────
# STAGE 3 — DATA AUGMENTATION
# ─────────────────────────────────────────────
data_augment_rules:
	@echo "=== Stage 3.1-3.2: Rule-based Mutation ==="
	$(PYTHON) stages/3_data_augmentation/01_encoding_mutations.py      --config $(CFG)
	$(PYTHON) stages/3_data_augmentation/02_tamper_scripts.py          --config $(CFG)

data_augment_grammar:
	@echo "=== Stage 3.3-3.6: Grammar + Template ==="
	$(PYTHON) stages/3_data_augmentation/03_grammar_sqli.py            --config $(CFG)
	$(PYTHON) stages/3_data_augmentation/04_grammar_xss.py             --config $(CFG)
	$(PYTHON) stages/3_data_augmentation/05_grammar_lfi_rfi.py         --config $(CFG)
	$(PYTHON) stages/3_data_augmentation/06_grammar_ssrf_cmdi.py       --config $(CFG)

data_augment_local_llm:
	@echo "=== Stage 3.7: Security LLM (local, offline) ==="
	$(PYTHON) stages/3_data_augmentation/07_local_llm_payloads.py      --config $(CFG) \
	    --model-path $(LOCAL_LLM_PATH)

data_augment_api_llm:
	@echo "=== Stage 3.8: Cloud LLM API (framing + benign edge cases) ==="
	$(PYTHON) stages/3_data_augmentation/08_api_llm_framing.py         --config $(CFG) \
	    --provider $(LLM_PROVIDER)

data_augment_benign:
	@echo "=== Stage 3.9-3.10: Benign Traffic Generation ==="
	$(PYTHON) stages/3_data_augmentation/09_benign_rest_traffic.py     --config $(CFG)
	$(PYTHON) stages/3_data_augmentation/10_benign_replay_traces.py    --config $(CFG)

data_filter:
	@echo "=== Stage 3.11-3.14: Quality Filtering ==="
	$(PYTHON) stages/3_data_augmentation/11_format_validation.py       --config $(CFG)
	$(PYTHON) stages/3_data_augmentation/12_tokenizer_coverage_check.py --config $(CFG)
	$(PYTHON) stages/3_data_augmentation/13_semantic_dedup.py          --config $(CFG)
	$(PYTHON) stages/3_data_augmentation/14_label_consistency.py       --config $(CFG)

data_validate:
	@echo "=== Stage 3.15-3-16: Augmentation Contribution Probe ==="
	$(PYTHON) stages/3_data_augmentation/15_augmentation_probe.py      --config $(CFG)
	$(PYTHON) stages/3_data_augmentation/16_taxonomy_inventory.py      --config $(CFG)
	@echo "Check reports/metrics/augmentation_delta.json before proceeding"

data_augment: data_augment_rules data_augment_grammar \
              data_augment_local_llm data_augment_api_llm \
              data_augment_benign data_filter data_validate

data_split:
	@echo "=== Stage 3.17: Stratified Split ==="
	$(PYTHON) stages/3_data_augmentation/17_stratified_split.py           --config $(CFG)
	# Produces: train / val / test / adversarial_holdout / synthetic_canary

# ─────────────────────────────────────────────
# STAGE 4 — TOKENIZATION (A/B parallel tracks)
# ─────────────────────────────────────────────
tokenize_a:
	@echo "=== Stage 3A: Pretrained Vocab Augmentation ==="
	$(PYTHON) stages/4_tokenization/01_augment_pretrained_vocab.py  --config $(CFG)
	$(PYTHON) stages/4_tokenization/02_measure_oov_track_a.py       --config $(CFG)

tokenize_b:
	@echo "=== Stage 3B: Custom BPE from Scratch ==="
	$(PYTHON) stages/4_tokenization/03_train_custom_bpe.py          --config $(CFG)
	$(PYTHON) stages/4_tokenization/04_measure_oov_track_b.py       --config $(CFG)

tokenize_eval:
	@echo "=== Stage 3: Tokenizer Comparison Report ==="
	$(PYTHON) stages/4_tokenization/05_compare_tokenizers.py        --config $(CFG)
	# Metrics: OOV rate, token fertility per attack class, avg seq length

tokenize: tokenize_a tokenize_b tokenize_eval

# ─────────────────────────────────────────────
# STAGE 5 — TEACHER MODEL TRAINING
# ─────────────────────────────────────────────
train_a_large:
	@echo "=== Stage 4.1: Track A — DeBERTa-v3-base fine-tune ==="
	$(PYTHON) stages/5_teacher_training/01_track_a_large.py             --config $(CFG) \
	    --mlflow-run-name "track_a_large_$(shell date +%Y%m%d_%H%M)"

train_a_small:
	@echo "=== Stage 4.2: Track A — small pretrained fine-tune ==="
	$(PYTHON) stages/5_teacher_training/02_track_a_small.py             --config $(CFG) \
	    --mlflow-run-name "track_a_small_$(shell date +%Y%m%d_%H%M)"

train_b_99m:
	@echo "=== Stage 4.3-4.4: Track B — 99M encoder from scratch ==="
	$(PYTHON) stages/5_teacher_training/03_track_b_99m.py               --config $(CFG) \
	    --mlflow-run-name "track_b_99m_$(shell date +%Y%m%d_%H%M)"
	$(PYTHON) stages/5_teacher_training/04_threshold_calibration.py     --config $(CFG) \
	    --model-path models/track_b/best_99m.pt            \
	    --target-fpr 0.001

train_b_99m_canary:
	@echo "=== Stage 4.5: Synthetic Canary Eval (distribution shift check) ==="
	$(PYTHON) stages/5_teacher_training/05_canary_eval.py               --config $(CFG) \
	    --model-path models/track_b/best_99m.pt

# ─────────────────────────────────────────────
# STAGE 6 — DISTILLATION & COMPRESSION
# ─────────────────────────────────────────────
distill_train:
	@echo "=== Stage 6.1–6.2: Knowledge Distillation → student ==="
	$(PYTHON) stages/6_distillation_and_compression/01_student_arch.py            --config $(CFG)
	$(PYTHON) stages/6_distillation_and_compression/02_distill_train.py           --config $(CFG) \
	    --teacher-path models/track_b/best_99m.pt          \
	    --mlflow-run-name "student_distill_$(shell date +%Y%m%d_%H%M)"

distill_quant:
	@echo "=== Stage 6.3=6.4: QAT vs PTQ Comparison ==="
	$(PYTHON) stages/6_distillation_and_compression/03_post_training_quant.py     --config $(CFG)
	$(PYTHON) stages/6_distillation_and_compression/04_qat_comparison.py          --config $(CFG)

distill_export:
	@echo "=== Stage 6.5–6.7: ONNX + TensorRT Export ==="
	$(PYTHON) stages/6_distillation_and_compression/05_export_onnx.py             --config $(CFG)
	$(PYTHON) stages/6_distillation_and_compression/06_export_trt.py              --config $(CFG)
	$(PYTHON) stages/6_distillation_and_compression/07_onnxruntime_bench.py       --config $(CFG)

distill: distill_train distill_quant distill_export

# ─────────────────────────────────────────────
# STAGE 7 — EVALUATION
# ─────────────────────────────────────────────
eval_detection:
	@echo "=== Stage 7.1: Detection Efficacy (all models) ==="
	$(PYTHON) stages/7_evaluation/01_detection_metrics.py          --config $(CFG)
	# F1, precision, recall, FPR, FNR, AUC-ROC, AUC-PR
	# Per model AND per attack class breakdown

eval_latency:
	@echo "=== Stage 7.2–7.3: Latency & Throughput Benchmark ==="
	$(PYTHON) stages/7_evaluation/02_latency_bench.py              --config $(CFG) \
	    --batch-sizes 1 8 32 64 256                        \
	    --devices gpu cpu
	$(PYTHON) stages/7_evaluation/03_memory_footprint.py           --config $(CFG)

eval_adversarial:
	@echo "=== Stage 7.4-7.6: Adversarial Robustness ==="
	$(PYTHON) stages/7_evaluation/04_evasion_payloads.py           --config $(CFG)
	$(PYTHON) stages/7_evaluation/05_obfuscation_robustness.py     --config $(CFG)
	$(PYTHON) stages/7_evaluation/06_novel_attack_generalization.py --config $(CFG)

eval_ablation:
	@echo "=== Stage 7.7-7.10: Ablation Studies ==="
	$(PYTHON) stages/7_evaluation/07_tokenizer_ablation.py         --config $(CFG)
	$(PYTHON) stages/7_evaluation/08_augmentation_ablation.py      --config $(CFG)
	$(PYTHON) stages/7_evaluation/09_model_size_scaling.py         --config $(CFG)
	$(PYTHON) stages/7_evaluation/10_label_smoothing_ablation.py   --config $(CFG)

eval_interp:
	@echo "=== Stage 7.11-7.13: Interpretability ==="
	$(PYTHON) stages/7_evaluation/11_attention_visualization.py    --config $(CFG)
	$(PYTHON) stages/7_evaluation/12_shap_analysis.py              --config $(CFG)
	$(PYTHON) stages/7_evaluation/13_error_analysis.py             --config $(CFG)

# Re-run classical baselines on the augmented splits for a fair comparison
# against the neural models in the master comparison table.
# Requires data_split (Stage 3.17) to have run first.
baselines_post_aug:
	@echo "=== Stage 7: Baselines (post-augmentation — fair comparison) ==="
	$(PYTHON) stages/2_baselines/03_classical_ml_baseline.py     --config $(CFG) \
	    --out-file baselines_post_aug.json --phase post_aug
	@echo "Post-augmentation baseline metrics → reports/metrics/baselines_post_aug.json"

compare_all: eval_detection eval_latency eval_adversarial eval_ablation eval_interp baselines_post_aug
	@echo "=== Stage 7.14: Master Comparison Table ==="
	$(PYTHON) stages/7_evaluation/14_comparison_table.py           --config $(CFG)
	# Produces Table 1 — all models vs all baselines (post-aug) vs all metrics

report:
	@echo "=== Final Report Generation ==="
	$(PYTHON) stages/7_evaluation/15_generate_report.py            --config $(CFG)
	@echo "Report written to reports/final_report.md"

# ─────────────────────────────────────────────
# CONVENIENCE TARGETS
# ─────────────────────────────────────────────

# Run the full pipeline end-to-end
all: setup baselines data_collect data_analyze data_augment data_split \
     tokenize train_b_99m distill compare_all report

# Run only Track B (custom tokenizer, scratch training) — fastest path to results
track_b_full: setup baselines data_collect data_augment data_split \
              tokenize_b train_b_99m distill compare_all

# Run only evaluation on already-trained models
eval_only: eval_detection eval_latency eval_adversarial eval_ablation \
           eval_interp compare_all

ui:
	$(MLFLOW) ui --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db 

stop_ui:                                                                                                    
	@pkill -f "mlflow ui" && echo "MLflow UI stopped." || echo "No MLflow UI process found."                  
   
# ─────────────────────────────────────────────
# CLEAN TARGETS (granular — don't nuke everything)
# ─────────────────────────────────────────────
clean_augmented:
	rm -rf data/augmented/**/*.parquet

clean_models:
	rm -rf models/track_a/* models/track_b/* models/student/*

clean_tokenizers:
	rm -rf tokenizers/track_a/* tokenizers/track_b/*

clean_reports:
	rm -rf reports/metrics/* reports/figures/* reports/latency/*

clean_splits:
	rm -rf data/splits/*

clean: clean_augmented clean_models clean_tokenizers clean_reports clean_splits