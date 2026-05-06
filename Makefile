# ai-waf-v2: Transformer-based WAF Classifier
# Flexible multi-stage pipeline with A/B tokenizer tracks,
# baseline comparison, distillation, and adversarial evaluation
# Usage: make <target> [TRACK=a|b] [SIZE=large|99m|student]

PYTHON     := .venv/bin/python
MLFLOW     := .venv/bin/mlflow
CFG        := config/pipeline.yaml
TRACK      ?= b
SIZE       ?= 99m

include .env
export $(shell sed 's/=.*//' .env)

.PHONY: all setup \
        baselines \
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
setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	mkdir -p data/{raw,normalized,augmented/{rules,grammar,local_llm,api_llm,benign},filtered,splits} \
	         models/{baselines,track_a,track_b,student} \
	         tokenizers/{track_a,track_b} \
	         reports/{metrics,figures,latency} \
	         mlruns

# ─────────────────────────────────────────────
# STAGE 0 — BASELINES & INFRASTRUCTURE
# ─────────────────────────────────────────────
baselines:
	@echo "=== Stage 0: Baselines ==="
	$(PYTHON) stages/baselines/00_define_slos.py           --config $(CFG)
	$(PYTHON) stages/baselines/01_modsecurity_crs.py       --config $(CFG)
	$(PYTHON) stages/baselines/02_tfidf_xgboost.py         --config $(CFG)
	$(PYTHON) stages/baselines/03_tfidf_lightgbm.py        --config $(CFG)
	$(PYTHON) stages/baselines/04_taxonomy_inventory.py    --config $(CFG)
	@echo "Baseline metrics written to reports/metrics/baselines.json"

# ─────────────────────────────────────────────
# STAGE 1 — DATA COLLECTION & CURATION
# ─────────────────────────────────────────────
data_collect:
	@echo "=== Stage 1.1–1.4: Collection & Normalization ==="
	$(PYTHON) stages/data/01_collect_datasets.py           --config $(CFG)
	$(PYTHON) stages/data/02_normalize_schema.py           --config $(CFG)
	$(PYTHON) stages/data/03_cross_dataset_dedup.py        --config $(CFG)
	$(PYTHON) stages/data/04_dataset_analysis.py           --config $(CFG)
	$(PYTHON) stages/data/05_datasheet.py                  --config $(CFG)

data_analyze: data_collect
	@echo "=== Stage 1: Coverage Analysis ==="
	$(PYTHON) stages/data/06_taxonomy_coverage.py          --config $(CFG)
	$(PYTHON) stages/data/07_length_distribution.py        --config $(CFG)

# ─────────────────────────────────────────────
# STAGE 2 — AUGMENTATION (substages are independent)
# ─────────────────────────────────────────────
data_augment_rules:
	@echo "=== Stage 2.1: Rule-based Mutation ==="
	$(PYTHON) stages/augment/01_encoding_mutations.py      --config $(CFG)
	$(PYTHON) stages/augment/02_tamper_scripts.py          --config $(CFG)

data_augment_grammar:
	@echo "=== Stage 2.2: Grammar + Template ==="
	$(PYTHON) stages/augment/03_grammar_sqli.py            --config $(CFG)
	$(PYTHON) stages/augment/04_grammar_xss.py             --config $(CFG)
	$(PYTHON) stages/augment/05_grammar_lfi_rfi.py         --config $(CFG)
	$(PYTHON) stages/augment/06_grammar_ssrf_cmdi.py       --config $(CFG)

data_augment_local_llm:
	@echo "=== Stage 2.3: Security LLM (local, offline) ==="
	$(PYTHON) stages/augment/07_local_llm_payloads.py      --config $(CFG) \
	    --model-path $(LOCAL_LLM_PATH)

data_augment_api_llm:
	@echo "=== Stage 2.4: Cloud LLM API (framing + benign edge cases) ==="
	$(PYTHON) stages/augment/08_api_llm_framing.py         --config $(CFG) \
	    --provider $(LLM_PROVIDER)

data_augment_benign:
	@echo "=== Stage 2.5: Benign Traffic Generation ==="
	$(PYTHON) stages/augment/09_benign_rest_traffic.py     --config $(CFG)
	$(PYTHON) stages/augment/10_benign_replay_traces.py    --config $(CFG)

data_filter:
	@echo "=== Stage 2.6: Quality Filtering ==="
	$(PYTHON) stages/augment/11_format_validation.py       --config $(CFG)
	$(PYTHON) stages/augment/12_tokenizer_coverage_check.py --config $(CFG)
	$(PYTHON) stages/augment/13_semantic_dedup.py          --config $(CFG)
	$(PYTHON) stages/augment/14_label_consistency.py       --config $(CFG)

data_validate:
	@echo "=== Stage 2.7: Augmentation Contribution Probe ==="
	$(PYTHON) stages/augment/15_augmentation_probe.py      --config $(CFG)
	@echo "Check reports/metrics/augmentation_delta.json before proceeding"

data_augment: data_augment_rules data_augment_grammar \
              data_augment_local_llm data_augment_api_llm \
              data_augment_benign data_filter data_validate

data_split:
	@echo "=== Stage 2.8: Stratified Split ==="
	$(PYTHON) stages/data/08_stratified_split.py           --config $(CFG)
	# Produces: train / val / test / adversarial_holdout / synthetic_canary

# ─────────────────────────────────────────────
# STAGE 3 — TOKENIZATION (A/B parallel tracks)
# ─────────────────────────────────────────────
tokenize_a:
	@echo "=== Stage 3A: Pretrained Vocab Augmentation ==="
	$(PYTHON) stages/tokenizer/01_augment_pretrained_vocab.py  --config $(CFG)
	$(PYTHON) stages/tokenizer/02_measure_oov_track_a.py       --config $(CFG)

tokenize_b:
	@echo "=== Stage 3B: Custom BPE from Scratch ==="
	$(PYTHON) stages/tokenizer/03_train_custom_bpe.py          --config $(CFG)
	$(PYTHON) stages/tokenizer/04_measure_oov_track_b.py       --config $(CFG)

tokenize_eval:
	@echo "=== Stage 3: Tokenizer Comparison Report ==="
	$(PYTHON) stages/tokenizer/05_compare_tokenizers.py        --config $(CFG)
	# Metrics: OOV rate, token fertility per attack class, avg seq length

tokenize: tokenize_a tokenize_b tokenize_eval

# ─────────────────────────────────────────────
# STAGE 4 — MODEL TRAINING
# ─────────────────────────────────────────────
train_a_large:
	@echo "=== Stage 4.1: Track A — DeBERTa-v3-base fine-tune ==="
	$(PYTHON) stages/train/01_track_a_large.py             --config $(CFG) \
	    --mlflow-run-name "track_a_large_$(shell date +%Y%m%d_%H%M)"

train_a_small:
	@echo "=== Stage 4.2: Track A — small pretrained fine-tune ==="
	$(PYTHON) stages/train/02_track_a_small.py             --config $(CFG) \
	    --mlflow-run-name "track_a_small_$(shell date +%Y%m%d_%H%M)"

train_b_99m:
	@echo "=== Stage 4.3: Track B — 99M encoder from scratch ==="
	$(PYTHON) stages/train/03_track_b_99m.py               --config $(CFG) \
	    --mlflow-run-name "track_b_99m_$(shell date +%Y%m%d_%H%M)"
	$(PYTHON) stages/train/04_threshold_calibration.py     --config $(CFG) \
	    --model-path models/track_b/best_99m.pt            \
	    --target-fpr 0.001

train_b_99m_canary:
	@echo "=== Stage 4.4: Synthetic Canary Eval (distribution shift check) ==="
	$(PYTHON) stages/train/05_canary_eval.py               --config $(CFG) \
	    --model-path models/track_b/best_99m.pt

# ─────────────────────────────────────────────
# STAGE 5 — DISTILLATION & COMPRESSION
# ─────────────────────────────────────────────
distill_train:
	@echo "=== Stage 5.1–5.2: Knowledge Distillation → student ==="
	$(PYTHON) stages/distill/01_student_arch.py            --config $(CFG)
	$(PYTHON) stages/distill/02_distill_train.py           --config $(CFG) \
	    --teacher-path models/track_b/best_99m.pt          \
	    --mlflow-run-name "student_distill_$(shell date +%Y%m%d_%H%M)"

distill_quant:
	@echo "=== Stage 5.3: QAT vs PTQ Comparison ==="
	$(PYTHON) stages/distill/03_post_training_quant.py     --config $(CFG)
	$(PYTHON) stages/distill/04_qat_comparison.py          --config $(CFG)

distill_export:
	@echo "=== Stage 5.4–5.5: ONNX + TensorRT Export ==="
	$(PYTHON) stages/distill/05_export_onnx.py             --config $(CFG)
	$(PYTHON) stages/distill/06_export_trt.py              --config $(CFG)
	$(PYTHON) stages/distill/07_onnxruntime_bench.py       --config $(CFG)

distill: distill_train distill_quant distill_export

# ─────────────────────────────────────────────
# STAGE 6 — EVALUATION
# ─────────────────────────────────────────────
eval_detection:
	@echo "=== Stage 6.1: Detection Efficacy (all models) ==="
	$(PYTHON) stages/eval/01_detection_metrics.py          --config $(CFG)
	# F1, precision, recall, FPR, FNR, AUC-ROC, AUC-PR
	# Per model AND per attack class breakdown

eval_latency:
	@echo "=== Stage 6.2–6.3: Latency & Throughput Benchmark ==="
	$(PYTHON) stages/eval/02_latency_bench.py              --config $(CFG) \
	    --batch-sizes 1 8 32 64 256                        \
	    --devices gpu cpu
	$(PYTHON) stages/eval/03_memory_footprint.py           --config $(CFG)

eval_adversarial:
	@echo "=== Stage 6.4: Adversarial Robustness ==="
	$(PYTHON) stages/eval/04_evasion_payloads.py           --config $(CFG)
	$(PYTHON) stages/eval/05_obfuscation_robustness.py     --config $(CFG)
	$(PYTHON) stages/eval/06_novel_attack_generalization.py --config $(CFG)

eval_ablation:
	@echo "=== Stage 6.5: Ablation Studies ==="
	$(PYTHON) stages/eval/07_tokenizer_ablation.py         --config $(CFG)
	$(PYTHON) stages/eval/08_augmentation_ablation.py      --config $(CFG)
	$(PYTHON) stages/eval/09_model_size_scaling.py         --config $(CFG)
	$(PYTHON) stages/eval/10_label_smoothing_ablation.py   --config $(CFG)

eval_interp:
	@echo "=== Stage 7: Interpretability ==="
	$(PYTHON) stages/eval/11_attention_visualization.py    --config $(CFG)
	$(PYTHON) stages/eval/12_shap_analysis.py              --config $(CFG)
	$(PYTHON) stages/eval/13_error_analysis.py             --config $(CFG)

compare_all: eval_detection eval_latency eval_adversarial eval_ablation eval_interp
	@echo "=== Stage 6.6: Master Comparison Table ==="
	$(PYTHON) stages/eval/14_comparison_table.py           --config $(CFG)
	# Produces Table 1 — all models vs all baselines vs all metrics

report:
	@echo "=== Final Report Generation ==="
	$(PYTHON) stages/eval/15_generate_report.py            --config $(CFG)
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
	$(MLFLOW) ui --port 5000 --backend-store-uri ./mlruns

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