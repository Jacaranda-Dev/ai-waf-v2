# Model Card — ai-waf-v2 student classifier

> **📖 Docs:** [Index](README.md) · [User Guide](USER_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [API](API.md) · [Stages](stages/stages.md) · [Model Card](MODEL_CARD.md) · [Repo README](../README.md)

> Template following Mitchell et al., *Model Cards for Model Reporting* (2019).
> Fields marked _(fill in)_ should be populated from the latest evaluation run
> (`reports/7_evaluation/15_final_evaluation_report.json`, `reports/7_evaluation/metrics/14_master_comparison_table.csv`) before publishing.

## Model details

- **Developed by:** _(fill in)_
- **Model type:** Transformer encoder (custom `WafEncoder` architecture), binary
  sequence classifier over tokenized HTTP requests.
- **Variants:**
  - *Teacher* — ~99M params (d_model 768, 13 layers, 12 heads), trained from
    scratch with the Track B custom BPE tokenizer.
  - *Student* — ~10M params (d_model 256, 6 layers, 4 heads), knowledge-distilled
    from the teacher; the deployment target.
- **Tokenizer:** byte-level HTTP-aware BPE, vocab 8000, sequence length 256.
- **Version / checkpoint:** _(fill in — git SHA + MLflow run id)_
- **License:** _(fill in)_

## Intended use

- **Primary use:** inline or offline classification of HTTP requests as benign vs
  malicious, as one signal in a Web Application Firewall.
- **Users:** security engineers deploying WAF tooling; ML researchers studying
  attack detection.
- **Out of scope:** sole authority for blocking decisions without human/defence-in-
  depth review; classification of non-HTTP traffic; attribution of attacker
  identity; use as an offensive tool.

## Factors

Performance varies by **attack class** (sqli, xss, lfi, ssrf, cmdi, benign) and by
**obfuscation/encoding** (see the adversarial tamper suite). Report per-class
metrics, not just aggregates.

## Metrics

- **Operating point:** FPR = 0.001 (false positives are costly for a WAF).
- **Reported:** F1, precision, recall, FPR, FNR, AUC-ROC, AUC-PR — overall and
  per attack class (`ai_waf_v2/eval/metrics.py`).
- **Latency SLOs:** p99 < 5 ms inline (batch 1), < 50 ms offline (batch 64).
- **Latest results:** _(fill in from `reports/7_evaluation/metrics/14_master_comparison_table.csv`)_

| Metric | Teacher 99M | Student ~10M | XGBoost baseline | ModSecurity CRS |
|---|---|---|---|---|
| AUC-PR | _(fill)_ | _(fill)_ | _(fill)_ | _(fill)_ |
| Recall @ FPR 0.001 | _(fill)_ | _(fill)_ | _(fill)_ | _(fill)_ |
| p99 latency (ms) | _(fill)_ | _(fill)_ | _(fill)_ | _(fill)_ |

## Training data

- **Sources:** public HTTP datasets (CSIC 2010, SR-BH 2020, ECML/PKDD 2007, plus
  any declared in `config/pipeline.yaml`), normalized to the `HttpRecord` schema
  and cross-dataset deduplicated.
- **Augmentation:** recursive-PCFG/grammar/mutator/tamper/LLM-synthesised attacks
  and benign traffic, framed with PCAP-fitted request metadata; filtered by a
  five-pass quality gate with a test/canary leakage guard.
- **Splits:** 70% train / 15% val / 10% test / 3% adversarial / 2% canary,
  stratified by `(label × attack_class)`. See the corpus datasheet at
  `reports/corpus_report.html`.

## Evaluation data

Held-out `test` split plus dedicated `adversarial` and `canary` splits. Stage 7
additionally evaluates obfuscation robustness and generalisation to held-out
attack classes.

## Ethical considerations & limitations

- **False negatives** allow attacks through; **false positives** block legitimate
  users. The FPR=0.001 operating point and human review mitigate the latter.
- Trained largely on **public/synthetic** data; real-world traffic distribution
  shift is expected — monitor the canary split and recalibrate.
- Adversaries adapt; novel encodings or attack classes not represented in training
  may evade detection. Treat as one layer of defence, not a complete control.
- Synthetic generation may encode biases of its source LLMs/grammars.

## Caveats & recommendations

Recalibrate the decision threshold on in-domain validation data before deployment
(`stages/6_distillation_and_compression/03_student_calibrate.py`) and re-run the
adversarial suite after any retraining.
