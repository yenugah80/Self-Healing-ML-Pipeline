# Self-Healing Machine Learning Pipelines Using Agentic AI

A reference implementation of the framework proposed in:

> **Self-Healing Machine Learning Pipelines Using Agentic AI: A Framework for Autonomous Model Monitoring and Retraining**
> Mohammad Nasim, Harika Yenuga, Itauma Itauma — Northwood University, Midland, MI, USA
> *International Business Analytics Conference for Academic and Industry Professionals (IBAC), Vol. 01, Issue 01, May 2026.*

---

## Overview

Machine learning models in production degrade over time as input distributions shift (**data drift**) or as the relationship between inputs and labels evolves (**concept drift**). Traditional MLOps pipelines react with fixed retraining schedules or manual intervention.

This project implements a **closed-loop, self-healing ML pipeline** that:

1. Monitors live model performance (F1, precision, recall, ROC-AUC).
2. Detects distributional shift using **KL divergence**.
3. Uses an **Agentic AI decision controller** to choose one of: `Continue`, `Warning`, `Tune Hyperparameters`, `Retrain`, or `Manual Review`.
4. Autonomously retrains and selects the best candidate model when drift + degradation are detected.

The framework is evaluated on the **Credit Card Fraud Detection** dataset (highly imbalanced, ~0.17% positive class), with simulated drift across 18 streaming batches.

---

## Architecture

```
                ┌──────────────────┐
                │  Incoming Data   │
                │   (Batch t)      │
                └────────┬─────────┘
                         ▼
        ┌────────────────────────────────┐
        │   Performance Monitoring       │  → F1, Precision, Recall, ROC-AUC
        │   Drift Detection (KL Div.)    │  → KL(P‖Q)
        └────────────────┬───────────────┘
                         ▼
        ┌────────────────────────────────┐
        │   Agentic AI Decision Engine   │
        │   Rules over (F1_t, KL_t)      │
        └────────────────┬───────────────┘
                         ▼
   ┌──────────┬──────────┬─────────┬──────────┬──────────────┐
   │ Continue │ Warning  │  Tune   │ Retrain  │ Manual Review│
   └──────────┴──────────┴─────────┴────┬─────┴──────────────┘
                                        ▼
                      ┌──────────────────────────────┐
                      │  Retraining Engine           │
                      │  (RF / XGBoost / LogReg)     │
                      │  Pick argmax F1 → Deploy     │
                      └────────────┬─────────────────┘
                                   ▼
                          (Feedback / Audit Log)
```

### Decision rules (Section 5.7 of the paper)

| Condition | Action |
|---|---|
| `F1 ≥ 0.70` and `KL < 0.05` | Continue |
| `F1 < 0.70` and `KL < 0.05` | Tune Hyperparameters |
| `F1 ≥ 0.70` and `KL ≥ 0.05` | Warning |
| `F1 < 0.70` and `KL ≥ 0.05` | Retrain |
| `F1 < 0.40` and `KL ≥ 0.10` | Manual Review |

---

## Repository Structure

```
.
├── step1_baseline_experiment.py          # Train Logistic Regression, RF, XGBoost baselines
├── step2_drift_monitoring.py             # Simulate drifted batches, compute KL divergence
├── step3_agentic_decision_controller.py  # Apply rule-based agent to each batch
├── step4_self_healing_retraining.py      # Trigger retraining + candidate model selection
├── step5_results_visualization.py        # Generate the five charts used in the paper
│
├── step1_baseline_results.csv            # Baseline metrics for the three models
├── step2_drift_monitoring_results.csv    # Per-batch KL drift scores
├── step3_agentic_decision_results.csv    # Per-batch agent decisions + reasons
├── step4_candidate_model_log.csv         # Candidate models trained during retraining
├── step4_self_healing_results.csv        # Before/after F1 with healing flag
│
├── chart1_baseline_f1.png                # Figure 2 — Baseline F1 comparison
├── chart2_f1_trend.png                   # Figure 3 — F1 trend with self-healing
├── chart3_drift_vs_f1.png                # Figure 4 — KL drift vs. F1
├── chart4_agent_distribution.png         # Figure 5 — Agent action distribution
├── chart5_self_healing.png               # Figure 6 — Before vs. after retraining
│
├── requirements.txt
└── README.md
```

---

## Installation

```bash
# Clone
git clone https://github.com/MohammadNasim/Self_Healing_ML_Pipeline.git
cd Self_Healing_ML_Pipeline

# Create environment (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `matplotlib`, `seaborn`.

---

## Dataset

The pipeline uses the **Credit Card Fraud Detection** dataset (Dal Pozzolo et al., 2015):

- 284,807 transactions, 492 fraudulent (~0.17%)
- 28 PCA-anonymized features (`V1`–`V28`) plus `Time`, `Amount`, `Class`
- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The CSV (`creditcard.csv`, ~144 MB) is **not committed** because it exceeds GitHub's 100 MB file limit. Download it from Kaggle and place it in the project root before running Step 1.

---

## How to Run

Run the scripts in order — each one consumes outputs from the previous step:

```bash
python step1_baseline_experiment.py            # → step1_baseline_results.csv
python step2_drift_monitoring.py               # → step2_drift_monitoring_results.csv
python step3_agentic_decision_controller.py    # → step3_agentic_decision_results.csv
python step4_self_healing_retraining.py        # → step4_self_healing_results.csv
                                               #   step4_candidate_model_log.csv
python step5_results_visualization.py          # → chart1..chart5 PNGs
```

### Drift simulation (Section 5.4)

Step 2 splits the test set into **18 sequential batches of 5,000** transactions and injects Gaussian noise into features `V1, V2, V3, V4, V10, V11, V12, V14, Amount`:

| Batches | Drift strength | Description |
|---|---|---|
| 1–5   | 0.0 | No drift (baseline) |
| 6–10  | 0.3 | Moderate drift |
| 11–18 | 0.7 | Severe drift |

---

## Key Results

### Baseline F1 (Section 6.1)

| Model | F1-score |
|---|---|
| Logistic Regression | 0.12 |
| **Random Forest**   | **0.82** |
| XGBoost             | 0.53 |

Random Forest is selected as the initial production model.

### Self-Healing Impact (Sections 6.5–6.6)

| Metric | Value |
|---|---|
| Average F1 (before healing) | 0.7338 |
| Average F1 (after healing)  | 0.7688 |
| Net improvement             | **+0.0350** |
| Retrain events triggered    | 2 (Batches 12 and 16) |

Targeted retraining recovered F1 from **0.54 → 0.86** at Batch 12 and **0.61 → 0.93** at Batch 16, demonstrating that condition-driven retraining outperforms scheduled retraining while minimizing operational overhead.

### Agent Decision Distribution (Section 6.4)

| Action | Batches | Share |
|---|---|---|
| Continue            | 8 | 44.4% |
| Warning             | 5 | 27.8% |
| Tune Hyperparameters| 2 | 11.1% |
| Retrain             | 2 | 11.1% |
| Manual Review       | 1 | 5.6%  |

---

## Citation

If you use this code or framework, please cite:

```bibtex
@inproceedings{nasim2026selfhealing,
  title     = {Self-Healing Machine Learning Pipelines Using Agentic AI:
               A Framework for Autonomous Model Monitoring and Retraining},
  author    = {Nasim, Mohammad and Yenuga, Harika and Itauma, Itauma},
  booktitle = {International Business Analytics Conference for Academic and
               Industry Professionals (IBAC)},
  volume    = {1},
  number    = {1},
  year      = {2026},
  address   = {Midland, MI, USA}
}
```

---

## Authors

- **Mohammad Nasim** — Senior AI Solution Architect, Ph.D. in Computer Science. Specializes in Agentic AI, Generative AI, RAG pipelines, and multi-agent orchestration. Adjunct faculty. Email: mnasimsiddiqui@gmail.com
- **Harika Yenuga** — AI/ML Engineer with 8+ years across finance, retail, and enterprise. M.S. in Business Analytics.
- **Itauma Itauma** — Analytics Professor working at the intersection of data science, AI, and education.

---

## Future Work

Directions outlined in Section VIII of the paper:

- Replace the rule-based agent with **reinforcement learning** or **LLM-driven** decision policies.
- Move from batch simulation to **real-time streaming** (Kafka, Spark Streaming).
- Add **multivariate / deep-learning-based drift detectors**.
- Integrate **explainability** for retraining and escalation decisions (regulatory compliance in finance/healthcare).
- Validate across **additional domains** — cybersecurity, healthcare diagnostics, IoT.
