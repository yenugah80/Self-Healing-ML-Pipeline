<div align="center">

<img src="architecture.png" alt="Self-Healing ML Pipeline" width="100%"/>

# 🧠 Self-Healing ML Pipeline

### *Autonomous Model Monitoring, Drift Detection & Retraining Using Agentic AI*

<br/>

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![pandas](https://img.shields.io/badge/pandas-2.2.3-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)

[![Conference](https://img.shields.io/badge/IBAC_2026-Published-8A2BE2?style=for-the-badge&logo=academia&logoColor=white)](#citation)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle_Credit_Fraud-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Pipeline-Production_Ready-22C55E?style=for-the-badge&logo=checkmarx&logoColor=white)](#)

[![F1 Score](https://img.shields.io/badge/Best_F1-0.8157-FF69B4?style=flat-square&logo=target&logoColor=white)](#baseline-results)
[![ROC-AUC](https://img.shields.io/badge/Best_ROC--AUC-0.9732-FF8C00?style=flat-square)](#baseline-results)
[![Healing Rate](https://img.shields.io/badge/Self--Healing_Rate-100%25-00C896?style=flat-square&logo=automate&logoColor=white)](#self-healing-impact)
[![Improvement](https://img.shields.io/badge/F1_Improvement-+4.54%25-1D9BF0?style=flat-square)](#self-healing-impact)
[![Batches](https://img.shields.io/badge/Streaming_Batches-18-9333EA?style=flat-square)](#drift-simulation)

<br/>

> **Self-Healing Machine Learning Pipelines Using Agentic AI: A Framework for Autonomous Model Monitoring and Retraining**
> *Mohammad Nasim · Harika Yenuga · Itauma Itauma*
> International Business Analytics Conference for Academic and Industry Professionals (IBAC), Vol. 01, Issue 01 — May 2026

<br/>

</div>

---

## 📌 Table of Contents

| | Section |
|---|---|
| 🎯 | [Overview](#-overview) |
| 🏗️ | [Architecture](#%EF%B8%8F-architecture) |
| 📊 | [Key Results](#-key-results) |
| 🤖 | [Agent Decision Engine](#-agent-decision-engine) |
| ⚡ | [Quick Start](#-quick-start) |
| 🗂️ | [Project Structure](#%EF%B8%8F-project-structure) |
| 🌊 | [Drift Simulation](#-drift-simulation) |
| 👥 | [Authors](#-authors) |
| 📄 | [Citation](#-citation) |

---

## 🎯 Overview

Machine learning models in production **degrade silently** — as data distributions shift, fraud patterns evolve, and real-world behaviour diverges from training assumptions. Traditional MLOps pipelines rely on fixed retraining schedules or manual intervention, both of which are reactive, expensive, and error-prone.

This project implements a **closed-loop, self-healing ML pipeline** that:

```
📥 Ingest streaming batches
        │
        ▼
📡 Monitor performance (F1, Recall, ROC-AUC) + KL divergence drift
        │
        ▼
🧠 Agentic Decision Controller → Continue / Warn / Tune / Retrain / Escalate
        │
        ▼
🔧 Autonomously retrain & select the best candidate model
        │
        ▼
🚀 Deploy improved model → feedback loop
```

Evaluated on the **Credit Card Fraud Detection** dataset (284,807 transactions, 0.17% positive class) across **18 simulated streaming batches** with escalating drift strengths.

---

## 🏗️ Architecture

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SELF-HEALING ML PIPELINE                         │
│                                                                     │
│  ┌───────────┐    ┌────────────┐    ┌──────────────────────────┐   │
│  │  STREAMING │───▶│ MONITORING │───▶│  AGENTIC CONTROLLER      │   │
│  │  BATCHES  │    │  MODULE    │    │  (Rule-Based Agent)      │   │
│  │           │    │            │    │                          │   │
│  │ 5000 tx   │    │ • F1 Score │    │ KL ≥ 0.10 → Manual Rev  │   │
│  │ per batch │    │ • Recall   │    │ F1↓ + KL↑ → Retrain     │   │
│  │           │    │ • ROC-AUC  │    │ F1↓ + KL↓ → Tune HPs   │   │
│  │ drift=0.0 │    │ • KL Div   │    │ F1✓ + KL↑ → Warning     │   │
│  │ drift=0.3 │    │            │    │ F1✓ + KL↓ → Continue    │   │
│  │ drift=0.7 │    └────────────┘    └────────────┬─────────────┘   │
│  └───────────┘                                   │                 │
│                                                  ▼                 │
│  ┌──────────────────┐    ┌──────────────────────────────────────┐  │
│  │  MODEL REGISTRY  │◀───│  RETRAINING ENGINE                   │  │
│  │                  │    │                                      │  │
│  │ • LR  baseline   │    │  Candidate Models:                   │  │
│  │ • RF  ← current  │    │  ├─ Logistic Regression              │  │
│  │ • XGB candidate  │    │  ├─ Random Forest  ✓ (winner)        │  │
│  └──────────────────┘    │  └─ XGBoost                          │  │
│          │               └──────────────────────────────────────┘  │
│          ▼                                                          │
│  ┌──────────────────┐                                              │
│  │   DEPLOYMENT     │──────────────── feedback loop ──────────────▶│
│  │   MODULE         │                                              │
│  └──────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────┘
```

</div>

---

## 📊 Key Results

### Baseline Results

> Three models trained on 199,364 transactions, evaluated on 85,443 holdout transactions.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 97.86% | 0.0670 | 0.8784 | 0.1245 | 0.9680 |
| **Random Forest** ⭐ | **99.94%** | **0.9720** | **0.7027** | **0.8157** | 0.9275 |
| XGBoost | 99.74% | 0.3853 | 0.8514 | 0.5305 | **0.9732** |

> ⭐ Random Forest selected as the initial production model — highest F1 on this imbalanced dataset.

<br/>

### Self-Healing Impact

> Pipeline ran across 18 batches with three drift phases. Self-healing triggered autonomously.

<div align="center">

| Metric | Value |
|:-------|:-----:|
| 📈 Average F1 **before** healing | `0.7255` |
| 🚀 Average F1 **after** healing | `0.7709` |
| ✨ Net F1 improvement | **`+0.0454`** |
| 🔄 Retrain events triggered | **3** (Batches 12, 15, 16) |
| ✅ Self-healing success rate | **100%** (3/3) |

</div>

<br/>

**Per-Healing-Event Breakdown:**

| Batch | Drift Strength | F1 Before | F1 After | Recovery |
|:-----:|:--------------:|:---------:|:--------:|:--------:|
| 12 | 0.7 (severe) | 0.6667 | 0.8571 | **+28.6%** 🟢 |
| 15 | 0.7 (severe) | 0.6667 | 0.9091 | **+36.4%** 🟢 |
| 16 | 0.7 (severe) | 0.6154 | **1.0000** | **+62.5%** 🟢 |

<br/>

### Agent Decision Distribution

> 18 batches processed across all three drift phases.

| Action | Count | Share | Visual |
|:-------|:-----:|:-----:|:-------|
| ✅ Continue | 8 | 44.4% | `████████████████████░░░░░░░░░░░░░░░░░░░░░` |
| ⚠️ Warning | 4 | 22.2% | `██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |
| 🔄 Retrain | 3 | 16.7% | `███████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |
| 🔧 Tune Hyperparameters | 2 | 11.1% | `█████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |
| 🚨 Manual Review | 1 | 5.6% | `██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░` |

---

## 🤖 Agent Decision Engine

The agentic controller maps `(F1-Score, KL-Divergence)` pairs to one of five actions using deterministic rules derived from the paper (Section 5.7):

```
                    KL < 0.05              KL ∈ [0.05, 0.10)         KL ≥ 0.10
                ┌──────────────────┬───────────────────────────┬──────────────────┐
  F1 ≥ 0.70    │   ✅ CONTINUE    │      ⚠️  WARNING           │  🚨 MANUAL REVIEW │
                │                  │   (monitor closely)        │  (human-in-loop) │
                ├──────────────────┼───────────────────────────┤                  │
  F1 < 0.70    │  🔧 TUNE HPs     │      🔄 RETRAIN            │                  │
                │  (no drift,      │   (self-healing fires,     │                  │
                │   perf issue)    │    3 candidates evaluated) │                  │
                └──────────────────┴───────────────────────────┴──────────────────┘
```

**Retraining Engine** — when `Retrain` is triggered:
1. Appends the drifted batch to the dynamic training pool
2. Trains three candidates: Logistic Regression, Random Forest, XGBoost
3. Evaluates each on a held-out 20% validation split
4. Promotes the winner back to production — all autonomously, no human needed

---

## ⚡ Quick Start

### Prerequisites

```bash
git clone https://github.com/yenugah80/Self_Healing_ML_Pipeline.git
cd Self_Healing_ML_Pipeline
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Dataset

The pipeline uses the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (~144 MB, not committed due to GitHub size limits):

```bash
# Option A — Kaggle CLI
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip

# Option B — Manual
# Download creditcard.csv from the Kaggle link above and place it in the project root.
```

### Run the Full Pipeline

```bash
python step1_baseline_experiment.py         # → step1_baseline_results.csv
python step2_drift_monitoring.py            # → step2_drift_monitoring_results.csv
python step3_agentic_decision_controller.py # → step3_agentic_decision_results.csv
python step4_self_healing_retraining.py     # → step4_self_healing_results.csv
                                            #   step4_candidate_model_log.csv
python step5_results_visualization.py       # → chart1..chart5 PNGs
```

> **Runtime:** Steps 1, 2, 4 each take 2–5 minutes depending on hardware (Random Forest on 200K rows). Step 5 is near-instant.

---

## 🌊 Drift Simulation

The test set is split into **18 sequential batches of 5,000 transactions** each. Gaussian noise is injected into features `V1, V2, V3, V4, V10, V11, V12, V14, Amount`:

| Phase | Batches | Drift Strength | Noise Distribution | What Happens |
|:------|:-------:|:--------------:|:------------------:|:-------------|
| 🟢 Stable | 1 – 5 | `0.0` | None | Baseline performance |
| 🟡 Moderate | 6 – 10 | `0.3` | `N(0.3, 0.3)` | KL rises, F1 mostly holds |
| 🔴 Severe | 11 – 18 | `0.7` | `N(0.7, 0.7)` | KL > 0.05, self-healing triggers |

---

## 🗂️ Project Structure

```
Self_Healing_ML_Pipeline/
│
├── 🐍 step1_baseline_experiment.py          # Train LR, RF, XGBoost baselines
├── 🐍 step2_drift_monitoring.py             # Simulate batches, compute KL divergence
├── 🐍 step3_agentic_decision_controller.py  # Apply rule-based agent to each batch
├── 🐍 step4_self_healing_retraining.py      # Trigger retraining + model selection
├── 🐍 step5_results_visualization.py        # Generate charts (Figures 2–6 in paper)
│
├── 📊 step1_baseline_results.csv            # Baseline metrics for 3 models
├── 📊 step2_drift_monitoring_results.csv    # Per-batch KL drift scores
├── 📊 step3_agentic_decision_results.csv    # Per-batch agent decisions + reasons
├── 📊 step4_candidate_model_log.csv         # Candidate models during retraining
├── 📊 step4_self_healing_results.csv        # Before / after F1 + healing flag
│
├── 🖼️ architecture.png                      # Figure 1 — System architecture
├── 🖼️ chart1_baseline_f1.png               # Figure 2 — Baseline F1 comparison
├── 🖼️ chart2_f1_trend.png                  # Figure 3 — F1 trend with self-healing
├── 🖼️ chart3_drift_vs_f1.png               # Figure 4 — KL drift vs. F1
├── 🖼️ chart4_agent_distribution.png        # Figure 5 — Agent action distribution
├── 🖼️ chart5_self_healing.png              # Figure 6 — Before vs. after retraining
│
├── 📋 requirements.txt
└── 📖 README.md
```

---

## 👥 Authors

<div align="center">

| | Author | Role & Affiliation |
|---|:---|:---|
| 🧑‍💻 | **Mohammad Nasim** | Senior AI Solution Architect · Ph.D. Computer Science · Adjunct Faculty, Northwood University · Specializes in Agentic AI, RAG, multi-agent orchestration |
| 👩‍💻 | **Harika Yenuga** [![GitHub](https://img.shields.io/badge/@yenugah80-181717?style=flat-square&logo=github)](https://github.com/yenugah80) | AI/ML Engineer · 8+ years in finance, retail & enterprise · M.S. Business Analytics, Northwood University |
| 👨‍🏫 | **Itauma Itauma** | Analytics Professor · Northwood University · Data Science, AI & Education |

</div>

---

## 🔭 Future Work

<details>
<summary><strong>Click to expand roadmap</strong></summary>

| Direction | Description |
|:----------|:------------|
| 🤖 LLM-driven agent | Replace rule-based controller with a GPT/Claude policy that reasons over context and history |
| 🌊 Real-time streaming | Move from batch simulation to live Kafka / Spark Streaming ingestion |
| 🔍 Deep drift detection | Multivariate and autoencoder-based drift detectors beyond KL divergence |
| 🔎 Explainability | SHAP / LIME integration for retraining and escalation decisions (regulatory compliance) |
| 🌐 Cross-domain validation | Cybersecurity intrusion detection, healthcare diagnostics, IoT anomaly detection |
| 🔁 Reinforcement learning | Train a reward-driven agent to minimize cumulative performance degradation |

</details>

---

## 📄 Citation

If you use this code or framework in your research, please cite:

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

<div align="center">

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Northwood University](https://img.shields.io/badge/Northwood_University-Midland,_MI-003087?style=for-the-badge)](https://northwood.edu)
[![IBAC 2026](https://img.shields.io/badge/IBAC_2026-Peer_Reviewed-8A2BE2?style=for-the-badge)](https://ibacconference.com)

*© 2026 Mohammad Nasim, Harika Yenuga, Itauma Itauma — Northwood University*

</div>
