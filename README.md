# Credit Default Prediction — Risk Modeling Case Study

A end-to-end credit risk model built to predict loan default and evaluate whether it is suitable for underwriting decisions. The project covers data preprocessing, feature engineering, model development, and a production readiness assessment.

**Verdict: Not recommended for deployment as a primary underwriting model.** See [Conclusion](#5-conclusion) for details.

---

## Project Overview

| | |
|---|---|
| **Task** | Binary classification — predict probability of credit default |
| **Models** | Logistic Regression, XGBoost (with automated hyperparameter tuning via FLAML) |
| **Key techniques** | Time-based train/test split, inverse probability weighting (IPW) for selection bias, SHAP feature importance |
| **Primary metrics** | PR-AUC, KS Statistic (chosen over accuracy due to class imbalance) |

---

## Repository Structure

```
├── src/
│   ├── preprocessing.py       # Data loading, merging, type casting
│   └── modeling.py            # Model classes, tuner, evaluation utilities
├── credit_default_modeling.ipynb  # Full analysis notebook
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Data Challenges Addressed

**Selection bias** — Default labels are only observed for approved applicants (`CARDHLDR=1`), not the full applicant pool. To correct for this, a propensity model estimates each approved applicant's probability of approval, and inverse probability weights (IPW) are applied during training to approximate performance on the broader population.

**Temporal leakage** — A time-based train/test split (85th percentile of application date as cutoff) ensures no future information leaks into training. Standard random splits would overstate performance in a real deployment setting.

**Data leakage** — `SPENDING` is only observed post-approval and was excluded. `AGE` was replaced with `AGE_AT_APPLICATION` derived from application date and birth year to reflect information available at decision time.

### 2. Feature Engineering

- `AGE_AT_APPLICATION` — age at time of application (clipped at 18, the minimum legal age)
- `INCOME_PER_DEP` — monthly income divided by dependents, capturing financial burden
- `INCOME_LOG`, `ACADMOS_LOG` — log transforms to reduce right skew
- `MAJORDRG_MAX`, `MINORDRG_MAX` — worst derogatory behavior across 12-month lag window
- `MAJORDRG_AVG_NONZERO`, `MINORDRG_AVG_NONZERO` — average severity of derogatory events when they occur

Highly correlated features (`ANNUAL_INCOME` / `MONTHLY_INCOME`) and near-zero variance features were removed prior to modeling.

### 3. Model Development

Both logistic regression and XGBoost were trained and compared. XGBoost was selected as the primary model based on ranking performance. Hyperparameter tuning was performed using FLAML with a holdout validation strategy. A reduced model using only the top 4 SHAP-ranked features was also evaluated to assess whether a simpler model could match full-feature performance.

---

## Results

| Model | ROC-AUC | PR-AUC | KS Statistic |
|---|---|---|---|
| Logistic Regression | ~0.56 | ~0.10 | ~0.13 |
| XGBoost (tuned, full features) | ~0.58 | ~0.12 | ~0.16 |
| XGBoost (top 4 SHAP features) | comparable | comparable | comparable |

Baseline PR-AUC (predict all positive) ≈ 0.09 (prevalence rate).

---

## 5. Conclusion

The tuned XGBoost model achieves a ROC-AUC of ~0.58 and a PR-AUC of ~0.12 — approximately 33% above the baseline prevalence rate, but still insufficient for deployment as a primary underwriting model. The KS statistic of ~0.16 indicates modest discriminatory power.

**Why not deploy:**
- A ROC-AUC of 0.58 is borderline. Production credit models typically require 0.70+ for underwriting decisions.
- Weak mutual information scores across features suggest the available data is fundamentally limited in predictive signal for this target.
- The IPW correction, while methodologically appropriate, is only as reliable as the propensity model — which itself faces the same data constraints.

**Recommended alternative use:** Given that the model has *some* discriminatory power, it could serve as a risk scoring layer to flag applications for manual review, rather than as an automated approval/denial engine.

**To improve performance:** Additional features with stronger predictive signal (bureau data, behavioral data, alternative data sources) or a larger dataset to support more complex modeling would be the primary levers.

---

## Setup

```bash
pip install -r requirements.txt
```

Data files are not included in this repository (proprietary take-home dataset). To reproduce, supply two CSV files at `data/interview homework file A.csv` and `data/interview homework file B.csv` with the expected schema.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-blue)
![SHAP](https://img.shields.io/badge/SHAP-0.51-green)
![FLAML](https://img.shields.io/badge/FLAML-2.5-purple)
