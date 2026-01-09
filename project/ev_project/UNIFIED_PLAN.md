# UNIFIED PLAN: EV Project Restart & Classification Strategy

**Date:** December 10, 2025  
**Project:** EV Charging Duration Prediction (Trondheim)  
**Status:** 🟢 Restarting with Robust Data Pipeline & Classification Focus

---

## Executive Summary: The "Fresh Start"

We are executing a strategic **Project Restart**. Previous iterations revealed two critical insights:
1. **Data Pipeline Issues:** The merging of Session and Weather data was potentially lossy or misaligned. We are fixing this from the ground up.
2. **Regression Viability:** Predicting exact duration for extreme outliers (>24h) is statistically infeasible with this dataset due to "Regression to the Mean" on sparse tail data.

**New Direction:**
We are shifting from **Regression** (predicting hours) to **Probabilistic Binary Classification** (predicting probability of "Long Session" ≥ 24h). This aligns better with the data distribution and provides more actionable insights for grid operators.

---

## Part 1: Robust Data Pipeline (The Fix)

We are rewriting `EV_Data_Cleaning_and_Preparation.ipynb` to ensure valid data merging.

### 1.1 The Merge Strategy
- **Primary Source:** `Dataset 1_EV charging reports.csv` (Sessions)
- **Secondary Source:** `Norway_Trondheim_ExactLoc_Weather.csv` (Weather)
- **Join Key:** Date (`Start_plugin` date -> Weather `date`)
- **Join Type:** **Left Join** (Keep all sessions, attach weather where available)
- **Validation:**
  - Asset `merged_rows == original_rows`
  - Assert `weather_nulls` are minimal (< 5%)

### 1.2 Target Definition
We define a new binary target `is_short_session`:
- **Class 1 (Short):** Duration < 24 hours (Majority, ~73%)
- **Class 0 (Long):** Duration ≥ 24 hours (Minority, ~27%)

*Note: We continue to clean timestamps and remove physical impossibilities (duration < 0.05h).*

---

## Part 2: Classification Strategy (The Solution)

### 2.1 Why Classification?
- **Data Balance:** 73/27 split is manageable with class weights, unlike the 96/4 split for regression outliers.
- **Actionability:** A probability score (e.g., "85% chance this car stays > 24h") is more valuable for capacity planning than a noisy regression estimate.

### 2.2 Model Architecture
- **Input:** Standardized features (Time, Location, Weather).
- **Model:** Neural Network (MLP) with Regularization (Dropout, L2).
- **Output:** Sigmoid (Probability [0, 1]).
- **Loss:** Binary Crossentropy.
- **Metrics:** ROC-AUC, Precision, Recall (Class 0), Accuracy.

---

## Part 3: Historical Context (Regression Lessons)
 - **Loss:** Binary Crossentropy.
 - **Metrics:** ROC-AUC, Precision, Recall (Class 0), Accuracy.

## Part 2.5: Two-Stage Pipeline (Classifier → Short-Session Regression)

We will implement an end-to-end pipeline that first classifies whether a session is likely "Long" (≥ 24h) and, only for predicted "Short" (< 24h) sessions, runs a regression model to estimate hours (and optionally energy). This preserves full-scope coverage while avoiding tail-induced bias in regression.

### Stage 1: Binary Classification (≥ 24h vs < 24h)
- Input: Time, location, weather, and engineered features (hour_sin/cos, weekday, Garage_ID).
- Model: Regularized MLP (Dropout, L2, BatchNorm) with class weights; monitor AUC.
- Output: Probability `p_long` in [0, 1] for "Long" (≥ 24h).
- Threshold: Tune decision threshold `τ` via ROC/PR analysis (default 0.5; consider cost-weighted selection).
- Calibration: Optional probability calibration for more reliable `p_long` (e.g., Platt scaling).

### Stage 2: Short-Session Regression (< 24h)
- Routing: If `p_long < τ`, route to short-session regressor.
- Scope: Train only on sessions with `Duration_hours < 24` (reduces tail variance).
- Models: Compare Regularized MLP, Random Forest, and XGBoost; pick best by validation R²/RMSE/MAE.
- Features: Same as classifier, plus weather and user-level aggregates (avg duration/energy, frequency).
- Loss/Training: Early stopping, ReduceLROnPlateau; log1p transform optional for stability.

### Inference Flow
1. Compute features → Stage 1 classifier → `p_long`.
2. If `p_long ≥ τ` → Output: "Long" classification with probability and optional duration band heuristic.
3. If `p_long < τ` → Stage 2 regressor → Output: Predicted duration hours (and energy, if needed).

### Evaluation & Reporting
- Classification (Stage 1): AUC, Accuracy, Precision/Recall/F1 (focus on Long class), confusion matrix.
- Regression (Stage 2): RMSE, MAE, R² on short-only test set; residual analysis.
- Pipeline-level metrics: Composite evaluation on full dataset using routing decisions; report separate performance for (<24h, ≥24h) strata.
- Transparency: Document that regression is domain-limited to short sessions by design; provide operational guidance for long-session handling.

### Operational Notes
- Threshold tuning: Optimize `τ` based on false-negative vs false-positive costs for long stays.
- Robustness: Use chronological splits and month-wise validation; monitor drift.
- Extensibility: Optional second classifier for ≥ 40h to flag extra-long stays.

---

## Part 3: Historical Context (Regression Lessons)

*Retained for Academic Completeness*

We previously attempted a Pure Regression approach (`EV_Modeling_Regularized.ipynb`).
- **Result:** R² = 0.59 (Good for short sessions, failed for long sessions).
- **Diagnosis:** The model learned to predict the mean for tail events.
- **Status:** This notebook is preserved to demonstrate "What didn't work" and why the pivot to classification is justified.

---

## Roadmap (Updated Jan 9, 2026)

1. **Fix Data:** Overwrite `EV_Data_Cleaning_and_Preparation.ipynb` [COMPLETED]
2. **Implement Classification (Stage 1):** `EV_Charging_Classification.ipynb` (train, tune threshold, evaluate) [IN PROGRESS]
3. **Implement Short-Only Regression (Stage 2):** `<24h` subset with aggregates + log1p [COMPLETED]
  - Results snapshot: RF v2 R² ≈ 0.202, RMSE ≈ 5.80, MAE ≈ 4.11
  - Artifacts: [project/ev_project/fig/modeling_regularized/short_regression_metrics_v2.csv](project/ev_project/fig/modeling_regularized/short_regression_metrics_v2.csv)
4. **Pipeline Integration:** Route predictions via tuned threshold; compute end-to-end metrics [PENDING]
5. **Verification:** Compare pipeline (AUC + short-only R²/RMSE) vs pure regression baselines; document trade-offs [PENDING]

### Immediate Next Actions
- Re-optimize Stage 1 threshold using cost-sensitive criteria and consider probability calibration.
- Implement pipeline evaluation notebook to combine Stage 1 routing with Stage 2 RF v2 regressor.
- Add recency-based features and hyperparameter tuning for RF; benchmark `HistGradientBoostingRegressor`.
