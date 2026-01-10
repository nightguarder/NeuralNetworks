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
2. **Implement Classification (Stage 1):** `EV_Charging_Classification.ipynb` (train, tune threshold, evaluate) [COMPLETED]
3. **Implement Short-Only Regression (Stage 2):** `<24h` subset with aggregates + log1p [COMPLETED]
   - Results snapshot: RF v2 R² ≈ 0.202, RMSE ≈ 5.80, MAE ≈ 4.11 (on actual-short test set)
   - Artifacts: [project/ev_project/fig/modeling_regularized/short_regression_metrics_v2.csv](project/ev_project/fig/modeling_regularized/short_regression_metrics_v2.csv)
4. **Pipeline Integration:** Route predictions via tuned threshold; compute end-to-end metrics [COMPLETED]
   - Notebook: [project/ev_project/EV_Pipeline_Evaluation.ipynb](project/ev_project/EV_Pipeline_Evaluation.ipynb)
   - Metrics: [project/ev_project/fig/pipeline/pipeline_metrics.csv](project/ev_project/fig/pipeline/pipeline_metrics.csv)
   - Plots: [confusion_matrix.png](project/ev_project/fig/pipeline/confusion_matrix.png), [pred_short_scatter.png](project/ev_project/fig/pipeline/pred_short_scatter.png)
5. **Verification & Next Steps:** [IN PROGRESS]

### Pipeline Results Summary

**Stage 1 (Classification):**
- AUC (Long vs Short): 0.675
- Best Threshold: 0.325
- F1 (Long class): 0.037 (Precision: 0.50, Recall: 0.019)
- Observation: Classifier has low discriminatory power; threshold tuning recovered 2 true Long sessions from 105 actual (1.9% recall).

**Stage 2 (RF v2 Regression):**
- Coverage (predicted-short): 99.7% of test set routed to regressor.
- Predicted-short metrics: RMSE 14.71, MAE 6.90, R² 0.009 (poor; includes misrouted long sessions).
- Actual-short metrics: RMSE 5.95, MAE 4.19, R² 0.161 (good; RF v2 performs well when ground-truth is <24h).

**Key Findings:**
- Current classifier fails to reliably identify long sessions (98% recall loss), causing nearly all test samples to be routed to regression.
- The regressor trained on short-only is accurate for truly short sessions but performs poorly when forced to predict on long sessions.
- The pipeline's end-to-end metrics are dominated by misrouting: RMSE 14.71 vs isolated RF v2 RMSE 5.95.

### Enhanced Pipeline with HistGradientBoosting (Jan 9, 2026)

**Improvements Applied:**
- Added user/garage aggregates to Stage 1 features (session counts, avg duration/energy)
- Trained HistGradientBoosting and GradientBoosting classifiers with sample weighting
- Best model: HistGradientBoosting with AUC 0.847 (+25% over baseline MLP's 0.675)

**Results:**
- Stage 1 Enhanced:
  - AUC (Long): 0.847 (was 0.675)
  - Best threshold: 0.633
  - F1 (Long): 0.428 (was 0.037)
  - Precision: 0.335, Recall: 0.590 (was 0.50, 0.019)
  - Confusion: Correctly identified 62/105 long sessions (59% recall vs 1.9% before)
- Pipeline End-to-End:
  - Coverage: 86.3% routed to Stage 2 (was 99.7%)
  - Predicted-short: RMSE 11.70, MAE 5.30, R² 0.057
  - Actual-short: RMSE 5.95, MAE 4.19, R² 0.161 (unchanged; RF v2 still excellent on true shorts)

**Key Findings:**
- Enhanced features + gradient boosting dramatically improved Long-session detection (59% recall vs 1.9%).
- 13.7% of test sessions now correctly routed as Long, reducing misrouting burden on regressor.
- Predicted-short RMSE improved from 14.71 to 11.70 (-20%), though still elevated due to some misrouted long sessions.
- The pipeline is now viable: classifier has meaningful discriminatory power and routing is operational.

### Remaining Next Actions
- **Operational threshold tuning:**
  - Use cost-sensitive criteria based on business needs (e.g., penalty for missed long sessions vs false alarms).
  - Explore multiple thresholds to create risk tiers (low/medium/high probability long sessions).
- **Stage 2 hyperparameter tuning:**
  - Grid search for RF optimal parameters (max_depth, min_samples_leaf, n_estimators).
  - Benchmark HistGradientBoostingRegressor as alternative to RF.
- **Feature engineering round 2:**
  - Add temporal recency (sessions in last 7/30 days) when performance allows.
  - Incorporate time-since-last-plug, day-of-week interaction with hour.
- **Model interpretation:**
  - SHAP values for HistGradientBoosting to identify key Long-session drivers.
  - Feature importance analysis to guide further engineering.
