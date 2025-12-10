# Modeling Results: EV Charging Prediction

Date: December 3, 2025

This report summarizes model performance for predicting Energy (El_kWh) and Duration (Duration_hours). Models trained on the cleaned dataset using a chronological 80/20 split.

---

## Models Compared

- Ridge Regression (linear baseline)
- Random Forest Regressor (tree-based)
- XGBoost Regressor (gradient boosting)
- Keras Neural Network (MLP with 2 hidden layers)

Features include cyclical hour encodings, categorical one-hot features (weekday, month, garage, plugin/duration categories), and duration (for energy prediction only).

---

## Key Results (Test Set)

See CSV: `project/ev_project/fig/modeling/metrics.csv`

Figures (saved to `project/ev_project/fig/modeling/`):

- Energy:
  - `energy_rf_pred_vs_actual.png`
  - `energy_rf_residuals.png`
  - `energy_xgb_pred_vs_actual.png`
  - `energy_xgb_residuals.png`
  - `energy_nn_pred_vs_actual.png`
  - `energy_nn_residuals.png`
- Duration:
  - `duration_rf_pred_vs_actual.png`
  - `duration_rf_residuals.png`
  - `duration_xgb_pred_vs_actual.png`
  - `duration_xgb_residuals.png`
  - `duration_nn_pred_vs_actual.png`
  - `duration_nn_residuals.png`

---

## Interpretation

- Tree-based models (Random Forest, XGBoost) and Neural Networks outperform linear baselines on both targets due to non-linear temporal and categorical interactions.
- **Best Duration Predictor:** Keras NN (R² = 0.61, RMSE = 8.38 hours) slightly outperforms Random Forest (R² = 0.60).
- **Best Energy Predictor:** Random Forest (R² = 0.24, RMSE = 10.41 kWh) with XGBoost close behind.
- Including `Duration_hours` as a predictor substantially improves Energy prediction; we avoid using `El_kWh` in Duration models to prevent leakage.
- Neural Network shows competitive performance but needs regularization (dropout, early stopping) to potentially exceed tree-based models.

---

## Robustness (Month-wise)

CSV summaries:

- `monthwise_energy_metrics.csv`
- `monthwise_duration_metrics.csv`

Check February specifically for stability; no rebalancing was applied during cleaning.

---

## Next Steps

### Immediate (Following Lecture 4: Regularization)

- **Apply dropout layers** (0.2-0.5) to Neural Network to reduce overfitting.
- **Implement early stopping** callback to prevent overtraining.
- **Add L1/L2 regularization** to dense layers.
- **Test batch normalization** for training stability.

### Feature Engineering

- Add user/garage aggregation features (rolling means, counts) to capture behavioral histories.
- Add weather features (temp, precip, solar, wind) to improve performance and seasonal robustness.
- Create lag features (previous session energy/duration).

### Advanced Models

- Implement LSTM/RNN for sequence-based prediction (following course syllabus).
- Try LightGBM and perform hyperparameter tuning with time-series CV.
- Add SHAP/Permutation importance for interpretability.

### Validation

- Extend month-wise validation to all 13 months.
- Implement 5-fold time-series cross-validation.
- Test model robustness on February data specifically.
