# EV Charging Behavior Prediction Project

**Status:** 🟢 Active Development  
**Phase:** Classical ML Complete | Regularization Ready for Execution  
**Last Updated:** December 10, 2025

---

## 📋 Project Overview

This project predicts Electric Vehicle (EV) charging behavior using machine learning techniques. We analyze 6,880 charging sessions from Trondheim, Norway (Dec 2018 - Dec 2019) to forecast:

1. **Energy Consumption** (El_kWh) — How much energy will be consumed
2. **Charging Duration** (Duration_hours) — How long the vehicle will charge

### Business Value

- **Grid operators:** Optimize energy procurement and load balancing
- **Station operators:** Improve resource allocation and reduce congestion
- **Users:** Better availability predictions and charging recommendations

---

## 📂 Project Structure

```
ev_project/
├── README.md                          # This file
├── 1_Data_analysis.md                 # Initial EDA findings
├── 2_Data_cleanup.md                  # Data cleaning documentation
├── 3_Modeling_Results.md              # Baseline model results
├── 4_Regularization_Results.md        # 🆕 Regularization techniques & results
├── EV_Charging_Data_Analysis.ipynb    # Exploratory analysis notebook
├── EV_Data_Cleaning_and_Preparation.ipynb  # Data preprocessing
├── EV_Modeling.ipynb                  # Baseline modeling (RF, XGB, NN)
├── EV_Modeling_Regularized.ipynb      # 🎯 Regularized NN models (Lecture 4)
├── data/
│   └── ev_sessions_clean.csv          # Cleaned dataset (6,880 sessions)
└── fig/
    ├── energy_consumption_graph.png
    ├── charging_duration_graph.png
    ├── modeling/                       # Baseline model plots
    │   ├── metrics.csv
    │   ├── monthwise_*_metrics.csv
    │   ├── *_pred_vs_actual.png
    │   └── *_residuals.png
    └── modeling_regularized/           # 🆕 Regularization results
        ├── regularized_metrics.csv
        ├── all_models_comparison.csv
        ├── *_training_history.png
        ├── *_predictions_comparison.png
        └── *_residuals_comparison.png
```

---

## 🎯 Current Status & Results

### ✅ Phase 1: Classical ML + Baseline NN (Complete)

**Models Implemented:**

- Ridge Regression (baseline)
- Random Forest Regressor
- XGBoost Regressor
- Keras Neural Network (MLP - no regularization)

**Best Performance (Test Set):**

| Target       | Best Model    | RMSE        | MAE        | R²       | Status      |
| ------------ | ------------- | ----------- | ---------- | -------- | ----------- |
| **Duration** | Keras NN      | 8.38 hours  | 3.25 hours | **0.61** | ✅ Good     |
| **Duration** | Random Forest | 11.38 hours | 3.45 hours | 0.60     | ✅ Good     |
| **Energy**   | Random Forest | 10.41 kWh   | 6.59 kWh   | **0.24** | ⚠️ Moderate |
| **Energy**   | XGBoost       | 10.96 kWh   | 7.01 kWh   | 0.15     | ⚠️ Moderate |

**Key Findings:**

- ✅ Tree-based models outperform linear baselines
- ✅ Duration prediction is more accurate than energy (R² = 0.60 vs 0.24)
- ✅ Month-wise validation shows stable performance (Dec-Jan)
- ⚠️ Energy prediction needs improvement (consider adding duration as feature)

**Generated Artifacts:**

- 12 visualization plots (pred vs actual, residuals for RF/XGB/NN)
- CSV metrics for overall and month-wise performance
- Complete model comparison table

---

## 🔬 Data Summary

### Dataset Characteristics

- **Sessions:** 6,880 charging events
- **Period:** December 2018 - December 2019 (13 months)
- **Locations:** 42 unique garages in Trondheim
- **Users:** 127 unique private EV owners
- **Data Quality:** Excellent (>99.9% complete)

### Target Variables

**Energy Consumption (El_kWh):**

- Mean: 14.23 kWh | Median: 11.85 kWh | Std: 10.47 kWh
- Range: 0.01 - 77.88 kWh
- Distribution: Right-skewed (mean > median)

**Charging Duration (Duration_hours):**

- Mean: 11.50 hours | Median: 10.03 hours | Std: 14.15 hours
- Range: 0.00 - 255.03 hours
- Distribution: Highly variable, median ~10h suggests overnight charging

### Key Predictors

**Temporal Features:**

- Hour of day (peak: 18:00-23:00)
- Day of week (weekdays > weekends)
- Month/season (winter slightly higher consumption)

**Categorical Features:**

- Garage location (42 unique locations)
- User ID (127 users with consistent patterns)
- Plugin time category (5 time ranges)

**Engineered Features:**

- Cyclical encodings (hour_sin, hour_cos for periodicity)
- One-hot encoded categories
- Duration (for energy prediction only, to avoid leakage)

**Correlation:**

- Duration ↔ Energy: r = 0.68 (strong positive)
- Temporal features show non-linear relationships

---

## 🚀 Next Steps & Roadmap

### 🎯 Immediate Priorities (Week 1-2)

#### 1. Apply Course Lecture 4 Techniques: Regularization

**Goal:** Improve Neural Network performance using regularization methods

**Actions:**

- [ ] **Dropout Layers**

  - Add dropout (0.2-0.5) between dense layers
  - Test different dropout rates
  - Compare with/without dropout on validation loss

- [ ] **L1/L2 Regularization**

  - Add kernel regularizers to dense layers
  - Test L2 (ridge) regularization
  - Monitor weight distributions

- [ ] **Early Stopping**

  - Implement `EarlyStopping` callback
  - Monitor validation loss with patience=10
  - Restore best weights automatically

- [ ] **Batch Normalization**
  - Add BatchNormalization layers
  - Test placement (before/after activation)
  - Measure impact on training stability

**Expected Outcome:** Reduce overfitting, improve generalization, achieve R² > 0.65 for duration

#### 3. Enhance Feature Engineering

**Goal:** Capture behavioral patterns and temporal context

**Actions:**

- [ ] **User Aggregation Features**

  - User average energy/duration
  - User charging frequency
  - User preferred garages/times

- [ ] **Temporal Context**

  - Lag features (previous session energy/duration)
  - Rolling averages (7-day, 30-day)
  - Time since last charge

- [ ] **Location Features**
  - Garage-level aggregations (avg energy, traffic)
  - Garage capacity/occupancy proxies

**Expected Outcome:** Capture individual behavior patterns, improve energy R² to > 0.35

---

### 🔮 Medium-Term Goals (Week 3-4)

#### 4. Incorporate External Data

**Available Datasets:**

- `Dataset 6_Local traffic distribution.csv` — 5 traffic monitoring locations
- `Norway_Trondheim_ExactLoc_Weather.csv` — 35+ weather variables

**Actions:**

- [ ] **Weather Integration**

  - Temperature (affects battery efficiency)
  - Precipitation (user behavior changes)
  - Solar radiation (renewable energy proxy)
  - Merge by date (daily → session mapping)

- [ ] **Traffic Integration**
  - Align traffic locations with garage locations
  - Use as proxy for general activity patterns
  - Test correlation with charging demand

**Expected Outcome:** Improve seasonal robustness, capture weather/activity effects

#### 5. Advanced Neural Network Architectures

**Goal:** Apply course material on RNNs/LSTMs for sequence modeling

**Rationale:** Charging behavior has temporal dependencies

- Users have habitual patterns
- Time-of-day/week seasonality
- Historical session context matters

**Actions:**

- [ ] **LSTM Model (Lecture 3 & Syllabus)**
  - Design sequence prediction architecture
  - Use user charging history as sequences
  - Predict next session energy/duration
- [ ] **Architecture Design**

  - Input: Last N charging sessions per user
  - Features: temporal + categorical + contextual
  - Output: Next session prediction
  - Apply regularization from Lecture 4

- [ ] **Comparison**
  - LSTM vs Random Forest/XGBoost
  - Sequential vs tabular approach
  - Measure improvement over best classical model

**Expected Outcome:** R² > 0.70 for duration, R² > 0.40 for energy

---

### 🎓 Long-Term Vision (Future Iterations)

#### 6. Model Optimization & Production

- [ ] **Hyperparameter Tuning**

  - Bayesian optimization
  - Cross-validation with time-series splits
  - Grid search for regularization parameters

- [ ] **Model Interpretability**

  - SHAP values for feature importance
  - Permutation importance
  - Partial dependence plots

- [ ] **Deployment Considerations**
  - Model serialization (pickle/SavedModel)
  - Inference pipeline design
  - Real-time prediction API

#### 7. Advanced Techniques

- [ ] **Ensemble Methods**

  - Stack Random Forest + XGBoost + Neural Network
  - Weighted averaging based on validation performance

- [ ] **Anomaly Detection**

  - Identify unusual charging patterns
  - Flag outliers for separate handling

- [ ] **Multi-Task Learning**
  - Single model predicting both energy and duration
  - Shared representations for correlated targets

---

## 📊 Evaluation Metrics

### Primary Metrics

- **RMSE** (Root Mean Squared Error) — Penalizes large errors
- **MAE** (Mean Absolute Error) — Average error magnitude
- **R²** (R-Squared) — Variance explained (0=baseline, 1=perfect)

### Target Thresholds

| Metric   | Duration Target | Energy Target |
| -------- | --------------- | ------------- |
| **R²**   | > 0.65          | > 0.40        |
| **RMSE** | < 2.0 hours     | < 8.0 kWh     |
| **MAE**  | < 1.5 hours     | < 5.0 kWh     |

### Validation Strategy

- **Train/Test Split:** 80/20 chronological (respects time order)
- **Month-wise Validation:** Test performance per month for robustness
- **Cross-Validation:** 5-fold time-series CV for hyperparameter tuning

---

## 🔧 Technical Stack

**Languages & Frameworks:**

- Python 3.x
- Keras/TensorFlow (Neural Networks)
- scikit-learn (Classical ML, preprocessing)
- XGBoost (Gradient Boosting)

**Data Science:**

- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib/seaborn (visualization)

**Notebooks:**

- Jupyter Lab/Notebook

**Key Libraries:**

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

---

## 📚 References & Course Alignment

This project applies concepts from:

1. **Lecture 3: Neural Networks Basics**

   - MLP architecture design
   - Activation functions (ReLU)
   - Forward/backward propagation

2. **Lecture 3 Part 2: Metaparameters**

   - Learning rate tuning
   - Early stopping implementation
   - Optimizer selection (Adam)

3. **Lecture 4: Regularization** ⭐ _Next Implementation_

   - Dropout layers
   - L1/L2 regularization
   - Batch normalization
   - Overfitting prevention

4. **Syllabus: Advanced Architectures** _Future_
   - RNNs/LSTMs for sequence modeling
   - Time series applications

---

## 🎓 Learning Outcomes

Through this project, students demonstrate:

✅ **Data Science Skills:**

- Exploratory data analysis and visualization
- Feature engineering for tabular data
- Handling temporal and categorical features

✅ **Machine Learning:**

- Model comparison and selection
- Hyperparameter tuning
- Cross-validation strategies
- Performance evaluation and interpretation

✅ **Neural Networks:**

- Keras model building (Sequential API)
- Layer configuration and optimization
- Regularization techniques
- Training callbacks

✅ **Software Engineering:**

- Jupyter notebook workflows
- Code organization and documentation
- Reproducible analysis pipeline

---

## 📝 How to Use This Project

### 1. Explore the Analysis

```bash
jupyter lab EV_Charging_Data_Analysis.ipynb
```

Review data characteristics, distributions, and correlations.

### 2. Review Data Cleaning

```bash
jupyter lab EV_Data_Cleaning_and_Preparation.ipynb
```

See preprocessing steps and feature engineering.

### 3. Run Modeling

```bash
jupyter lab EV_Modeling.ipynb
```

Train models, evaluate performance, generate visualizations.

### 4. Review Results

### 5. Review Results

- **Baseline:** Read `3_Modeling_Results.md` and check `fig/modeling/`
- **Regularized:** Read `4_Regularization_Results.md` and check `fig/modeling_regularized/`
- Compare baseline vs regularized performance

---

## 🐛 Known Issues & Limitations

1. **Energy Prediction Performance**

   - Current R² = 0.24 is moderate
   - Need to add more relevant features (weather, user history)
   - Consider adding duration as predictor (careful with leakage)

2. **Limited Temporal Context**

   - Only 13 months of data
   - May not capture all seasonal patterns
   - Need regularization to prevent overfitting

3. **Outliers**

   - Some sessions > 100 hours (vacation parking)
   - May benefit from separate modeling or capping

4. **Feature Leakage Risk**
   - Energy models cannot use duration in production
   - Need to predict both simultaneously or sequence appropriately

---

## 🤝 Contributing

This is a course project. Improvements welcome:

- Implement new regularization techniques
- Add LSTM/RNN architectures
- Integrate weather/traffic data
- Improve visualization dashboards

---

## 📄 License

Educational project for Neural Networks course (OTH Regensburg).  
Data sourced from Trondheim EV charging infrastructure.

---

**Project Lead:** Course Student  
**Course:** Neural Networks: Theory and Applications  
**Instructor:** Prof. Dr. Stefanie Vogl  
**Institution:** OTH Regensburg

---

## 📞 Support

For questions or issues:

1. Review the markdown documentation files
2. Check Jupyter notebook outputs
3. Refer to course lectures (especially Lecture 4 for next steps)
4. Consult SYLLABUS.md for theoretical background

---

_"Predicting the future of EV charging, one session at a time."_ ⚡🔋🚗
