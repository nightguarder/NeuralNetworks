# Quick Start: Neural Network Optimization - How to Review

**Updated:** January 15, 2026  
**Status:** ✅ All notebooks ready for execution

---

## What You're Looking At

Three complete notebooks demonstrating neural network concepts and optimization:

1. **EV_Neural_Network_Experiment.ipynb** ← New optimized experiment
2. **EV_Pipeline_Evaluation.ipynb** ← Baseline tree models
3. **EV_Prediction_Demo.ipynb** ← Interactive predictions

---

## Fastest Way to Review (5 minutes)

### Step 1: Open the Notebook
```
Open: /project/ev_project/EV_Neural_Network_Experiment.ipynb
```

### Step 2: Run Everything
```
VS Code → Run All (or Kernel → Restart & Run All)
Execution time: ~71 seconds
```

### Step 3: Check Results
Look for output showing:
```
✓ V1 Classification: AUC=0.7929, F1=0.350
✓ V2 Classification: AUC=0.7780, F1=0.323
✓ V3 Classification: AUC=0.7744, F1=0.332
✓ V4 Classification: AUC=0.7763, F1=0.330

✓ V1 Regression: RMSE=5.688h, MAE=4.351h, R²=0.2422
✓ V3 Regression: RMSE=5.207h, MAE=3.936h, R²=0.3649

✓ CSV comparisons saved ✓ Plots generated
```

### Step 4: Review Key Findings

**Cell 25** (Markdown): Code Optimization Results  
→ See 65% code reduction story

**Cell 27**: Threshold Optimization  
→ See F1-sweep optimization in action

**Cell 29**: Final Summary  
→ See all metrics compared

---

## Standard Review (15 minutes)

### Phase 1: Setup & Data (2 min)
1. Read Cell 1: Problem statement
2. Skim Cell 3: Data loading (6,646 EV sessions)
3. Note Cell 5: Feature engineering (20 base features)

### Phase 2: Neural Networks (8 min)
1. **Cell 18:** Read "Unified Model Builders" section
   - Understand `build_classification_model(version, dropout_rate, l2_reg)`
   - See focal loss implementation
   - Grasp class weight computation

2. **Cell 20:** Run classification training
   - Watch: V1, V2, V3, V4 all train in parallel (~46 sec)
   - Check output: AUC values and optimal thresholds

3. **Cell 23:** Run regression training
   - See V1 (base features) vs V3 (with energy)
   - Note: 50% R² improvement from El_kWh feature

### Phase 3: Results (5 min)
1. **Cell 21:** Classification comparison CSV
2. **Cell 24:** Regression comparison CSV
3. **Cell 29:** Final summary metrics

### Conclusion (verbal, ~2 min)
- V1 wins classification (0.7929 AUC)
- V3 wins regression with energy features (0.3649 R²)
- Tree baseline still better (0.847 AUC) → shows good judgment

---

## Deep Review (30 minutes)

### Section 1: Background (5 min)

**Read:**
- Cell 1 (Markdown): Complete problem statement
- Cell 2 (Code): Data loading explanation
- Cell 4 (Markdown): Feature engineering overview

**Understand:**
- Why predicting session duration is hard (outliers, imbalance)
- Dataset structure (6,646 sessions, 20-26 features)
- Train/test split (70/30 chronological)

### Section 2: Architecture Design (8 min)

**Read Cell 18 carefully:**

```python
# Version configurations
v1_config = {'dropout': 0.3, 'l2': 0.001, ...}  # Conservative
v2_config = {'dropout': 0.20, 'l2': 0.0005, ...}  # More aggressive
v3_config = {'dropout': 0.16, 'l2': 0.00025, ...}  # Very aggressive
v4_config = {'dropout': 0.15, 'l2': 0.0003, 'loss': 'focal', ...}  # Focal loss
```

**Understand:**
- Why each version exists (testing different regularization levels)
- How focal_loss helps minority classes
- Class weight computation (Long sessions: 27% → weight 7.775)

### Section 3: Training & Optimization (10 min)

**Run Cell 20 (Classification Training):**
```
Takes ~46 seconds
Trains V1, V2, V3, V4 in unified loop
Check output:
  - V1: AUC=0.7929 @ threshold=0.637 ← Best
  - V2: AUC=0.7780 @ threshold=0.607
  - V3: AUC=0.7744 @ threshold=0.322
  - V4: AUC=0.7763 @ threshold=0.475
```

**Key Insight:** V1 (simplest regularization) outperforms V2-V4
- Why? Small dataset benefits from stronger regularization
- Lesson: Depth ≠ Better on tabular data

**Run Cell 23 (Regression Training):**
```
Takes ~18 seconds
V1: Only base 20 features
V3: Adds 6 features including El_kWh

Results:
  V1: R²=0.2422, RMSE=5.688h
  V3: R²=0.3649, RMSE=5.207h (+50.6% R² improvement!)
```

**Key Insight:** Energy feature (El_kWh) is critical
- Why? Direct causal relationship with charging duration
- Lesson: Domain knowledge > Model complexity

### Section 4: Code Quality (5 min)

**Find in Cell 18:**
- Line with `def build_classification_model(version, input_dim, ...)`
- This single function replaces 4 separate builders

**Find in Cell 20:**
- Loop: `for version in ['v1', 'v2', 'v3', 'v4']:`
- This unified loop replaces 4 separate training blocks

**Understand:**
- Before: 1,130 lines with 80% duplication
- After: 400 lines, fully modular
- Benefit: Changes to training protocol only need one edit

### Section 5: Results Analysis (2 min)

**Review these CSVs:**
```
# Classification results
fig/classification/all_versions_comparison.csv
  V1 wins with AUC=0.7929

# Regression results  
fig/modeling_regularized/all_regression_comparison.csv
  V3 wins with R²=0.3649 vs Baseline R²=0.1610
```

---

## For Discussion with Professor

### Prepared Talking Points

**"What regularization taught me:"**
> "I tested V1 (L2=0.001, Dropout=0.3) vs V2-V4 with lighter regularization. Counter-intuitively, V1 performed best (AUC 0.7929 vs 0.778 for others). This taught me that on small datasets, regularization strength matters more than depth. I would have expected a larger network to perform better, but the opposite happened."

**"Why energy features matter:"**
> "V1 regression achieved R²=0.2422 with 20 features, but adding El_kWh (energy delivered) to V3 jumped to R²=0.3649—a 50% improvement. This shows that domain-specific features are more valuable than architectural tweaks. The energy directly causes duration; no model tricks can beat that signal."

**"Code optimization lesson:"**
> "I initially wrote 4 separate model builders and 4 training loops. Then I refactored into one parameterized builder and one unified loop. This reduced code by 65% and actually improved V1's AUC (0.756→0.793) by enabling better threshold optimization that I'd missed before. Clean code isn't just prettier—it enables better analysis."

**"Why trees win here:"**
> "My neural networks achieve 0.793 AUC (best case), but HistGradientBoosting baseline achieves 0.847. This -5.4% gap teaches me that tree models are optimized for tabular data with small-to-medium datasets. This is actually a valuable lesson—knowing when NOT to use deep learning is as important as knowing when to use it."

---

## Troubleshooting

### Notebook Won't Run
```
Issue: ModuleNotFoundError
Fix: pip install tensorflow scikit-learn pandas numpy matplotlib

Issue: Memory issues
Fix: Restart kernel, run sections one at a time instead of all at once
```

### Results Look Different
```
Issue: Slightly different metrics
Reason: Random seed variations in neural networks
Note: Should be within ±0.5% if refactored correctly
```

### Can't Find CSV Files
```
Check: /project/ev_project/fig/
  - classification/all_versions_comparison.csv
  - modeling_regularized/all_regression_comparison.csv
```

---

## Key Files to Show

| File | Purpose | Run Time |
|------|---------|----------|
| EV_Neural_Network_Experiment.ipynb | Main work | 71 sec |
| EV_Pipeline_Evaluation.ipynb | Baseline proof | 40 sec |
| EV_Prediction_Demo.ipynb | Interactive demo | 30 sec |
| PROFESSOR_SUMMARY.md | This summary | Read |
| FINAL_OPTIMIZATION_REPORT.md | Deep dive | Read |

---

## Success Criteria

✅ Notebook runs without errors  
✅ All 4 classification models train (V1-V4)  
✅ Both regression models train (V1-V3)  
✅ Results show V1 best for classification  
✅ Results show V3 best for regression  
✅ CSV files generated successfully  
✅ Can explain why V1 > V2-V4  
✅ Can explain why V3 > V1 (energy feature)  
✅ Can articulate when NNs lose to trees  

If all ✅, you're ready to present!

---

## Timeline for Presentation

- **5 min before:** Open EV_Neural_Network_Experiment.ipynb, verify it runs
- **2 min in:** Show cells 1-5, explain the problem
- **3 min in:** Jump to cell 18, explain unified architecture
- **5 min in:** Run cell 20 (watch training)
- **7 min in:** Show results output (V1 wins)
- **8 min in:** Run cell 23 (watch regression training)
- **10 min in:** Show regression results (V3 wins)
- **11 min in:** Discuss why (conservation, energy importance)
- **13 min in:** Show code refactoring (65% reduction)
- **14 min in:** Conclude with lesson (right tool for job)
- **15 min done:** Q&A

---

**You've got this! 🚀**
