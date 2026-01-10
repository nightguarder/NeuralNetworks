import json

with open('EV_Pipeline_Evaluation.ipynb') as f:
    nb = json.load(f)

baseline_code = "# First, compute baseline pipeline metrics for comparison\npred_short_mask_baseline = (proba_long >= thr).astype(bool)\npred_long_mask_baseline = ~pred_short_mask_baseline\n\n# Baseline regression on predicted-short\nX_test_reg_baseline = test_enh.loc[pred_short_mask_baseline, num2 + cat2]\ny_test_reg_baseline = test_enh.loc[pred_short_mask_baseline, 'Duration_hours'].values\n\nif len(X_test_reg_baseline) > 0:\n    y_pred_log_baseline = rf_pipe.predict(X_test_reg_baseline)\n    y_pred_baseline = np.expm1(y_pred_log_baseline)\n    \n    rmse_pred_short_baseline = np.sqrt(mean_squared_error(y_test_reg_baseline, y_pred_baseline))\n    mae_pred_short_baseline = mean_absolute_error(y_test_reg_baseline, y_pred_baseline)\n    r2_pred_short_baseline = r2_score(y_test_reg_baseline, y_pred_baseline) if len(y_pred_baseline) > 1 else np.nan\nelse:\n    rmse_pred_short_baseline = np.nan\n    mae_pred_short_baseline = np.nan\n    r2_pred_short_baseline = np.nan\n    y_pred_baseline = np.array([])\n    y_test_reg_baseline = np.array([])\n\ncoverage_baseline = pred_short_mask_baseline.mean()\n\n"

cell17_src = ''.join(nb['cells'][17]['source'])
new_src = baseline_code + cell17_src

# Split by newlines and preserve newlines
lines = new_src.split('\n')
nb['cells'][17]['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]

with open('EV_Pipeline_Evaluation.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("OK")
