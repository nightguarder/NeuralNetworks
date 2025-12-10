# EV Charging Data Cleanup Plan

Date: December 3, 2025

This document captures the agreed data-cleaning decisions for the Trondheim EV charging dataset and the reasoning behind them. It also lists the outputs produced by the cleaning notebook.

---

## Objectives

1. Remove clearly invalid or non-informative sessions (e.g., near-zero duration, impossible values).
2. Remove extreme outliers that are not useful for forecasting typical charging behavior.
3. Standardize and validate timestamps, then recompute duration to ensure consistency.
4. Produce a clean, analysis-ready CSV for modeling.

---

## Cleaning Rules (Final)

1. Timestamp normalization

- Parse `Start_plugin` and `End_plugout` as datetime with day-first format (e.g., `21.12.2018 10:20`).
- Ensure `End_plugout >= Start_plugin`; drop records violating this (rare/inconsistent).
- Recompute `Duration_hours_calc = (End_plugout - Start_plugin).total_hours`.
- If `|Duration_hours - Duration_hours_calc| > 0.5h`, replace with `Duration_hours_calc`.

2. Remove invalid values

- Drop rows with `El_kWh <= 0`.
- Drop rows with `Duration_hours <= 0.05` (likely brief plug/unplug tests).

3. Remove extreme outliers

- Drop sessions with `Duration_hours > 200` hours (per requirement).
- Optionally flag (not drop) remaining top 1% of `El_kWh` as outliers for later robustness checks. No capping by default in this phase.

4. Feature consistency

- Keep and refresh convenience columns: `Start_plugin_hour`, `End_plugout_hour`, `weekdays_plugin`, `month_plugin` from timestamps to avoid drift.

5. Handle missing account ownership

- **Shared_ID Column**: All values were NaN (empty) in raw data
- **Decision**: Replace NaN with `"Private"` to indicate non-shared/individual accounts
- **Rationale**:
  - Current dataset contains only private EV users (all User_type = "Private")
  - Filling NaN provides semantic clarity: "Private" = individual owner, future "Shared" = shared account
  - Prevents missing values from causing issues in downstream analysis
  - Enables future distinction between account ownership types as data evolves

---

## February Anomaly (Fewer Sessions)

Observation: February has substantially fewer sessions relative to other months.

Decision: Keep February in the dataset.

- Rationale: The dataset represents real-world seasonality and operations. Removing February introduces selection bias. Instead, we will:
  - Use chronological train/test splits to avoid leakage.
  - Monitor month-wise performance during evaluation.
  - Consider per-month weighting only if the model underperforms specifically on February.

This issue can be acknowledged but safely ignored for cleaning. We will not drop or oversample February at this stage.

---

## Outputs

- Clean dataset saved to: `project/ev_project/data/ev_sessions_clean.csv`
- Log summary printed by notebook: counts of rows removed by rule and remaining dataset shape (overall and by month).

---

## Next Steps

- Proceed to feature engineering (temporal encodings, user/location aggregations, weather merge).
- Start baseline modeling (energy and duration) and track metrics by month to validate robustness.
