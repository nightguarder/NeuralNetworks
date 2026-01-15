# 🇳🇴 Norway 12-Location EV Charging Project

**Dataset:** Sørensen et al., Data in Brief 2024 (PMCID: PMC11404051, DOI: 10.1016/j.dib.2024.110883)  
**Goal:** Replicate published research methodology to generate complete EV charging datasets + Neural Network experiments  
**Status:** 🚀 Ready to Execute  
**Timeline:** 3 days ⚡ (Streamlined!)

---

## 🎯 Project Overview - STREAMLINED APPROACH

This project **replicates the scientific methodology** from Sørensen et al. 2024 in **3 focused phases**:

### **PHASE 1 (Day 1):** Data Pipeline + Dataset Generation

- Clean raw data (Table 5 rules)
- Generate Dataset 2: User predictions (261 users)
- Generate Dataset 3: Session predictions (34K sessions)
- Generate Dataset 4: Hourly predictions (500K+ hours)
- Engineer ML features

### **PHASE 2 (Day 2):** Neural Network Training

- Experiment 1: Connection time classification (Long/Short)
- Experiment 2: Energy session regression
- Experiment 3: Idle time prediction (flexibility)
- Compare: Trondheim (6,880) vs Norway (35,377) performance

### **PHASE 3 (Day 3):** Demo + Presentation

- Build interactive demo notebook
- Create HTML presentation with car charging diagram
- Integrate key findings from study
- Professional 60-minute presentation ready

---

## 📊 Why This Dataset is PERFECT

| Metric          | Value                 | Advantage                         |
| --------------- | --------------------- | --------------------------------- |
| **Sessions**    | 35,377                | 5.1× more than Trondheim (6,880)  |
| **Users**       | 267                   | 2.1× more than Trondheim (127)    |
| **Locations**   | 12 Norwegian sites    | Geographic diversity              |
| **Time Period** | 3.5 years (2018-2021) | Long observation window           |
| **Data Size**   | Perfect for NNs       | 30K+ samples → better performance |

**Bottom Line:** Ideal dataset size for demonstrating Neural Network superiority with larger data!

---

## 📁 What We Have

### Dataset 1 (Raw Data)

**File:** `Dataset1_charging_reports.csv`

- **35,377 charging sessions**
- **267 unique users**
- **12 residential locations** across Norway
- **Period:** February 2018 - August 2021 (3.5 years)

**Columns:**

- `location` - CP location (ASK, BAR, BER, BOD, KRO, OSL_1/2/S/T, TRO, TRO_R, BAR_2)
- `user_id` - User identifier (267 unique)
- `session_id` - Charging session ID (35,377 unique)
- `plugin_time` - Plug-in timestamp (CET with DST)
- `plugout_time` - Plug-out timestamp (CET with DST)
- `connection_time` - Connection duration [hours]
- `energy_session` - Energy charged [kWh]

---

## 🎓 What We'll Generate (Following Paper's Methodology)

### Dataset 2: User Predictions

**Goal:** Predict charging_power and battery_capacity for each user  
**Method:** Equations 1-5 from paper  
**Output:** 261 users (6 removed as outliers)

**Columns:**

- `user_id` - User ID
- `charging_power` [kW] - Predicted charging power (3.7-11 kW typical)
- `battery_capacity` [kWh] - Predicted net battery capacity (24-100 kWh typical)

### Dataset 3: Session Predictions

**Goal:** Generate per-session charging metrics  
**Method:** Equations 6-11 from paper  
**Output:** ~34,500 sessions (only users with valid predictions)

**Columns:**

- `user_id`, `session_id`
- `charging_time` [h] - Actual charging duration
- `SoC_diff` [%] - State of charge difference
- `SoC_start` [%] - Starting SoC (assuming end = 95%)
- `idle_time` [h] - Non-charging idle time
- `idle_session` [kWh] - Idle energy capacity (flexibility potential)
- `non_flex_session` [kWh] - Energy for sessions with <1h idle time

### Dataset 4: Hourly Predictions

**Goal:** Expand each session into hourly granularity  
**Method:** Equations 6, 9-11 applied hourly  
**Output:** ~500,000-1,00 (STREAMLINED - Only 3!)

1. **Norway_Complete_Data_Pipeline.ipynb** (Day 1)

   - Load & clean Dataset 1
   - Generate Datasets 2, 3, 4 in one notebook
   - Feature engineering for ML

2. **Norway_Neural_Network_Experiments.ipynb** (Day 2)

   - 3 experiments: Classification, Energy regression, Idle time
   - Performance comparison vs Trondheim
   - Learning curves showing data advantage

3. **Norway_Final_Presentation.ipynb** (Day 3)
   - Interactive demo for professor
   - Integrated study findings
   - Complete results showcase

---

## 📚 Notebooks to Create

1. **Norway_01_Data_Exploration.ipynb** - Load, clean, validate Dataset 1
2. **Norway_02_Exploratory_Analysis.ipynb** - Patterns across 12 locations
3. **Norway_03_User_Predictions.ipynb** - Generate Dataset 2
4. **Norway_04_Session_Predictions.ipynb** - Generate Dataset 3
5. **Norway_05_Hourly_Predictions.ipynb** - Generate Dataset 4
6. **Norway_06_Neural_Network_Experiments.ipynb** - Train NNs, compare with Trondheim
7. **Norway_07_Comprehensive_Analysis.ipynb** - Validate results vs paper
8. **Norway_08_Final_Presentation.ipynb** - Live demo for professor

---

## 🧹 Data Cleaning (Table 5 from Paper)

Apply these rules to Dataset 1:

1. ❌ Remove sessions with `energy_session ≤ 0.5 kWh` or `> 150 kWh`
2. ❌ Remove sessions with `connection_time < 2 min` or `> 5 days`
3. ❌ Calculate `avg_power = energy_session / connection_time`
4. ❌ Remove sessions with `avg_power ≥ 11.5 kW` (except OSL_T special cases)
5. ❌ Remove users with `< 10 charging sessions`
6. ✅ Handle CET timezone and DST transitions

**Expected Result:** ~32,000 clean sessions, 261 users

---

## 🧮 Key Equations to Implement

From Sørensen et al. 2024:

**Charging Power (Eq. 1-3):**

```
connection_time = plugout_time - plugin_time
avg_power = energy_session / connection_time
charging_power = max(avg_power) per user, validated against levels
```

**Battery Capacity (Eq. 4-5):**

```
Ebattery = max(energy_session) × 0.88  (efficiency factor)
battery_capacity = Ebattery / SoC_range
  where SoC_range = 0.80 (small/medium EVs) or 0.75 (large EVs)
```

**Charging Time (Eq. 7):**

```
charging_time = energy_session / (charging_power × 0.88)
```

**Idle Time (Eq. 8):**

```
idle_time = connection_time - charging_time
```

**State of Charge (Eq. 11):**

```
SoC_diff = (energy_session × 0.88) / battery_capacity × 100
SoC_start = 95% - SoC_diff  (assuming end = 95%)
```

---

## 🧠 Neural Network Experiments

**Why This Dataset is Better for NNs:**

- 35,377 samples vs Trondheim's 6,880 (5× more)
- More diverse (12 locations vs 1)
- Longer period (better temporal patterns)

**Experiments to Run:**

1. **Charging Power Classification** - Predict charger level (L1/L2/L3)
2. **Battery Capacity Regression** - Predict battery_capacity from patterns
3. **Connection Time Prediction** - Two-stage (long/short classifier + regression)
4. **Energy Flexibility Prediction** - Predict idle_time (smart charging potential)

**Expected Results:**

- Classification AUC: ~0.88 (vs 0.79 Trondheim) ✅ 11% improvement
- Regression R²: ~0.35 (vs 0.16 Trondheim) ✅ 2× improvement
- Learning curves: Clear benefit from 5× more data

---

## 📈 Expected Outcomes

**Data Generation Success:**

- ✅ 3 new datasets (2, 3, 4) matching paper's schema
- ✅ Validation: Our distributions match published results
- ✅ ~32,000 clean sessions, 261 users with predictions

**Neural Network Performance:**

- ✅ Improved metrics vs Trondheim project
- ✅ Learning curves showing data size advantage
- ✅ Regularization techniques from Lecture 4

**Scientific Contribution:**

- ✅ Reproduced research methodology
- ✅ Validated published approach
- ✅ Extended with NN predictions
- ✅ Location-specific insights (12 sites)

---

## 🎬 Final Presentation Structure

**60-Minute Live Demo for Professor:**

1. **Introduction (5 min)** - Dataset overview, project goals
2. **Methodology (15 min)** - Show equations, cleaning rules, implementation
3. **Results: Datasets 2 & 3 (10 min)** - User & session predictions
4. **Results: Dataset 4 (10 min)** - Hourly granularity, load profiles
5. **Neural Networks (15 min)** - 4 experiments, comparison with Trondheim
6. **Validation (5 min)** - Our results vs paper's published statistics
7. **Conclusions (5 min)** - Impact, future work, Q&A

**Deliverables:**

- 8 working Jupyter notebooks
- 3 generated CSV datasets (2, 3, 4)
- HTML presentation (standalone)
- CSTREAMLINED_PLAN.md` ⭐ **START HERE** - Complete 3-day plan
- `QUICK_START_GUIDE.md` - Day-by-day checklist
- `NOTEBOOK_INDEX.md` - Which notebook does what
- `PROFESSOR_SUMMARY.md` - Presentation talking points
- `STUDY_FINDINGS.md` - Key insights from paper to cite

---

## 🚀 Quick Start (3-Day Sprint!)

### Day 1: Data Pipeline (10-12 hours)

1. ✅ Load & clean Dataset 1 (32K sessions)
2. ✅ Generate Dataset 2 (261 users with predictions)
3. ✅ Generate Dataset 3 (34K session predictions)
4. ✅ Generate Dataset 4 (500K hourly predictions)
5. ✅ Engineer ML features

### Day 2: Neural Networks (8-10 hours)

1. ✅ Train classification model (Long/Short sessions)
2. ✅ Train energy regression model
3. ✅ Train idle time prediction model
4. ✅ Compare performance vs Trondheim
5. ✅ Generate learning curves & feature importance

### Day 3: Presentation (8-10 hours)

1. ✅ Build demo notebook
2. ✅ Create HTML presentation with car diagram
3. ✅ Integrate study findings
4. ✅ Practice 60-minute presentation
5. ✅ Polish & backupset 3 (session predictions)

- Days 5-6: Generate Dataset 4 (hourly predictions)

### Next Week

- Days 6-7: Neural network experiments
- Days 8-9: Analysis & validation
- Days 9-10: Presentation materials

---

## 💡 Key Advantages Over Trondheim Project

1. **5× More Data** → Better neural network performance
2. **Scientific Replication** → Following published methodology
3. **3 Generated Datasets** → Complete deliverable suite
4. **12 Locations** → Geographic diversity analysis
5. **Hourly Granularity** → Grid planning applications

---

## ✅ Success Criteria

s:\*\*

- Data article: https://pmc.ncbi.nlm.nih.gov/articles/PMC11404051/
- Methodology: https://www.sciencedirect.com/science/article/pii/S2352467723002035
- Car diagram (Fig. 15): From methodology paper

**Status:** 🟢 Ready to Execute - STREAMLINED!  
**Timeline:** 3 days (24-30 hours)  
**Confidence:** ⭐⭐⭐⭐⭐ High  
**Approach:** Fast, focused, professional

\*\*Let's execute! 🚀⚡sentation ready

- [ ] Complete documentation suite

---

**Reference Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11404051/  
**Status:** 🟢 Ready to Execute  
**Timeline:** 7-10 days (60-80 hours)  
**Confidence:** ⭐⭐⭐⭐⭐ High

**Let's build this! 🚀**
