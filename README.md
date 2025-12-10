# Neural Networks: Theory and Applications

**Course:** Neural Networks (Prof. Dr. Stefanie Vogl)  
**Institution:** OTH Regensburg  
**Last Updated:** December 10, 2025

This repository contains lecture materials, practical exercises, notebooks, and a comprehensive course project for an introductory neural networks course.

---

## 📁 Repository Structure

```
├── lectures/           # Jupyter notebooks covering course topics
├── notebooks/          # Practice notebooks and tutorials
├── code/              # Runnable example scripts (Keras/TensorFlow/PyTorch)
├── data/              # Course datasets and examples
├── project/           # Main course project
│   └── ev_project/    # ✨ EV Charging Behavior Prediction (Active)
├── requirements-macos.txt
└── SYLLABUS.md        # Detailed course syllabus
```

---

## 🚀 Quick Start

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# macOS / zsh
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements-macos.txt
```

### 3. Install PyTorch (Apple Silicon)

```bash
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

### 4. Verify Installation

Test PyTorch:

```bash
python code/test_pytorch.py
```

Test TensorFlow:

```bash
python code/test_tensorflow.py
```

### 5. Launch Jupyter

```bash
jupyter lab
```

Start with `notebooks/simple_mlp.ipynb` or explore the lectures.

---

## 📚 Course Topics

Based on the [SYLLABUS.md](SYLLABUS.md), the course covers:

1. **Neural Network Fundamentals**

   - Architecture, activation functions, layer types
   - Forward/backward propagation

2. **Training & Optimization**

   - Loss functions, gradient descent variants (SGD, Adam)
   - Learning rate scheduling

3. **Regularization Techniques** (Lecture 4)

   - L1/L2 regularization, dropout
   - Early stopping, batch normalization
   - Preventing overfitting

4. **Advanced Architectures**

   - Convolutional Neural Networks (CNNs)
   - Recurrent Neural Networks (RNNs)
   - Long Short-Term Memory (LSTMs)

5. **Practical Applications**
   - Image recognition
   - Time series forecasting
   - Real-world case studies

---

## 🎯 Course Project: EV Charging Prediction

**Location:** [`project/ev_project/`](project/ev_project/)

### Project Goal

Predict Electric Vehicle (EV) charging behavior to optimize charging infrastructure and energy management.

### Current Status

✅ **Phase 1: Classical ML Models (Complete)**

- Implemented and evaluated Ridge, Random Forest, XGBoost, and Keras NN
- Best performance: **Random Forest** with R² = 0.61 (Duration), R² = 0.24 (Energy)
- See [`project/ev_project/README.md`](project/ev_project/README.md) for details

### Key Achievements

- Comprehensive data analysis (6,880 charging sessions)
- Feature engineering (temporal encodings, categorical features)
- Model comparison with month-wise validation
- Visualization suite for prediction analysis

### Next Steps

- Implement regularization techniques from Lecture 4
- Add LSTM/RNN models following course material
- Incorporate weather and traffic data
- Hyperparameter tuning with early stopping

---

## 📖 Lectures

| Notebook                                 | Topic              | Key Concepts                     |
| ---------------------------------------- | ------------------ | -------------------------------- |
| `NN_Lecture1_Basics.ipynb`               | Introduction       | Perceptron, activation functions |
| `NN_Lecture2_Graphics_and_Data.ipynb`    | Data handling      | Visualization, preprocessing     |
| `NN_Lecture3_Neuro_get_started.ipynb`    | Getting started    | Keras basics, model building     |
| `NN_Lecture3_Part2_Metaparameters.ipynb` | Hyperparameters    | Learning rate, early stopping    |
| `NN_Lecture4_Regularisation.ipynb`       | **Regularization** | L1/L2, dropout, overfitting      |

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Deep Learning:** TensorFlow/Keras, PyTorch
- **ML Libraries:** scikit-learn, XGBoost
- **Data Science:** pandas, numpy, matplotlib, seaborn
- **Notebooks:** Jupyter Lab

---

## 📊 Project Results Snapshot

**Current Best Models (Test Set):**

| Target   | Model         | RMSE  | MAE  | R²   |
| -------- | ------------- | ----- | ---- | ---- |
| Duration | Random Forest | 11.38 | 3.45 | 0.60 |
| Duration | Keras NN      | 8.38  | 3.25 | 0.61 |
| Energy   | Random Forest | 10.41 | 6.59 | 0.24 |
| Energy   | XGBoost       | 10.96 | 7.01 | 0.15 |

_Full results and analysis in `project/ev_project/3_Modeling_Results.md`_

---

## 📝 Additional Resources

- **TensorFlow on macOS:** https://www.tensorflow.org/install/pip#macos
- **Course Syllabus:** [SYLLABUS.md](SYLLABUS.md)
- **Project Documentation:** [project/ev_project/](project/ev_project/)

---

## 🎓 Learning Outcomes

By completing this course and project, students will:

- ✅ Understand neural network architectures and training
- ✅ Implement models using Keras and PyTorch
- ✅ Apply regularization techniques to prevent overfitting
- ✅ Evaluate and optimize model performance
- ✅ Work with real-world datasets and time series
- ✅ Compare classical ML vs. deep learning approaches

---

**Note:** The `ev_charging_project/` folder has been archived as the LSTM/GRU approach did not perform well. The active project now focuses on classical ML models with plans to incorporate neural network techniques from the lectures.
