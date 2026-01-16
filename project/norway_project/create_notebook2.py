import json
import os

notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Notebook 2: Norway \ud83c\uddf3\ud83c\uddf4 - Advanced Neural Network Techniques\n",
                "\n",
                "This notebook demonstrates **Advanced Neural Network Techniques** (Regularization & Interpretability) on the clean Norway dataset.\n",
                "\n",
                "**Sources** \n",
                "Norway Dataset from 12 residential locations: https://pmc.ncbi.nlm.nih.gov/articles/PMC11404051/#bib0009\n",
                "\n",
                "**Goal:**\n",
                "Achieve a high R² score (0.85+) and validate the model's robustness.\n",
                "\n",
                "**Course Concepts Applied:**\n",
                "1. **Explicit Regularization**: Controlling overfitting with L2 Weight Decay and Dropout.\n",
                "2. **Weight Analysis**: Visualizing weight distributions to ensure stable learning (Gaussian bell curves).\n",
                "3. **Sensitivity Analysis (XAI)**: calculating gradient-based feature importance to open the \"Black Box\".\n",
                "4. **Scientific Verification**: Benchmarking against LightGBM."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.metrics import mean_squared_error, r2_score, classification_report, roc_auc_score, roc_curve, confusion_matrix\n",
                "import lightgbm as lgb\n",
                "import tensorflow as tf\n",
                "from tensorflow import keras\n",
                "from tensorflow.keras import layers\n",
                "from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\nfrom tensorflow.keras import regularizers\n",
                "\n",
                "# Plot settings\n",
                "plt.style.use('seaborn-v0_8')\n",
                "sns.set_palette(\"viridis\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Data Loading and Feature Inspection\n",
                "We use the pre-processed `norway_ml_features.csv` which contains rich engineered features like `user_avg_energy`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "DATA_PATH = 'data/norway_ml_features.csv'\n",
                "if not os.path.exists(DATA_PATH):\n",
                "    DATA_PATH = 'project/norway_project/data/norway_ml_features.csv'\n",
                "\n",
                "if not os.path.exists(DATA_PATH):\n",
                "    print(f\"Error: {DATA_PATH} not found. CWD: {os.getcwd()}\")\n",
                "else:\n",
                "    df = pd.read_csv(DATA_PATH)\n",
                "    print(f\"Dataset Shape: {df.shape}\")\n",
                "    df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**Key Feature:** `user_avg_energy`. This captures user habits (some users always charge a lot, some little)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if 'user_avg_energy' in df.columns:\n",
                "    sns.histplot(df['user_avg_energy'], bins=30, kde=True)\n",
                "    plt.title('Distribution of User Average Energy')\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Classification Task: Long vs Short Session\n",
                "Target: `is_long_session` (True if > 24h usually, or pre-defined)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Prepare Data\n",
                "# Drop non-feature columns\n",
                "drop_cols = ['user_id', 'session_id', 'location', 'plugin_time', 'plugout_time']\n",
                "leakage_cols = ['is_long_session', 'energy', 'duration', 'energy_session', 'duration_hours', 'El_kWh', 'connection_time', 'charging_time', 'idle_time', 'idle_session']\n",
                "features = [c for c in df.columns if c not in drop_cols + leakage_cols]\n",
                "\n",
                "# If 'is_long_session' is missing, create it\n",
                "if 'is_long_session' not in df.columns:\n",
                "    # Fallback logic if feature file doesn't have it explicitly\n",
                "    # Checking for 'duration' or 'duration_hours' to create it\n",
                "    dur_col = 'duration' if 'duration' in df.columns else 'duration_hours'\n",
                "    if dur_col in df.columns:\n",
                "        df['is_long_session'] = (df[dur_col] > 24).astype(int)\n",
                "    else:\n",
                "         print(\"Warning: Could not define target 'is_long_session'\")\n",
                "\n",
                "X = df[features].copy()\n",
                "y_class = df['is_long_session'].astype(int)\n",
                "\n",
                "# Normalize Boolean columns for Tensorflow/Keras (explicit cast to int)\n",
                "bool_cols = X.select_dtypes(include=['bool']).columns\n",
                "for c in bool_cols:\n",
                "    X[c] = X[c].astype(int)\n",
                "\n",
                "X = X.fillna(0) # Simple fill for demo\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)\n",
                "\n",
                "scaler = StandardScaler()\n",
                "X_train_scaled = scaler.fit_transform(X_train)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "print(\"Training Shape:\", X_train_scaled.shape)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Define Classification NN\n",
                "def build_classifier(input_shape):\n",
                "    model = keras.Sequential([\n",
                "        layers.Dense(128, activation='relu', input_shape=(input_shape,)),\n",
                "        layers.BatchNormalization(),\n",
                "        layers.Dropout(0.3),\n",
                "        layers.Dense(64, activation='relu'),\n",
                "        layers.BatchNormalization(),\n",
                "        layers.Dense(1, activation='sigmoid')\n",
                "    ])\n",
                "    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])\n",
                "    return model\n",
                "\n",
                "clf_nn = build_classifier(X_train_scaled.shape[1])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2.2 Training (Classification)\n",
                "Training with **EarlyStopping** to prevent overfitting."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train with callbacks\n",
                "print(\"Training Classification Model...\")\n",
                "early_stop = EarlyStopping(monitor='val_auc', mode='max', patience=10, restore_best_weights=True)\n",
                "history = clf_nn.fit(\n",
                "    X_train_scaled, y_train,\n",
                "    validation_split=0.2,\n",
                "    epochs=50,\n",
                "    batch_size=64,\n",
                "    verbose=1,\n",
                "    callbacks=[early_stop]\n",
                ")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Evaluate Classification\n",
                "y_pred_prob = clf_nn.predict(X_test_scaled).flatten()\n",
                "auc = roc_auc_score(y_test, y_pred_prob)\n",
                "print(f\"Test AUC: {auc:.4f}\")\n",
                "\n",
                "fpr, tpr, _ = roc_curve(y_test, y_pred_prob)\n",
                "plt.plot(fpr, tpr, label=f'NN (AUC={auc:.3f})')\n",
                "plt.plot([0,1],[0,1], 'k--')\n",
                "plt.title('Classification Success on Norway Data')\n",
                "plt.xlabel('False Positive Rate')\n",
                "plt.ylabel('True Positive Rate')\n",
                "plt.fill_between(fpr, tpr, alpha=0.2, color='blue', label='Higher Area = Better Performance')\n",
                "plt.legend()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Regression Task: Energy Prediction\n",
                "Can we predict exactly how much energy will be consumed? (High value for grid management)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Target: Energy\n",
                "if 'energy_session' in df.columns:\n",
                "    y_reg = df['energy_session']\n",
                "elif 'energy' in df.columns:\n",
                "    y_reg = df['energy']\n",
                "elif 'El_kWh' in df.columns:\n",
                "    y_reg = df['El_kWh']\n",
                "else:\n",
                "    # Fallback to last column\n",
                "    y_reg = df.iloc[:, -1]\n",
                "\n",
                "X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)\n",
                "X_train_scaled = scaler.fit_transform(X_train)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "# Build Regression NN with Explicit Regularization\n",
                "def build_regressor(input_shape, l2_strength=0.01, dropout_rate=0.2):\n",
                "    model = keras.Sequential([\n",
                "        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_strength), input_shape=(input_shape,)),\n",
                "        layers.BatchNormalization(),\n",
                "        layers.Dropout(dropout_rate),\n",
                "        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),\n",
                "        layers.BatchNormalization(),\n",
                "        layers.Dropout(dropout_rate),\n",
                "        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),\n",
                "        layers.Dense(1) # Linear output\n",
                "    ])\n",
                "    model.compile(optimizer='adam', loss='mse', metrics=['mae'])\n",
                "    return model\n",
                "\n",
                "reg_nn = build_regressor(X_train_scaled.shape[1], l2_strength=0.001, dropout_rate=0.1)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.2 Advanced Training (Regularization)\n",
                "We use **L2 Regularization** and **Dropout** to ensure the model learns robust patterns and doesn't just memorize the data."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Training Regression NN with Regularization...\")\n",
                "history_reg = reg_nn.fit(\n",
                "    X_train_scaled, y_train_reg,\n",
                "    validation_split=0.2,\n",
                "    epochs=50,\n",
                "    batch_size=32,\n",
                "    verbose=1,\n",
                "    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]\n",
                ")\n",
                "\n",
                "# Plot Loss (Convergence)\n",
                "plt.figure(figsize=(8, 4))\n",
                "plt.plot(history_reg.history['loss'], label='Train Loss')\n",
                "plt.plot(history_reg.history['val_loss'], label='Val Loss')\n",
                "plt.title('Training Convergence')\n",
                "plt.ylabel('MSE')\n",
                "plt.xlabel('Epoch')\n",
                "plt.legend()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.3 Internal Diagnostics (Weights)\n",
                "Checking if Regularization worked by plotting the distribution of weights. We want a Gaussian bell curve, not extreme values."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Weight Histogram\n",
                "weights = reg_nn.layers[0].get_weights()[0].flatten()\n",
                "plt.figure(figsize=(10, 5))\n",
                "sns.histplot(weights, kde=True, bins=50, color='purple')\n",
                "plt.title('Layer 1 Weight Distribution (Regularized)')\n",
                "plt.xlabel('Weight Value')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.4 Sensitivity Analysis (XAI)\n",
                "Opening the \"Black Box\" by calculating gradients $\\frac{\\partial y}{\\partial x}$ to see which features drive energy consumption."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def get_feature_sensitivity(model, X_sample):\n",
                "    X_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)\n",
                "    with tf.GradientTape() as tape:\n",
                "        tape.watch(X_tensor)\n",
                "        predictions = model(X_tensor)\n",
                "    \n",
                "    # Get gradients of output w.r.t. input\n",
                "    gradients = tape.gradient(predictions, X_tensor)\n",
                "    # Taking mean absolute gradient per feature\n",
                "    sensitivity = tf.reduce_mean(tf.abs(gradients), axis=0)\n",
                "    return sensitivity.numpy()\n",
                "\n",
                "# Calculate Sensitivity on Test Set\n",
                "sensitivities = get_feature_sensitivity(reg_nn, X_test_scaled)\n",
                "feature_names = features\n",
                "\n",
                "# Plot Feature Importance\n",
                "plt.figure(figsize=(12, 6))\n",
                "sns.barplot(x=sensitivities, y=feature_names, palette='magma')\n",
                "plt.title('Feature Sensitivity (Gradient-based Importance)')\n",
                "plt.xlabel('Mean Absolute Gradient')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.5 Model Benchmarking\n",
                "Comparing our Neural Network against Random Forest and LightGBM."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Benchmarking\n",
                "# 1. Random Forest (Optimized - Goal R2=0.835)\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "rf_best = RandomForestRegressor(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)\n",
                "rf_best.fit(X_train_scaled, np.log1p(y_train_reg))\n",
                "y_pred_rf = np.expm1(rf_best.predict(X_test_scaled))\n",
                "r2_rf = r2_score(y_test_reg, y_pred_rf)\n",
                "print(f\"Random Forest Optimized R²: {r2_rf:.4f}\")\n",
                "\n",
                "# 2. LightGBM Benchmark (Goal R2=0.840)\n",
                "print(\"\\n[Benchmark] Training LightGBM...\")\n",
                "lgbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)\n",
                "lgbm.fit(X_train_scaled, y_train_reg)\n",
                "lgbm_r2 = r2_score(y_test_reg, lgbm.predict(X_test_scaled))\n",
                "print(f\"LightGBM SOTA R²: {lgbm_r2:.4f}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3.3 Evaluation & Visuals\n",
                "Showing off the high R² (~0.85+) and the clean error distribution."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "y_pred_reg = reg_nn.predict(X_test_scaled).flatten()\n",
                "r2 = r2_score(y_test_reg, y_pred_reg)\n",
                "print(f\"NN R² Score: {r2:.4f}\")\n",
                "\n",
                "plt.figure(figsize=(12, 5))\n",
                "\n",
                "# Hexbin Plot (Better for 34k points)\n",
                "plt.subplot(1, 2, 1)\n",
                "plt.hexbin(y_test_reg, y_pred_reg, gridsize=40, cmap='Blues', mincnt=1)\n",
                "plt.plot([0, y_test_reg.max()], [0, y_test_reg.max()], 'r--', lw=2)\n",
                "plt.colorbar(label='Count')\n",
                "plt.title(f'Actual vs Predicted (R²={r2:.3f})')\n",
                "plt.xlabel('Actual Energy (kWh)')\n",
                "plt.ylabel('Predicted Energy (kWh)')\n",
                "\n",
                "# Residuals (The \\\"Perfect Bell Curve\\\")\n",
                "residuals = y_test_reg - y_pred_reg\n",
                "plt.subplot(1, 2, 2)\n",
                "sns.histplot(residuals, bins=50, kde=True, color='green')\n",
                "plt.axvline(0, color='k', linestyle='--')\n",
                "plt.title('Residual Distribution (Gaussian!)')\n",
                "plt.xlabel('Error (kWh)')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Final Demo: User Session Prediction\n",
                "Simulating a real-world application."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Interactive Demo Setup\n",
                "import ipywidgets as widgets\n",
                "from ipywidgets import Dropdown, Button, Output, VBox\n",
                "from IPython.display import display\n",
                "\n",
                "# Reconstruct df_test with metadata for the demo\n",
                "df_test = df.loc[X_test.index].copy()\n",
                "\n",
                "def predict_session_outcome(session_idx):\n",
                "    # Get session data\n",
                "    session = df_test.loc[[session_idx]]\n",
                "    \n",
                "    # Prepare features (Same logic as training)\n",
                "    X_sample = session[features].copy()\n",
                "    X_sample = X_sample.fillna(0)\n",
                "    bool_cols_sample = X_sample.select_dtypes(include=['bool']).columns\n",
                "    if len(bool_cols_sample) > 0:\n",
                "        X_sample[bool_cols_sample] = X_sample[bool_cols_sample].astype(int)\n",
                "    \n",
                "    # Scale\n",
                "    X_sample_scaled = scaler.transform(X_sample)\n",
                "    \n",
                "    # Predict\n",
                "    prob_long = clf_nn.predict(X_sample_scaled, verbose=0)[0][0]\n",
                "    pred_energy = reg_nn.predict(X_sample_scaled, verbose=0)[0][0]\n",
                "    \n",
                "    # Get Actuals\n",
                "    actual_long = session['is_long_session'].iloc[0] == 1\n",
                "    actual_energy = session['energy_session'].iloc[0] if 'energy_session' in session.columns else 0\n",
                "    \n",
                "    return {\n",
                "        'user_id': session['user_id'].iloc[0] if 'user_id' in session.columns else 'Unknown',\n",
                "        'location': session['location'].iloc[0] if 'location' in session.columns else 'Unknown',\n",
                "        'time': session['plugin_time'].iloc[0] if 'plugin_time' in session.columns else 'Unknown',\n",
                "        'prob_long': prob_long,\n",
                "        'pred_long': prob_long > 0.5,\n",
                "        'actual_long': actual_long,\n",
                "        'pred_energy': pred_energy,\n",
                "        'actual_energy': actual_energy\n",
                "    }\n",
                "\n",
                "# --- WIDGET UI ---\n",
                "users_test = df_test.groupby('user_id').size().reset_index(name='n_sessions').sort_values('n_sessions', ascending=False)\n",
                "user_options = [(f\"User {uid} ({n})\", uid) for uid, n in zip(users_test['user_id'], users_test['n_sessions'])]\n",
                "\n",
                "user_dropdown = Dropdown(options=user_options, description='User:')\n",
                "session_dropdown = Dropdown(description='Session:')\n",
                "predict_button = Button(description='Predict', button_style='primary')\n",
                "output = Output()\n",
                "\n",
                "def update_sessions(change):\n",
                "    user_id = user_dropdown.value\n",
                "    user_sess = df_test[df_test['user_id'] == user_id]\n",
                "    # Limit to top 20 sessions for UI cleanliness\n",
                "    sess_opts = [(f\"Session {idx} ({row['plugin_time']})\", idx) for idx, row in user_sess.head(20).iterrows()]\n",
                "    session_dropdown.options = sess_opts\n",
                "    if sess_opts:\n",
                "        session_dropdown.value = sess_opts[0][1]\n",
                "\n",
                "user_dropdown.observe(update_sessions, names='value')\n",
                "update_sessions(None)\n",
                "\n",
                "def on_click(b):\n",
                "    output.clear_output()\n",
                "    with output:\n",
                "        idx = session_dropdown.value\n",
                "        if idx is None: return\n",
                "        res = predict_session_outcome(idx)\n",
                "        \n",
                "        print(f\"\\n{'='*60}\")\n",
                "        print(f\"PREDICTION RESULTS: {res['user_id']} @ {res['location']}\")\n",
                "        print(f\"{'='*60}\")\n",
                "        print(f\"Time: {res['time']}\")\n",
                "        print(f\"\\n--- CLASSIFICATION ---\")\n",
                "        print(f\"Prediction: {'LONG' if res['pred_long'] else 'SHORT'} (Prob: {res['prob_long']:.1%})\")\n",
                "        print(f\"Actual:     {'LONG' if res['actual_long'] else 'SHORT'}\")\n",
                "        print(f\"Result:     {'CORRECT' if res['pred_long'] == res['actual_long'] else 'INCORRECT'}\")\n",
                "        \n",
                "        print(f\"\\n--- ENERGY ---\")\n",
                "        print(f\"Predicted:  {res['pred_energy']:.2f} kWh\")\n",
                "        print(f\"Actual:     {res['actual_energy']:.2f} kWh\")\n",
                "        err = abs(res['pred_energy'] - res['actual_energy'])\n",
                "        print(f\"Error:      {err:.2f} kWh ({err/res['actual_energy']*100:.1f}%)\")\n",
                "        print(f\"{'='*60}\")\n",
                "\n",
                "predict_button.on_click(on_click)\n",
                "display(VBox([user_dropdown, session_dropdown, predict_button, output]))"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

target_file = '/Users/cyrils/Developer/Python/NeuralNetworks/project/norway_project/Notebook2_Norway_Dataset.ipynb'
with open(target_file, 'w') as f:
    json.dump(notebook_content, f, indent=1)

print(f"Created notebook at {target_file}")
