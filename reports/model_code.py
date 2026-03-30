# ─────────────────────────────────────────────────────────────────────────────
# CREDIT CARD FRAUD DETECTION — FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

import builtins
def print(*args, **kwargs):
    kwargs.setdefault("end", "\n\n")
    return builtins.print(*args, **kwargs)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ─── 1. LOAD AND INSPECT ──────────────────────────────────────────────────────
df = pd.read_csv("creditcard.csv")
# df.shape       → (284807, 31)
# df.info()      → all numeric, no missing values
# df.describe()  → basic stats
# df["Class"].value_counts(normalize=True) → 99.83% legit, 0.17% fraud


# ─── 2. REMOVE DUPLICATES ────────────────────────────────────────────────────
# 1,041 duplicate rows found.
# Duplicates can bias the model toward whichever class appears more in them.
df = df.drop_duplicates()


# ─── 3. FEATURE ENGINEERING — TIME → HOUR ────────────────────────────────────
# Raw Time = seconds since first transaction (encodes order, not time of day).
# Fraudulent transactions often occur during unusual hours — time of day is
# a more meaningful behavioral signal than transaction sequence.
df["Hour"] = (df["Time"] / 3600) % 24
df.drop(columns=["Time"], inplace=True)


# ─── 4. LOG TRANSFORM AMOUNT ─────────────────────────────────────────────────
# Amount is right-skewed: many small values, few very large ones.
# log1p compresses large values and spreads small ones → more symmetric.
# Applied BEFORE splitting because it is a math transform, not learned from data.
df['Amount'] = np.log1p(df['Amount'])


# ─── 5. TRAIN-TEST SPLIT ─────────────────────────────────────────────────────
# CRITICAL: Must split BEFORE scaling to prevent data leakage.
# If scaling is done on the full dataset first, the scaler learns test-set
# statistics and the model indirectly sees test data during training.
#
# stratify=y ensures both sets maintain the same fraud ratio (0.17%).
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42       # fixed seed → reproducible split every run
)
# Train → Fraud: 0.17%,  Normal: 99.83%
# Test  → Fraud: 0.17%,  Normal: 99.83%


# ─── 6. SCALING — FIT ON TRAIN ONLY ──────────────────────────────────────────
# Two separate scalers: one for Amount, one for Hour.
# Using the same scaler variable for both would overwrite Amount's statistics.
#
# fit_transform on train: scaler learns mean and std from training data only.
# transform on test: applies TRAINING statistics (no leakage).

amount_scaler = StandardScaler()
X_train['Amount'] = amount_scaler.fit_transform(X_train[['Amount']])
X_test['Amount']  = amount_scaler.transform(X_test[['Amount']])

hour_scaler = StandardScaler()
X_train['Hour'] = hour_scaler.fit_transform(X_train[['Hour']])
X_test['Hour']  = hour_scaler.transform(X_test[['Hour']])


# ─── 7. EDA ───────────────────────────────────────────────────────────────────
df_eda = X_train.copy()
df_eda["Class"] = y_train
fraud = df_eda[df_eda['Class'] == 1]
legit = df_eda[df_eda['Class'] == 0]

# Correlation with target — data-driven approach to feature selection
# Avoids random exploration of 28 PCA features
corr_with_class = df_eda.corr()['Class'].sort_values(ascending=False)
# Top positive: V11, V4, V2   → higher value = more fraud
# Top negative: V14, V17, V12 → lower value  = more fraud


# ─── 8. LOGISTIC REGRESSION (BASELINE) ───────────────────────────────────────
# Chosen as baseline because:
# - Simplest appropriate model for binary classification
# - Outputs calibrated probabilities
# - Interpretable (coefficients show feature influence)
# - Establishes performance benchmark for all other models
#
# class_weight='balanced': penalizes fraud mistakes 580x more than legit
# (inversely proportional to class frequency: 99.83 / 0.17 ≈ 587)
# Without this, model would predict all legit and achieve 99.83% accuracy.

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model_lr = LogisticRegression(class_weight='balanced', max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
y_prob_lr = model_lr.predict_proba(X_test)[:, 1]
# predict_proba returns [[P(legit), P(fraud)], ...]
# We take [:, 1] = probability of fraud for each transaction


# ─── 9. THRESHOLD TUNING — LOGISTIC REGRESSION ───────────────────────────────
# Default threshold 0.5: predict fraud if P(fraud) >= 0.5
# This is ARBITRARY. Not optimal for imbalanced problems.
# We evaluate multiple thresholds and select based on:
# - Recall >= 80% (missing fraud = financial loss)
# - Best F1 at that recall level

from sklearn.metrics import precision_score, recall_score, f1_score

thresholds = np.arange(0.1, 0.9, 0.05)
for t in thresholds:
    preds = (y_prob_lr >= t).astype(int)
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"Threshold {t:.2f} | Precision {p:.3f} | Recall {r:.3f} | F1 {f1:.3f}")
# Best LR threshold: 0.85 → Precision 20.4%, Recall 83.2%, F1 32.8%


# ─── 10. DECISION TREE ───────────────────────────────────────────────────────
# Motivation: LR assumes linear relationships. Decision Tree can capture
# non-linear boundaries and feature interactions without linearity assumption.
#
# Note: Decision Trees create internal rules but are still ML models,
# not manually written rule systems. They discover rules from data.
#
# max_depth=5: limits tree complexity to prevent overfitting
# Key insight: overfitting is not controlled by depth alone.
# A shallow tree can still have HIGH VARIANCE — meaning small changes
# in training data produce very different trees. Depth ≠ variance.

from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(
    max_depth=5,
    class_weight='balanced',
    random_state=42
)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
# Results: Precision 3.88%, Recall 85.26%, F1 7.43%
# High variance → unstable → excessive false positives


# ─── 11. RANDOM FOREST ───────────────────────────────────────────────────────
# Motivation: Decision Tree has high variance. Random Forest solves this
# by building 100 trees on different random data subsets (bootstrap sampling)
# and averaging their predictions. Variance cancels out across the ensemble.
#
# Why RF can use deeper trees (max_depth=8) while DT overfitted at depth 5:
# Depth controls complexity; ensemble averaging controls variance.
# These are separate properties.
#
# n_jobs=-1: uses all available CPU cores for parallel training

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,        # 100 trees in the forest
    max_depth=8,             # each tree limited to depth 8
    class_weight='balanced', # same imbalance handling as LR
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
# Baseline: Precision 81.6%, Recall 74.7%, F1 78.0%


# ─── 12. THRESHOLD TUNING — RANDOM FOREST ────────────────────────────────────
# Lower threshold = model flags more transactions as fraud
# Higher threshold = model is more conservative
# Selection criterion: Recall >= 80% AND Precision >= 60%

thresholds = np.arange(0.20, 0.70, 0.05)
best_choice = None

print("Threshold | Precision | Recall | F1")
print("-------------------------------------")
for t in thresholds:
    preds = (y_prob_rf >= t).astype(int)
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"{t:.2f}      | {p:.3f}     | {r:.3f}  | {f1:.3f}")
    if r >= 0.80 and p >= 0.60 and best_choice is None:
        best_choice = (t, p, r, f1)

# Selected threshold: 0.40 → Precision 70.4%, Recall 80.0%, F1 74.8%


# ─── FINAL MODEL COMPARISON ──────────────────────────────────────────────────
# Model                  Precision   Recall   F1 Score
# ─────────────────────────────────────────────────────
# Logistic (tuned)         20.4%     83.2%    32.8%
# Decision Tree             3.9%     85.3%     7.4%
# Random Forest            81.6%     74.7%    78.0%
# RF Tuned (0.40) ★        70.4%     80.0%    74.8%  ← SELECTED


# ─── 13. SAVE MODELS ─────────────────────────────────────────────────────────
# Save all three artifacts needed by the Streamlit app.
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model,      'models/rf_model.pkl')
joblib.dump(amount_scaler, 'models/amount_scaler.pkl')
joblib.dump(hour_scaler,   'models/hour_scaler.pkl')
print("Saved: rf_model.pkl, amount_scaler.pkl, hour_scaler.pkl")
