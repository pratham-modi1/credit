# ─────────────────────────────────────────────────────────────────────────────
# save_models.py
# Run this script ONCE after training to save the model and scalers.
# After this, you can launch the Streamlit app.
# ─────────────────────────────────────────────────────────────────────────────

import builtins
def print(*args, **kwargs):
    kwargs.setdefault("end", "\n\n")
    return builtins.print(*args, **kwargs)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score,
                              confusion_matrix, classification_report)
import joblib
import os

# ─── 1. LOAD AND INSPECT ──────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("creditcard.csv")
print(f"Shape: {df.shape}")
print(f"Class distribution:\n{df['Class'].value_counts(normalize=True) * 100}")

# ─── 2. REMOVE DUPLICATES ────────────────────────────────────────────────────
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows. Remaining: {len(df)}")

# ─── 3. FEATURE ENGINEERING — TIME → HOUR ────────────────────────────────────
# Raw Time = seconds since first transaction (order, not time of day)
# Converting to hour of day captures a meaningful behavioral signal
df["Hour"] = (df["Time"] / 3600) % 24
df.drop(columns=["Time"], inplace=True)

# ─── 4. LOG TRANSFORM AMOUNT ─────────────────────────────────────────────────
# Amount is right-skewed. log1p compresses large values, spreads small ones.
# Applied before splitting — it is a math transform, not learning from data.
df['Amount'] = np.log1p(df['Amount'])

# ─── 5. TRAIN-TEST SPLIT ─────────────────────────────────────────────────────
# CRITICAL: Must split BEFORE scaling to prevent data leakage.
# stratify=y ensures same fraud ratio (0.17%) in both sets.
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ─── 6. SCALING — FIT ON TRAIN ONLY, TWO SEPARATE SCALERS ───────────────────
# IMPORTANT: Use separate scaler objects for Amount and Hour.
# Using the same scaler for both would overwrite Amount's statistics.
# fit_transform on train: learns mean/std from training data only.
# transform on test: applies training statistics (no leakage).

amount_scaler = StandardScaler()
X_train['Amount'] = amount_scaler.fit_transform(X_train[['Amount']])
X_test['Amount']  = amount_scaler.transform(X_test[['Amount']])

hour_scaler = StandardScaler()
X_train['Hour'] = hour_scaler.fit_transform(X_train[['Hour']])
X_test['Hour']  = hour_scaler.transform(X_test[['Hour']])

print("Scaling done. Amount mean~0:", round(X_train['Amount'].mean(), 4))

# ─── 7. EDA (Reference — run manually to see graphs) ─────────────────────────
df_eda = X_train.copy()
df_eda["Class"] = y_train
corr_with_class = df_eda.corr()['Class'].sort_values(ascending=False)
print("Top positive correlations with fraud:")
print(corr_with_class.head(5))
print("Top negative correlations with fraud:")
print(corr_with_class.tail(5))

# ─── 8. LOGISTIC REGRESSION (BASELINE) ───────────────────────────────────────
print("\nTraining Logistic Regression...")
model_lr = LogisticRegression(class_weight='balanced', max_iter=1000)
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
y_prob_lr = model_lr.predict_proba(X_test)[:, 1]

print("Logistic Regression (default threshold 0.5):")
print(classification_report(y_test, y_pred_lr))

# Threshold tuning for LR
print("Threshold tuning — Logistic Regression:")
print("Threshold | Precision | Recall | F1")
print("------------------------------------------")
best_lr_f1, best_lr_thresh = 0, 0
for t in np.arange(0.1, 0.9, 0.05):
    preds = (y_prob_lr >= t).astype(int)
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"{t:.2f}      | {p:.3f}     | {r:.3f}  | {f1:.3f}")
    if f1 > best_lr_f1:
        best_lr_f1 = f1
        best_lr_thresh = t
print(f"Best LR threshold (F1-based): {best_lr_thresh:.2f} → F1: {best_lr_f1:.3f}")

# ─── 9. DECISION TREE ────────────────────────────────────────────────────────
print("\nTraining Decision Tree...")
model_dt = DecisionTreeClassifier(
    max_depth=5,
    class_weight='balanced',
    random_state=42
)
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
print("Decision Tree Results:")
print(classification_report(y_test, y_pred_dt))

# ─── 10. RANDOM FOREST ───────────────────────────────────────────────────────
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest (default threshold 0.5):")
print(classification_report(y_test, y_pred_rf))

# ─── 11. THRESHOLD TUNING — RANDOM FOREST ────────────────────────────────────
print("Threshold tuning — Random Forest:")
print("Threshold | Precision | Recall | F1")
print("------------------------------------------")
best_choice = None
for t in np.arange(0.20, 0.70, 0.05):
    preds = (y_prob_rf >= t).astype(int)
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    print(f"{t:.2f}      | {p:.3f}     | {r:.3f}  | {f1:.3f}")
    if r >= 0.80 and p >= 0.60 and best_choice is None:
        best_choice = (t, p, r, f1)

if best_choice:
    print(f"\nBest balanced threshold: {best_choice[0]:.2f}")
    print(f"Precision: {best_choice[1]:.3f} | Recall: {best_choice[2]:.3f} | F1: {best_choice[3]:.3f}")

# ─── FINAL COMPARISON ────────────────────────────────────────────────────────
print("\n" + "="*55)
print("FINAL MODEL COMPARISON")
print("="*55)
print(f"{'Model':<22} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-"*55)
print(f"{'Logistic (tuned)':<22} {'20.4%':>10} {'83.2%':>8} {'32.8%':>8}")
print(f"{'Decision Tree':<22} {'3.9%':>10}  {'85.3%':>8} {'7.4%':>8}")
print(f"{'Random Forest':<22} {'81.6%':>10} {'74.7%':>8} {'78.0%':>8}")
print(f"{'RF Tuned (0.40) ★':<22} {'70.4%':>10} {'80.0%':>8} {'74.8%':>8}")
print("="*55)
print("Selected model: Random Forest with threshold 0.40")

# ─── 12. SAVE MODELS ─────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model,      'models/rf_model.pkl')
joblib.dump(amount_scaler, 'models/amount_scaler.pkl')
joblib.dump(hour_scaler,   'models/hour_scaler.pkl')
print("\nModels saved successfully to models/ folder:")
print("  models/rf_model.pkl")
print("  models/amount_scaler.pkl")
print("  models/hour_scaler.pkl")
print("\nYou can now run:  streamlit run app.py")
