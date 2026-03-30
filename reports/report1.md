# Report 1: Data Understanding and Preprocessing

---

## 1. Dataset Overview

The dataset contains **284,807 credit card transactions** made by European cardholders. Each transaction is labeled as fraudulent (`Class = 1`) or legitimate (`Class = 0`).

The dataset consists of 31 columns: `Time`, `Amount`, `V1` through `V28`, and `Class`.

---

## 2. Understanding the Features

**V1 to V28** are the result of Principal Component Analysis applied to the original transaction data, which included sensitive fields such as cardholder identity, merchant details, location, and behavioral patterns. PCA was applied to protect privacy while retaining the statistical structure of the data.

Properties of PCA features:
- Each V feature is a linear combination of all original attributes
- They are mathematically uncorrelated with each other
- Ordered by explained variance — V1 captures the most variance
- Cannot be interpreted directly, but carry strong predictive patterns

Key implication: Because V1–V28 encode behavioral and contextual patterns, fraud detection in this dataset is not about the transaction amount alone. A small transaction can be fraudulent and a large one can be legitimate. The model must learn subtle combinations of these features.

---

## 3. Class Imbalance — The Central Challenge

| Class | Count | Percentage |
|---|---|---|
| Legitimate (0) | 284,315 | 99.83% |
| Fraud (1) | 492 | 0.17% |

This extreme imbalance is not a data quality issue — it reflects reality. Fraud is rare by nature. However, it creates a fundamental modeling challenge: **a model that predicts every transaction as legitimate would achieve 99.83% accuracy while detecting zero fraud cases.** This makes accuracy a completely misleading metric for this problem.

All subsequent decisions — metric selection, class weighting, threshold tuning — are direct consequences of this imbalance.

---

## 4. Duplicate Row Handling

Initial check revealed **1,041 duplicate rows** in the dataset.

**Reason for removal:** Duplicate rows bias the model's learning. If duplicated rows are disproportionately from one class, the model implicitly treats that class as more important. To ensure the model learns from an unbiased distribution, all duplicates were removed before any further processing.

---

## 5. Feature Engineering — Time to Hour Conversion

The raw `Time` column represents seconds elapsed since the first transaction. In this form, it encodes transaction order, not time of day — which has no meaningful relationship with fraud.

Fraudulent transactions, however, often occur during unusual hours such as late night or early morning, when cardholders are less likely to notice unauthorized activity. Time of day is therefore a more informative representation.

```
Hour = (Time / 3600) % 24
```

This maps each transaction to its hour of the day (0 to 23). The original `Time` column was then dropped.

---

## 6. Handling Skewness in Amount

The `Amount` column exhibited strong right skew — the majority of transactions were small values, with a long tail of high-value transactions. This creates two problems:

1. Models that rely on distance or magnitude can be disproportionately influenced by extreme values
2. Standard scaling applied to a skewed distribution does not produce a well-centered result

**Solution:** Log transformation using `log1p` was applied before scaling.

| Original Amount | After log1p |
|---|---|
| 10 | 2.40 |
| 100 | 4.62 |
| 1000 | 6.91 |

---

## 7. Train-Test Split — Why It Must Happen Before Scaling

The dataset was split into training (80%) and test (20%) sets **before** applying StandardScaler.

This ordering is critical. If scaling is applied to the full dataset before splitting, the scaler learns the mean and standard deviation from all data — including the test set. This means the model indirectly gains information about test data during training, a problem known as **data leakage**.

**Correct procedure:**
- `scaler.fit_transform()` applied only to training data
- `scaler.transform()` applied to test data using training statistics only

The `stratify=y` parameter ensures both training and test sets maintain the same fraud-to-legitimate ratio (0.17%). Without this, a random split could produce a test set with disproportionately few fraud cases, making evaluation unreliable.

**Additional note:** Two separate scaler objects were used — one for `Amount` and one for `Hour`. Using the same scaler variable for both would overwrite the first scaler's learned statistics, resulting in incorrect scaling for `Amount`.

---

## 8. Preprocessing Summary

| Step | Action | Reason |
|---|---|---|
| 1 | Remove 1,041 duplicate rows | Prevent class bias |
| 2 | Convert Time → Hour of day | More meaningful feature |
| 3 | Apply log1p to Amount | Reduce right skew |
| 4 | Train-test split (80/20, stratified) | Prevent data leakage |
| 5 | StandardScaler on Amount (train only) | Equal feature contribution |
| 6 | StandardScaler on Hour (train only) | Equal feature contribution |

Dataset is now clean, properly split, and ready for EDA and modeling. No missing values exist. All features are numerical. No categorical encoding is required.
