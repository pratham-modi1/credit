# Report 3: Model Selection and Reasoning

---

## 1. Defining the Problem Before Selecting a Model

Before choosing any algorithm, the nature of the problem must be precisely defined. Selecting a model without this step is arbitrary.

**Problem characteristics identified:**
- Binary classification: each transaction is fraud (1) or legitimate (0)
- Highly imbalanced classes: 0.17% fraud rate
- Probability output required: a binary yes/no is insufficient — the model must output a fraud probability to allow threshold control
- Tabular structured data: no images, text, or sequences — classical ML models are appropriate
- No requirement for deep learning: designed for unstructured data; overkill here

---

## 2. Why Rule-Based Approaches Were Rejected

An initial consideration was whether fraud could be detected using fixed rules — for example, flagging transactions above a certain amount or occurring after midnight.

**Rejected for two reasons:**

1. EDA demonstrated that no single feature provides clean separation. Rules based on any one feature would produce high false positive rates.
2. Fraud patterns evolve. Fraudsters adapt their behavior to avoid detection. A fixed rule set becomes obsolete quickly.

Machine learning models that learn from data and output probabilities are the correct approach.

---

## 3. Why Accuracy Was Rejected as the Primary Metric

Given the 0.17% fraud rate, a model that predicts every transaction as legitimate would achieve **99.83% accuracy** while detecting zero fraud cases. Accuracy is therefore not only uninformative — it is actively misleading for this problem.

**Metrics selected instead:**

| Metric | Why It Matters |
|---|---|
| **Recall** | What proportion of actual fraud did the model catch? Missing fraud = direct financial loss |
| **Precision** | Of cases flagged as fraud, how many were actually fraud? Low precision = excessive false alarms |
| **F1 Score** | Harmonic mean — useful when both precision and recall matter |
| **Confusion Matrix** | Absolute counts of correct and incorrect predictions |

**Important clarification:** Recall is not a fixed property of an algorithm. It is controlled by the classification threshold, class weights, and training strategy. The question is not *"which model has the best recall"* but *"how should the modeling system be configured to achieve target recall with acceptable precision."*

---

## 4. Handling Class Imbalance

Three approaches were considered. `class_weight='balanced'` was implemented as the primary strategy.

**class_weight='balanced'** instructs the model to penalize mistakes on the minority class proportionally more than mistakes on the majority class. The weight assigned is inversely proportional to class frequency.

Effect: The model treats each missed fraud as approximately 580× more costly than a missed legitimate transaction, reflecting the actual class ratio. This does not change the data — it changes the learning objective.

Without this parameter, the model would default to predicting everything as legitimate (maximizing accuracy by ignoring fraud entirely).

---

## 5. Model Selection Strategy — Iterative Complexity

Rather than selecting the most complex model immediately, a structured progression from simple to complex was followed.

> **Principle:** Start with the simplest model that could plausibly work. Identify its specific limitations. Only then introduce a more complex model to address those exact limitations.

This approach ensures complexity is justified, not arbitrary.

---

## 6. Model 1 — Logistic Regression (Baseline)

**Why chosen:**
- Simplest appropriate model for binary classification
- Outputs calibrated probabilities natively
- Interpretable: coefficients directly indicate feature influence
- Establishes performance baseline for all subsequent models
- Works well when feature-target relationships are approximately linear

**Configuration:** `class_weight='balanced'`, `max_iter=1000`

Logistic Regression learns a weighted linear combination of all features and passes the result through a sigmoid function to produce a probability between 0 and 1.

**Known limitation:** Assumes linear relationships between features and the log-odds of fraud. Cannot capture interactions such as *"fraud occurs when V11 is high AND V17 is simultaneously low."*

---

## 7. Model 2 — Decision Tree Classifier

**Motivation:** Logistic Regression's linear assumption may be too restrictive. Decision Trees can capture non-linear boundaries and feature interactions.

**Important correction during this phase:** Decision Trees create internal rules during training, but this does not make them equivalent to manually written rule-based systems. Decision Trees are still learning models — they discover their rules from data and can output probabilities from leaf node class distributions.

**Configuration:** `max_depth=5`, `class_weight='balanced'`, `random_state=42`

**Limitation encountered:** Despite shallow depth, the Decision Tree showed high instability — excessive false positives due to overreaction to rare fraud patterns.

**Key insight:** Overfitting in a Decision Tree is not controlled by depth alone. A shallow tree can still exhibit high variance if small changes in training data lead to substantially different trees. **Depth controls complexity; variance is a separate problem.**

---

## 8. Model 3 — Random Forest Classifier

**Motivation:** The Decision Tree's instability identified the core problem — high variance. The solution is an ensemble method that averages multiple trees, canceling out individual variance.

Random Forest builds many decision trees, each trained on a different random subset of data (bootstrap sampling) and using a random subset of features at each split. The final prediction is the majority vote across all trees.

**Why this directly addresses the Decision Tree's weakness:**
- No single tree dominates the prediction
- Noise and outlier effects are averaged out
- The model generalizes rather than memorizing

**Configuration:** `n_estimators=100`, `max_depth=8`, `class_weight='balanced'`, `random_state=42`, `n_jobs=-1`

Note: Although each tree in the Random Forest can be deeper (max_depth=8 vs 5 for the single Decision Tree), ensemble averaging prevents overfitting. Depth and variance are separate properties.

| Model | Core Property |
|---|---|
| Decision Tree | High variance — unstable |
| Random Forest | Low variance — stable via averaging |

---

## 9. Threshold Tuning — Applied to All Models

All models output fraud probabilities, not binary labels. The classification threshold — the probability above which a transaction is flagged as fraud — is a separate decision from model training.

**Default threshold:** 0.5 (arbitrary, inappropriate for imbalanced problems)

**Tuning process:** Probabilities from `predict_proba` were evaluated against threshold values from 0.10 to 0.85. For each threshold, precision, recall, and F1 were calculated. The optimal threshold was selected based on the operational requirement: **recall ≥ 80% while precision remains meaningful.**

This separates two distinct decisions:
1. *What pattern does the model learn?* → determined by training
2. *At what confidence level do we act on the model's output?* → determined by threshold

Treating these as independent decisions gives precise control over the precision-recall tradeoff without retraining the model.
