# Report 4: Model Evaluation, Results, and Conclusions

---

## 1. Evaluation Framework

All models were evaluated on the held-out test set, which was not used during training or threshold selection. The test set contains 20% of the total data with the same class distribution as the full dataset (stratified split).

**Primary metrics:** Recall, Precision, F1 Score

**Secondary metric:** Confusion Matrix (absolute prediction counts)

**Rejected metric:** Accuracy — misleading for imbalanced datasets

---

## 2. Logistic Regression Results

**Default threshold (0.5):** High recall but extremely low precision — the model flagged a large proportion of legitimate transactions as fraud.

**After threshold tuning (optimal threshold = 0.85):**

| Metric | Value |
|---|---|
| Precision | 20.4% |
| Recall | 83.2% |
| F1 Score | 32.8% |

**Interpretation:** At the tuned threshold, the model catches 83% of all fraud cases. However, for every genuine fraud correctly identified, it also incorrectly flags approximately four legitimate transactions. This false alarm rate is operationally costly — it would require human reviewers to investigate large volumes of legitimate activity.

Logistic Regression's strength is its recall. Its weakness is that it cannot distinguish the fraud boundary precisely enough to avoid false positives, likely because the true decision boundary is non-linear.

---

## 3. Decision Tree Results

| Metric | Value |
|---|---|
| Precision | 3.88% |
| Recall | 85.26% |
| F1 Score | 7.43% |

**Interpretation:** The Decision Tree achieved marginally higher recall than Logistic Regression but at a catastrophic cost to precision. For every fraud correctly detected, it generated approximately 25 false alarms. This performance is operationally unusable.

The root cause is high variance — the single Decision Tree overreacted to minority class patterns despite depth restriction, drawing overly aggressive decision boundaries. This confirms that single trees are inherently unstable for imbalanced fraud detection.

---

## 4. Random Forest — Baseline Results

| Metric | Value |
|---|---|
| Precision | 81.61% |
| Recall | 74.73% |
| F1 Score | 78.02% |

**Confusion Matrix:**

|  | Predicted Legit | Predicted Fraud |
|---|---|---|
| **Actual Legit** | 56,635 | 16 |
| **Actual Fraud** | 24 | 71 |

**Interpretation:** Random Forest dramatically outperformed both previous models. The ensemble mechanism successfully reduced variance, eliminating the false alarm problem that plagued the Decision Tree.

Of the 95 total fraud cases in the test set, 71 were correctly identified with only 16 legitimate transactions incorrectly flagged. This represents a fundamental improvement in precision and overall balance.

The tradeoff: 24 fraud cases were missed. In a real banking context, each missed fraud represents a direct financial loss.

---

## 5. Random Forest — After Threshold Tuning

**Motivation:** The baseline Random Forest prioritized precision at the expense of recall. Given that missed fraud carries significant cost, recall was increased through threshold reduction.

**Selected threshold:** 0.40 (reduced from default 0.50)

| Metric | Value |
|---|---|
| Precision | 70.4% |
| Recall | 80.0% |
| F1 Score | 74.8% |

**Interpretation:** By lowering the classification threshold, the model now catches 80% of all fraud cases — an improvement of approximately 5 percentage points over baseline. Precision reduced from 81.6% to 70.4%, meaning slightly more false alarms. This tradeoff is acceptable given the cost of missed fraud versus the cost of investigating false alarms.

---

## 6. Complete Model Comparison

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| Logistic Regression (tuned) | 20.4% | 83.2% | 32.8% |
| Decision Tree | 3.9% | 85.3% | 7.4% |
| Random Forest (default) | 81.6% | 74.7% | 78.0% |
| **Random Forest (tuned) ★** | **70.4%** | **80.0%** | **74.8%** |

---

## 7. Final Model Selection Rationale

**Random Forest with threshold 0.40 was selected** as the final model for the following reasons:

- Achieves recall of 80% — 4 out of 5 fraud cases are detected
- Maintains precision of 70.4% — for every 10 flagged transactions, 7 are genuine fraud
- F1 Score of 74.8% represents the best overall balance
- Logistic Regression not selected despite higher recall: 20.4% precision is operationally impractical
- Decision Tree not selected: near-zero precision makes it completely unusable

---

## 8. Operational Interpretation for Banks

Two valid operational stances exist with different optimal configurations:

**Stance 1 — Maximum Fraud Detection (Recall-Optimized):**
Use Logistic Regression with low threshold. Catches 83% of fraud but generates high false alarm volume. Appropriate when human review capacity is large and the cost of missed fraud significantly exceeds the cost of investigation.

**Stance 2 — Balanced Detection with Manageable False Alarms:**
Use Random Forest at threshold 0.40. Catches 80% of fraud with substantially fewer false alarms. Appropriate when investigation resources are limited and the bank needs to prioritize reviewer time on high-confidence fraud flags.

Real banking systems typically favor the balanced approach, as excessive false positives erode customer trust and create operational bottlenecks. **Random Forest with threshold tuning is the production-appropriate choice.**

---

## 9. Limitations and Potential Improvements

**Current limitations:**
- SMOTE-based oversampling not implemented — may improve minority class representation
- Gradient Boosting not evaluated — typically outperforms Random Forest on tabular classification
- V1–V28 features are not interpretable — limits explainability to feature importance rankings
- Threshold selection based on a fixed rule rather than cost-based optimization

**Potential improvements:**
- Evaluate XGBoost or LightGBM as next-complexity step
- Implement Precision-Recall curve analysis for more rigorous threshold selection
- Cross-validation across multiple folds for more robust performance estimates
- Add SMOTE and compare results against class weighting

---

## 10. Conclusion

This project demonstrates that effective fraud detection requires deliberate decisions at every stage — metric selection, imbalance handling, model progression, and threshold tuning — each motivated by the specific constraints of the problem.

The final system detects **80% of fraud cases with 70% precision**, representing a practical balance between detection capability and operational feasibility.

> **Core principle demonstrated:** Model performance is not solely determined by algorithm choice. It is the product of correct problem framing, appropriate evaluation metrics, principled handling of data challenges, and deliberate threshold strategy.
>
> *EDA identifies what matters. Modeling learns how it matters. Evaluation ensures it works in reality.*
