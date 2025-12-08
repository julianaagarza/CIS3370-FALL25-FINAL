# Machine Learning Model Development

## 1. Overview
The core of this IDS is a **Supervised Multiclass Classifier** designed to detect behavioral anomalies in encrypted network traffic. To ensure robustness, we implemented the approach of training a **Logistic Regression** baseline and comparing it against a tuned **Random Forest** model.

## 2. Data Preparation & Splitting

### Preprocessing & Column Dropping
Before training, we removed identifiers and non-numeric columns that could lead to overfitting (e.g., learning specific IP addresses instead of behaviors).

* **Dropped Columns:** `['label', 'label_str', 'timestamp', 'flow_id', 'src_ip', 'dst_ip', 'source_ip', 'destination_ip']`
* **Final Feature Count:** 78 numeric features.

### Sampling & Stratified Split
Due to the massive size of the CIC-IDS-2018 dataset, we sampled **100,000 rows** to facilitate rapid hyperparameter tuning while maintaining statistical significance. We utilized a **Train / Validation / Test** split strategy to prevent data leakage during tuning.

**Data Shapes:**
* **Total Sample:** (100,000, 78)
* **Train Set (64%):** `(64000, 78)` - Used for model fitting.
* **Validation Set (16%):** `(16000, 78)` - Used for Hyperparameter Tuning (RandomizedSearchCV).
* **Test Set (20%):** `(20000, 78)` - Held out for final unbiased evaluation.

**Class Distribution:**
The split was stratified, preserving the exact class ratios across all sets:
* **Class 0 (Benign):** ~69.7%
* **Class 3:** ~13.5%
* **Class 6:** ~9.1%
* **Class 9:** ~3.8%
* **Class 12:** ~3.7%

> **Design Decision: Stratified Splitting**
> We chose stratified splitting over a simple random split to ensure that rare attack classes (Classes 9 and 12) were represented equally in the Training, Validation, and Test sets. A simple random split might have excluded these rare attacks from the validation set, skewing our tuning results.

## 3. Model Training & Tuning

We trained two distinct models to evaluate the trade-off between complexity and performance.

### A. Baseline: Logistic Regression
* **Pros:** Fast training, highly interpretable coefficients.
* **Cons:** Assumes linear relationships between features.

### B. Random Forest (Tuned)
* **Tuning Method:** `RandomizedSearchCV`
* **Configuration:** 3-Fold Cross-Validation with 5 candidate parameter settings.
* **Why Random Forest?** It creates a "Forest" of decision trees, allowing it to capture non-linear complex relationships in flow duration and packet variance that linear models might miss.


## 4. Performance Results

Both models performed exceptionally well, but **Random Forest** was selected for production due to its near-perfect precision and recall on the Test set.

### Comparative Metrics (Test Set)

| Metric | Logistic Regression (Baseline) | **Random Forest** |
| :--- | :--- | :--- |
| **Accuracy** | 99.70% | **99.98%** |
| **F1-Weighted** | 0.9970 | **0.9998** |
| **ROC-AUC** | 0.9990 | **1.0000** |

### Detailed Analysis (Random Forest Test Set)
The Random Forest model achieved a perfect **1.00** score across Precision, Recall, and F1-Score for all classes, including the minority classes. Note: This could possibly due to be overfitting, but for the sake of the pipeline we continued with this model in order to see our working real-time simulation in action.


```text
              precision    recall  f1-score   support
           0       1.00      1.00      1.00     13943
           3       1.00      1.00      1.00      2718
           6       1.00      1.00      1.00      1830
           9       1.00      1.00      1.00       766
          12       1.00      1.00      1.00       743
