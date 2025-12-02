# Setup & Dataset Usage Notes

This document explains how to set up the Colab environment, load the CICIDS2018 dataset, preprocess the data, and prepare it for machine learning. Each main section includes a short description of its purpose.

---

## 1. Environment Setup (Google Colab)

*This section covers how to prepare Google Colab so the dataset and required tools load properly.*

### 1.1. Open Colab

Open a new Google Colab notebook or an existing one stored in Drive or GitHub.

### 1.2. Install Required Python Packages

Run this cell at the top of your notebook:

    !pip install pandas numpy scikit-learn shap matplotlib seaborn elasticsearch

These packages support:

- Data cleaning (pandas, numpy)  
- Machine learning (scikit-learn)  
- Explainability (SHAP)  
- Visualization (matplotlib / seaborn)  
- Optional Elasticsearch integration  

### 1.3. Mount Google Drive (Where the Dataset Lives)

    from google.colab import drive
    drive.mount('/content/drive')

Your dataset should be located in:

    /content/drive/MyDrive/idsprojectcsvs/

---

## 2. Dataset Overview (CICIDS2018)

*This section explains which dataset files are used, what they contain, and common issues found in the raw CSVs.*

### 2.1. What We Use

**Training Days**
- 02-14-2018  
- 02-15-2018  
- 02-16-2018  
- 02-21-2018  
- 02-22-2018  

**Testing Days**
- 02-28-2018  
- 03-01-2018  
- 03-02-2018  

Each CSV contains:

- ~80 numeric flow-based features  
- Protocol information  
- Labels (benign/malicious)  

No PCAPs or Suricata logs are used.

### 2.2. Common Dataset Issues & Fixes

| Issue | Our Solution |
|-------|--------------|
| Column names have spaces/mixed casing | Normalize names |
| Numeric columns contain strings | Convert with `errors='coerce'` |
| Columns contain only zeros | Drop when needed |
| Infinity or missing values | Replace and fill with median |

---

## 3. Loading the Dataset in Colab

*This section explains how to load all dataset files into memory so they can be cleaned and processed.*

### 3.1. Load all CSVs into a dictionary

    import pandas as pd
    import os

    base_path = "/content/drive/MyDrive/idsprojectcsvs"
    dfs = {}

    for file in os.listdir(base_path):
        if file.endswith(".csv"):
            dfs[file] = pd.read_csv(os.path.join(base_path, file))

---

## 4. Preprocessing Pipeline (Simplified Explanation)

*This section describes the data cleaning steps applied to standardize the dataset before model training.*

### 4.1. Normalize column names

    df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')

### 4.2. Convert all feature columns to numeric

    safe = ['label','protocol']

    for c in df.columns:
        if c not in safe:
            df[c] = pd.to_numeric(df[c], errors='coerce')

### 4.3. Replace infinities, then fill missing

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

### 4.4. Label & protocol handling

- label → integer-encoded  
- protocol → numeric  

---

## 5. Train/Test Construction

*This section shows how the daily CSVs are combined into unified training and testing datasets.*

### 5.1. Merge training days

    train_days = ['02-14-2018.csv','02-15-2018.csv','02-16-2018.csv','02-21-2018.csv','02-22-2018.csv']
    train_df = pd.concat([dfs[d] for d in train_days], ignore_index=True)

### 5.2. Merge testing days

    test_days = ['02-28-2018.csv','03-01-2018.csv','03-02-2018.csv']
    test_df = pd.concat([dfs[d] for d in test_days], ignore_index=True)

---

## 6. Model Scaling (StandardScaler)

*This section describes how features are scaled prior to model training.*

    from sklearn.preprocessing import StandardScaler

    X_train = train_df.drop(columns=['label'])
    y_train = train_df['label']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

---

## 7. Notes for Users

*This section provides additional guidance, limitations, and expected outputs of the preprocessing pipeline.*

### 7.1. Memory Constraints

- Load one day at a time if necessary  
- Minimize duplicate dataframes  
- Colab Pro is helpful for large RAM workloads  

### 7.2. Why We Don't Use Suricata

The CICIDS2018 CSVs already contain processed flow metadata, so:

- No PCAPs  
- No Suricata logs  

### 7.3. Expected Output After Preprocessing

- ~79 numeric columns  
- Encoded labels  
- No missing or infinite values  
- Training set ≈ 5.2 million rows  
- Test size varies by day  

---

