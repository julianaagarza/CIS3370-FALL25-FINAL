# Preprocessing and Training Pipeline

This document explains the full preprocessing and baseline training pipeline used for the CICIDS2018 flows in Google Colab. It mirrors the steps in `preprocessingandtraining.ipynb`.

## 1. Imports and Setup

We start by importing core libraries for data handling, preprocessing, and model training.

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn import preprocessing
    from sklearn.feature_selection import RFE
    from sklearn.inspection import permutation_importance
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings("ignore")

Mount Google Drive and confirm that the CSVs are available:

    from google.colab import drive
    drive.mount('/content/drive')

    import os
    print(os.listdir('/content/drive/MyDrive/idsprojectcsvs'))

Expected layout:

- `/content/drive/MyDrive/idsprojectcsvs`
  - `02-14-2018.csv`
  - `02-15-2018.csv`
  - `02-16-2018.csv`
  - `02-21-2018.csv`
  - `02-22-2018.csv`
  - `02-28-2018.csv`
  - `03-01-2018.csv`
  - `03-02-2018.csv`

## 2. Load All Daily CSVs

Each CSV is loaded into a dictionary of DataFrames keyed by filename.

    folder_path = '/content/drive/MyDrive/idsprojectcsvs'
    dfs = {}

    for fname in os.listdir(folder_path):
        if fname.endswith('.csv'):
            path = os.path.join(folder_path, fname)
            dfs[fname] = pd.read_csv(path)

At this point, every daily CSV is in memory.

## 3. Verify Column Consistency

We ensure all days have the same columns.

    column_names = {name: df.columns.tolist() for name, df in dfs.items()}

    all_columns = set()
    for name, columns in column_names.items():
        all_columns.update(columns)

    for name, columns in column_names.items():
        missing = list(all_columns - set(columns))
        extra = list(set(columns) - all_columns)
        print(f"DataFrame: {name}")
        print(f"  Missing columns: {missing}")
        print(f"  Extra columns: {extra}")
        print("-" * 20)

In this run, no CSVs were missing or containing extra columns.

## 4. Normalize Column Names and Initial Type Conversion

Normalize names and convert numeric fields, except protected columns.

Protected:

- `label`
- `protocol`
- `day`
- `timestamp`

    for name, df in dfs.items():
        df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')

        non_numeric = ['label', 'protocol', 'day', 'timestamp']

        for c in df.columns:
            if c not in non_numeric:
                df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors='coerce')

        dfs[name] = df

## 5. Inspect Descriptive Stats

    for name, df in dfs.items():
        print(name, df.describe().T[['count','mean','std','min','max']].head(25))

This helps validate data shape and numeric ranges.

## 6. Drop Constant Columns

We identify and drop columns that contain only one unique value.

    for name, df in dfs.items():
        zero_cols = [c for c in df.columns if df[c].nunique() == 1]
        dfs[name] = df.drop(columns=zero_cols)
        print(f"Dropped from {name}: {zero_cols}")

These included several flag and average-rate columns that never vary.

## 7. Drop Timestamps

    for name, df in dfs.items():
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        dfs[name] = df

## 8. Second Pass Cleaning: Numeric Enforcement + Inf Handling

    safe_numeric = ['label','protocol']

    for name in list(dfs.keys()):
        df = dfs[name]

        df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')

        for c in df.columns:
            if c not in safe_numeric:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)

        dfs[name] = df
        print(name, "done", df.shape)

## 9. Enforce Common Columns Across All Days

    common_cols = set(dfs[next(iter(dfs))].columns)
    for df in dfs.values():
        common_cols &= set(df.columns)
    common_cols = list(common_cols)

    for name, df in dfs.items():
        dfs[name] = df[common_cols]

    for name, df in dfs.items():
        print(name, df.shape)

Result: all DataFrames now have 69 identical columns.

## 10. Build the Training Dataset

Training days:

- 02-14-2018
- 02-15-2018
- 02-16-2018
- 02-21-2018
- 02-22-2018

    train_days = [
        '02-14-2018.csv',
        '02-15-2018.csv',
        '02-16-2018.csv',
        '02-21-2018.csv',
        '02-22-2018.csv'
    ]

    train_df = pd.concat([dfs[d] for d in train_days], ignore_index=True)
    print(train_df.shape)

Shape: `(5242875, 69)`.

## 11. Encode Protocol and Label + Final Numeric Cleanup

    train_df['protocol'] = pd.to_numeric(train_df['protocol'], errors='coerce')
    train_df['label'] = train_df['label'].astype('category').cat.codes

    for c in train_df.columns:
        if c != 'label':
            train_df[c] = pd.to_numeric(train_df[c], errors='coerce')

    train_df = train_df.fillna(train_df.median(numeric_only=True))

## 12. Feature Scaling

    X = train_df.drop(columns=['label'])
    y = train_df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

## 13. Remove the Four Rarest Classes

    y_series = pd.Series(y)

    smallest_4 = y_series.value_counts().nsmallest(4).index
    print("Dropping classes:", list(smallest_4))

    mask = ~y_series.isin(smallest_4)

    X_filtered = X_scaled[mask]
    y_filtered = y_series[mask]

    print("New class distribution:")
    print(y_filtered.value_counts())

## 14. Train/Validation Split

    X_train, X_val, y_train, y_val = train_test_split(
        X_filtered,
        y_filtered,
        test_size=0.2,
        random_state=42,
        stratify=y_filtered
    )

## 15. Random Forest Baseline Model

    from sklearn.metrics import accuracy_score, classification_report

    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        max_features='sqrt',
        min_samples_split=4,
        n_jobs=2,
        class_weight='balanced'
    )

    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)

    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

Observed accuracy ≈ **0.983**.

## 16. Logistic Regression Baseline

    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        n_jobs=-1
    )

    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_val)

    print("Logistic Regression Accuracy:", accuracy_score(y_val, y_pred_lr))
    print(classification_report(y_val, y_pred_lr))

Observed accuracy ≈ **0.977**.

## 17. Summary

This notebook achieves:

- Loading and unifying CICIDS2018 CSVs.
- Normalizing column names and data types.
- Cleaning NaN/Inf values, dropping constants and timestamps.
- Encoding labels, scaling features, and balancing classes.
- Training two baseline models:
  - **Random Forest** (best performer)
  - **Logistic Regression**

These steps form the foundation of the project’s machine learning pipeline before applying further tuning or integration with Elasticsearch/Kibana.

