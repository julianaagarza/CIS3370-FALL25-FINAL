# In-Depth Preprocessing Steps

This document provides a detailed walkthrough of the preprocessing steps used to prepare the CICIDS2018 dataset for machine learning. The goal of preprocessing is to ensure all CSV files share consistent structure, contain clean numeric features, and are ready for feature selection and model training.

---

## 1. Importing All Dataset CSV Files

The CICIDS2018 CSV files were loaded into Google Colab and stored in a dictionary for easy access and per-file processing.

To verify column consistency across all files:

    column_names = {name: df.columns.tolist() for name, df in dfs.items()}

    all_columns = set()
    for name, columns in column_names.items():
        all_columns.update(columns)

    for name, columns in column_names.items():
        missing_columns = list(all_columns - set(columns))
        extra_columns = list(set(columns) - all_columns)
        print(f"DataFrame: {name}")
        print(f"  Missing columns: {missing_columns}")
        print(f"  Extra columns: {extra_columns}")
        print("-" * 20)

Output showed that no files were missing or containing extra columns, allowing preprocessing to continue.

---

## 2. Cleaning Column Names and Converting Features

Column names were cleaned to remove spaces, convert to lowercase, and replace spaces with underscores. Certain columns were protected from numeric conversion because they represent labels or metadata:

Protected columns:

- label  
- protocol  
- day  
- timestamp  

Cleaning and conversion:

    for name, df in dfs.items():

        # clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')

        # columns that should NOT be numerically converted
        non_numeric = ['label', 'protocol', 'day', 'timestamp']

        # convert all but the protected ones
        for c in df.columns:
            if c not in non_numeric:
                df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors='coerce')

        dfs[name] = df

---

## 3. Inspecting the Cleaned Columns

A quick profile of the DataFrames was generated to verify numeric conversion and detect abnormalities:

    for name, df in dfs.items():
        print(name, df.describe().T[['count','mean','std','min','max']].head(25))

This helped confirm that the structure and values looked consistent.

---

## 4. Dropping Constant Columns

Columns containing only a single repeated value (usually all zeros) provide no useful information and were removed.

    for name, df in dfs.items():
        zero_cols = [c for c in df.columns if df[c].nunique() == 1]
        dfs[name] = df.drop(columns=zero_cols)
        print(f"Dropped from {name}: {zero_cols}")

---

## 5. Removing the Timestamp Column

The timestamp column was removed to avoid the model learning temporal patterns or memorizing time-based identifiers.

    for name, df in dfs.items():
        if 'timestamp' in df.columns:
            df = df.drop(columns=['timestamp'])
        dfs[name] = df

---

## 6. Numeric Conversion and Missing Value Handling

Next, label and protocol were left as raw values while all other columns were converted to numeric. Infinite values were replaced, then missing values were filled with column medians.

    safe_numeric_cols = ['label','protocol']

    for name in list(dfs.keys()):
        df = dfs[name]

        # normalize column names again for safety
        df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')

        # convert each column to numeric except label/protocol
        for c in df.columns:
            if c not in safe_numeric_cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # replace inf -> nan -> median
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)

        dfs[name] = df
        print(name, "done", df.shape)

---

## 7. Building the Training DataFrame

Once each dataset file was cleaned, the training days were combined. Protocol and label columns were encoded as numeric values for model compatibility.

    # protocol to numeric
    train_df['protocol'] = pd.to_numeric(train_df['protocol'], errors='coerce')

    # label -> categorical index encoding
    train_df['label'] = train_df['label'].astype('category').cat.codes

    # verify all non-label columns numeric
    for c in train_df.columns:
        if c != 'label':
            train_df[c] = pd.to_numeric(train_df[c], errors='coerce')

    # final missing fill
    train_df = train_df.fillna(train_df.median(numeric_only=True))

---

## 8. Scaling the Training Data

Label was removed from the feature set and the remaining features were standardized using StandardScaler.

    X = train_df.drop(columns=['label'])
    y = train_df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

Scaling ensures that all features share similar ranges and prevents high-value features from dominating model training.

---

## 9. Handling Class Imbalance

Model training initially failed because one of the classes contained only a single sample.  
To improve class balance, the four smallest classes were removed:

    Dropping classes: [10, 11, 2, 1]

Resulting class distribution:

    label
    0     3519521
    3      686012
    6      461912
    9      193360
    12     187589
    7      139890
    5       41508
    8       10990
    4        1730
    Name: count, dtype: int64

Note: In hindsight, SMOTE could be explored, but very small class sizes make synthetic oversampling difficult.

---

## 10. Final Step: Model Training

After completing all preprocessing steps, the dataset was fully cleaned, scaled, and ready for model training.

Training was performed using the processed feature matrix `X_scaled` and encoded labels `y`.

---

## Summary

These preprocessing steps ensured that:

- All features were numeric  
- Missing and infinite values were addressed  
- Dataset files were consistent across days  
- Columns with no variance were dropped  
- Labels and protocol fields were encoded correctly  
- Features were scaled for training  
- Class imbalance was managed  

This stage provided a clean and reliable dataset for machine learning model development.

---
