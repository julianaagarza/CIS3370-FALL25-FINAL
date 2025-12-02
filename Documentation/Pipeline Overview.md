# Preprocessing & Feature Engineering Pipeline Overview

This document explains how our system cleans, validates, and prepares the CICIDS2018 dataset for machine learning. Each main section includes a short description of its purpose and how it fits into the full workflow.

---

## 1. Overview of the Pipeline

*This section provides a high-level explanation of what the preprocessing workflow accomplishes and why it is required.*

Our preprocessing workflow transforms the raw CICIDS2018 CSV files into a clean, standardized, machine-learning-ready dataset. It handles:

- Column name normalization  
- Data type repair and numeric conversion  
- Missing and infinite value handling  
- Label encoding  
- Feature consistency across days  
- Train/test dataset assembly  
- Feature scaling (standardization)  

The final output includes:

- X_train_scaled, X_test_scaled → numeric, standardized feature matrices  
- y_train, y_test → integer-encoded labels  
- Clean data stored in memory or optionally exported  

---

## 2. Step-by-Step Flow

*This section describes each preprocessing step in the exact order it occurs, from loading the dataset to scaling the features.*

### 2.1. Load all CICIDS2018 CSVs

Each CSV day is loaded into a dictionary:

    dfs = {file: pd.read_csv(path) for file in csv_files}

Benefits:

- Validate each day individually  
- Clean each day separately  
- Regroup conveniently for training/testing  

---

### 2.2. Normalize Column Names

Raw column names contain inconsistent spacing, casing, or symbols.

    df.columns = (
        df.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
    )

Example:

| Raw Column Name | Normalized |
|-----------------|------------|
| Flow Duration | flow_duration |
| Bwd PSH Flags | bwd_psh_flags |
| Fwd Pkts/s | fwd_pkts/s |

Benefit: Prevents scripts from breaking due to inconsistent naming.

---

### 2.3. Convert All Features to Numeric

Common raw data problems:

- Whitespace  
- Stray characters  
- Values like `"NaN"`, `"Infinity"`, `"Nan"`  
- Integers stored as strings  

Process:

    safe = ['label', 'protocol']

    for c in df.columns:
        if c not in safe:
            df[c] = pd.to_numeric(df[c], errors='coerce')

Any non-convertible value becomes NaN.

---

### 2.4. Replace Infinities and Fill Missing Values

Dataset issues include division-by-zero, corrupted values, and missing entries.

Fix:

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

Why median?

- Robust to outliers  
- Works well for skewed traffic data  

---

### 2.5. Encode Labels + Protocol

**Label Encoding**

The dataset contains labels such as:

- BENIGN  
- DoS Hulk  
- Bot  
- Infiltration  
- SSH-BruteForce  

Encoding:

    df['label'] = df['label'].astype('category').cat.codes

**Protocol Encoding**

    df['protocol'] = pd.to_numeric(df['protocol'], errors='coerce')

---

### 2.6. Consistency Check Across All Days

After cleaning each day:

- All CSVs must share identical column sets  
- Column order must match  
- All features must be numeric except label  
- Extra/missing columns are logged  

This ensures training and testing datasets align perfectly.

---

### 2.7. Merge Training Days & Testing Days

Training dataset:

    train_days = ['02-14-2018.csv','02-15-2018.csv','02-16-2018.csv','02-21-2018.csv','02-22-2018.csv']
    train_df = pd.concat([dfs[d] for d in train_days], ignore_index=True)

Testing dataset:

    test_days = ['02-28-2018.csv','03-01-2018.csv','03-02-2018.csv']
    test_df = pd.concat([dfs[d] for d in test_days], ignore_index=True)

Final training size: ~5.2 million rows × ~79 features

---

### 2.8. Final Scaling (Standardization)

Standardization ensures:

- Mean = 0  
- Variance = 1  
- Faster convergence  
- No feature dominates  

Scaling:

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df.drop('label', axis=1))
    X_test_scaled  = scaler.transform(test_df.drop('label', axis=1))

---

## 3. Summary of Features Used

*This section lists the feature categories included in the final cleaned dataset.*

The final dataset includes:

- 79 fully numeric flow-based features  
- Encoded label and protocol  
- Derived strictly from CICIDS2018 (not Suricata or PCAP)  

Feature categories include:

| Category | Examples |
|----------|----------|
| Traffic volume | tot_fwd_packets, tot_bwd_pkts, flow_duration |
| Timing | fwd_iat_mean, bwd_iat_max, flow_iat_std |
| Packet sizes | pkt_len_mean, pkt_len_std, subflow_bwd_bytes |
| Flags | fwd_psh_flags, bwd_urg_flags |
| Ratios & rates | fwd_pkts/s, bwd_byts/s |

The preprocessing pipeline ensures consistency across all days.

---

## 4. What This Enables

*This section explains what becomes possible after preprocessing completes successfully.*

After preprocessing, we can:

- Train machine learning models  
- Compute SHAP explainability values  
- Export predictions into Elasticsearch  
- Visualize performance in a Kibana dashboard  

---
