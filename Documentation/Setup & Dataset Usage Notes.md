# Setup & Dataset Usage Notes

This document explains how to set up the environment and work with the **CICIDS2018 dataset** for our Intrusion Detection System (IDS) Machine Learning Pipeline. These instructions assume you are using **Google Colab**.

---

## 1. Environment Setup (Google Colab)

### 1.1. Open Google Colab
You can begin with a new notebook, a GitHub notebook, or a notebook stored in your Google Drive. Once opened, continue with the environment setup steps below.

---

### 1.2. Install Required Python Packages

Run the following cell at the top of your notebook:

```python
!pip install pandas numpy scikit-learn shap matplotlib seaborn elasticsearch
```
## Package Overview

These packages support all major parts of the project:

| Package | Purpose |
|--------|---------|
| pandas / numpy | Data cleaning and preprocessing |
| scikit-learn | Machine learning models, evaluation, splitting |
| shap | Explainability and feature importance |
| matplotlib / seaborn | Visualizations and plotting |
| elasticsearch | Optional — send results/logs to Elasticsearch |

---

### 1.3. Mount Google Drive

Run this cell to mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```
Your dataset should be stored in:

`/content/drive/MyDrive/idsprojectcsvs/`

Make sure all **CICIDS2018 CSV files** are saved in this folder for easy loading.  

---





## 2. Dataset Usage Notes

### 2.1. Loading the CICIDS2018 Dataset

Because CICIDS2018 files are large, chunked loading is recommended:

    import pandas as pd

    chunks = pd.read_csv(
        "/content/drive/MyDrive/idsprojectcsvs/Thursday-WorkingHours-Morning-WebAttacks.csv",
        chunksize=100000
    )

    df = pd.concat(chunks)

This helps avoid memory issues and speeds up processing.

---

### 2.2. Common Data Fixes

Some CSV files may contain NaN or infinite values that must be cleaned before modeling:

    df.replace([float("inf"), -float("inf")], 0, inplace=True)
    df.dropna(inplace=True)

Additional cleanup steps may include:

- Removing extra unnamed index columns  
- Converting categorical values to numeric  
- Ensuring column consistency across all dataset files  

---

### 2.3. Verify Dataset Directory

Confirm that your dataset folder is accessible after mounting Google Drive:

    import os

    data_path = "/content/drive/MyDrive/idsprojectcsvs/"
    print(os.listdir(data_path))

Expected output should include files such as:

- Monday-WorkingHours.pcap_ISCX.csv  
- Thursday-WorkingHours-Morning-WebAttacks.csv  
- Friday-WorkingHours-Afternoon-DDos.csv  
(and others depending on what you downloaded)

If no files appear, double-check:

- The folder name  
- File placement in Google Drive  
- That your Drive mounted correctly  


