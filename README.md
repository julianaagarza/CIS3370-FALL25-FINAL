# Encrypted Traffic Intrusion Detection System
---
# Machine Learning IDS with SHAP Explainability and Elasticsearch/Kibana Visualization

This project implements a machine learning based Intrusion Detection System (IDS) for encrypted network traffic using the CICIDS2018 dataset. It includes preprocessing, feature engineering, ML model training, SHAP explainability, and visualization using Elasticsearch and Kibana.

This repository is part of a semester-long project in CIS 3370 (Intrusion Detection Systems) and demonstrates how statistical machine learning and model interpretability techniques can be applied to encrypted network traffic analysis.

---

## Project Summary

This project provides an end to end intrusion detection pipeline that includes:

- Data preprocessing and cleaning  
- Feature selection using permutation importance  
- Random Forest or XGBoost model training  
- SHAP explainability for model transparency  
- Export of predictions and SHAP values to Elasticsearch  
- Dashboard visualization in Kibana  

The goal is to detect malicious encrypted flows and provide clear explanations for all model decisions.

---

## Dataset: CICIDS2018

Selected labeled flow files from CICIDS2018 were used to form the training and evaluation sets.

Training days:

- 02-14-2018  
- 02-15-2018  
- 02-16-2018  
- 02-21-2018  
- 02-22-2018  

Testing and evaluation days:

- 02-28-2018  
- 03-01-2018  
- 03-02-2018  

Each CSV file contains approximately 80 flow-based numeric features and benign or malicious labels.

For full dataset details, refer to the Dataset Overview page in the Wiki.

---

## Machine Learning Pipeline Overview

1. Preprocessing  
   - Normalize column names  
   - Convert all features to numeric  
   - Handle NaN and infinity values  
   - Standardize feature scales  

2. Feature Selection  
   - Apply permutation importance  
   - Select approximately 20 to 30 top features  

3. Model Training  
   - Train Random Forest or XGBoost  
   - Perform hyperparameter tuning  
   - Evaluate using unseen test days  
   - Export confusion matrix and evaluation metrics  

4. Explainability (SHAP)  
   - Compute SHAP values for test samples  
   - Identify influential traffic features  
   - Export SHAP summaries with model outputs  

5. Export to Elasticsearch  
   - Index model predictions  
   - Index SHAP values for dashboard use  

6. Kibana Dashboard  
   - Alerts table (predicted vs true labels)  
   - Class distribution visualizations  
   - SHAP feature impact charts  
   - Confusion matrix  
   - Time series of alerts  

Detailed walkthroughs of each stage are available in the project Wiki.

---

## Quick Start Guide

Clone the repository:

    git clone https://github.com/your-org/your-repo.git
    cd your-repo

Open the Colab notebook:

Upload or open `model_training.ipynb` in Google Colab.

Install required dependencies:

    !pip install shap elasticsearch pandas numpy scikit-learn xgboost

Run the preprocessing, feature selection, model training, SHAP explainability, and evaluation steps inside Colab.

Export results to Elasticsearch:

    python src/export_to_elasticsearch.py

Open Kibana to view the dashboard and visualizations.

---

## Team Members

- Sarah Steadham  
- Juliana Garza  

---

## Documentation

The complete documentation is available in the GitHub Wiki.  
Pages include:

- Environment Setup  
- Dataset Overview  
- Preprocessing and Feature Engineering  
- Model Development  
- Elasticsearch Integration  
- Kibana Dashboard Guide  
- SHAP Explainability Notes  

The Wiki contains all detailed instructions, diagrams, walkthroughs, and design decisions.

---

## Final Deliverables

- Clean data preprocessing and feature pipeline  
- Trained model and evaluation metrics  
- SHAP explainability outputs  
- Elasticsearch indexed predictions  
- Kibana dashboard  
- Final written report  
- Fully documented project repository  

---

## License

Educational project for CIS 3370: Intrusion Detection Systems.  
Not intended for production use.

