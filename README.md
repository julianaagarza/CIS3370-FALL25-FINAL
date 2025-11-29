# Encrypted Traffic Intrusion Detection System
---

# Machine Learning + SHAP Explainability + Elasticsearch/Kibana Visualization
This project implements a machine learning–based Intrusion Detection System (IDS) for encrypted network traffic, built using the CICIDS2018 dataset. The system includes preprocessing, feature engineering, model training, SHAP explainability, and visualization using Elasticsearch and Kibana.

---

# Project Overview
- The project contains:
* End-to-end ML pipeline for encrypted traffic detection
* Data preprocessing and cleaning
* Feature selection using permutation importance
* Random Forest/XGBoost model training
* SHAP explainability outputs
* Export of predictions and explanations to Elasticsearch
* Kibana dashboard for analysis and visualization

---

## Dataset: CICIDS2018 

Training days used:
- 02-14-2018.csv
- 02-15-2018.csv
- 02-16-2018.csv
- 02-21-2018.csv
- 02-22-2018.csv
  
Testing and evaluation days:
- 02-28-2018.csv
- 03-01-2018.csv
- 03-02-2018.csv

---

## ML Pipeline Summary

1. Preprocessing
   
- Clean column names
- Convert to numeric
- Handle NaN and infinity values
- Normalize and scale data

2. Feature Selection
   
- Use permutation importance
- Select top ~20–30 features
- Retrain model with selected features

3. Model Training
- Train Random Forest or XGBoost
- Hyperparameter tuning
- Evaluate on unseen days
- Save metrics and confusion matrix

4. Explainability (SHAP)
- Generate SHAP values for test set
- Identify high-influence traffic features
- Export SHAP summaries with model predictions
- Exporting Results to Elasticsearch

5. Kibana Dashboard
- The Kibana dashboard includes:
- Alert table (predictions with features)
- Class distribution visualizations
- SHAP feature impact bar charts
- Confusion matrix
- Time-series of alerts
- Feature histograms for malicious flows

---

## Milestones
The project follows five milestones:
1. Setup and Data Preparation
2. Feature Extraction and Data Engineering
3. ML Model Development
4. Threat Intelligence / Elasticsearch Integration
5. Analysis, Testing, and Finalization
All tasks, priorities, and estimates are in GitHub Issues and the Wiki.

## Team Members:
- Sarah Steadham
- Juliana Garza
  
---

## Quick Start

1. Clone the Repository
git clone https://github.com/your-org/your-repo.git
cd your-repo
2. Open the Colab Notebook
Upload model_training.ipynb to Google Colab.
3. Install Dependencies
Inside Colab:
!pip install shap elasticsearch pandas numpy scikit-learn
4. Train the Model
Run preprocessing, feature selection, SHAP, and evaluation.
5. Export Predictions to Elasticsearch
python src/export_to_elasticsearch.py
6. View Dashboard in Kibana
Import the dashboard or build one directly from your indexed data.

---

## Documentation
Additional documentation:
- Preprocessing walkthrough
- Dataset description
- Elasticsearch mapping
- Dashboard build guide
- SHAP explainability notes

---

## Final Deliverables

- Clean feature pipeline
- Trained model + evaluation metrics
- SHAP explainability graphs
- Elasticsearch indexed predictions
- Kibana dashboard
- Final written report
- Fully documented repository
