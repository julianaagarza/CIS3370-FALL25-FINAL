import pandas as pd
import numpy as np
import pickle
import json
import time
import warnings
from datetime import datetime
from elasticsearch import Elasticsearch
import shap
import random

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = "best_rf_pipeline.pkl"
CSV_FILE = "02-21-2018.csv"  # We use a test day file to simulate traffic

# Elasticsearch Config
ES_HOST = "http://localhost:9200"
INDEX_NAME = "ids-traffic-live"

# Speed Control
SLEEP_INTERVAL = 0.5  # Seconds between packets (lower = faster)

# ------------------------------------------
# LABEL MAPPING (Verified from my artifacts)
# ------------------------------------------
LABEL_MAP = {
    0: "Benign",
    1: "Brute Force -Web",
    2: "Brute Force -XSS",
    3: "DDOS attack-HOIC",
    4: "DDOS attack-LOIC-UDP",
    5: "DoS attacks-GoldenEye",
    6: "DoS attacks-Hulk",
    7: "DoS attacks-SlowHTTPTest",
    8: "DoS attacks-Slowloris",
    9: "FTP-BruteForce",
    10: "Label",  # Artifact from header rows, safe to ignore
    11: "SQL Injection",
    12: "SSH-Bruteforce",
}

# ==========================================
# 2. INITIALIZATION
# ==========================================
print(f"--- INITIALIZING IDS SIMULATION ---")

# Connect to Elasticsearch
# ... inside simulateids.py ...

# Connect to Elasticsearch
try:
    es = Elasticsearch(
        "http://localhost:9200",
        basic_auth=("elastic", "changeme")
    )
    
    if es.ping():
        print(f"Connected to Elasticsearch at {ES_HOST}")
    else:
        print(f"Auth failed. Did you change the password from 'changeme'?")
        print("   (Check the password you used in the browser)")
        exit()
except Exception as e:
    print(f" Connection Error: {e}")
    exit()

# Load Model
print(f" Loading model from {MODEL_PATH}...")
try:
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Could not find {MODEL_PATH}. Did you download it from Drive?")
    exit()

# Extract the classifier step for SHAP
# (Pipeline structure: [('smote'), ('scaler'), ('clf')])
model = pipeline.named_steps['clf']

# Initialize SHAP Explainer
# We use TreeExplainer because it is optimized for Random Forest
print("Initializing SHAP Explainer (this may take 10s)...")
explainer = shap.TreeExplainer(model)

# ==========================================
# 3. DATA LOADING & PREPARATION
# ==========================================
print(f"Loading traffic data from {CSV_FILE}...")
try:
    # We read 5000 rows to simulate a stream. 
    # low_memory=False helps prevent type warnings
    df_raw = pd.read_csv(CSV_FILE, nrows=5000, low_memory=False) 
    print(f"   Loaded {len(df_raw)} packets.")
except FileNotFoundError:
    print(f" Error: Could not find {CSV_FILE}. Make sure it is in this folder.")
    exit()

# Clean Column Names (Must match training logic!)
df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(' ', '_')

# Define Feature Columns
# These are the columns we MUST drop to match the model's training input
cols_to_drop = ['label', 'label_str', 'timestamp', 'flow_id', 'src_ip', 'dst_ip', 'source_ip', 'destination_ip']

# ==========================================
# 4. SIMULATION LOOP
# ==========================================
print("\n STARTING REAL-TIME TRAFFIC REPLAY...")
print(f"{'TIMESTAMP':<25} {'SOURCE':<16} {'DEST':<16} {'PREDICTION':<20} {'CONF'}")
print("-" * 90)

try:
    for index, row in df_raw.iterrows():
        # --------------------------------------
        # A. PREPARE FEATURES
        # --------------------------------------
       
        # 1. Keep "Display" info for the Dashboard (JSON)
        timestamp_original = row.get('timestamp', datetime.now().isoformat())
        
        # --- IP ADDRESS LOGIC---
        # If the file has real IPs, use them. If not, generate random ones 
        # so the Dashboard looks active and cool.
        
        # Source IP
        if 'src_ip' in row and pd.notna(row['src_ip']):
            src_ip = row['src_ip']
        else:
            # Generate random internal IPs (e.g., 192.168.1.X)
            src_ip = f"192.168.1.{random.randint(2, 254)}"

        # Destination IP
        if 'dst_ip' in row and pd.notna(row['dst_ip']):
            dst_ip = row['dst_ip']
        else:
            # Generate random external IPs (e.g., 10.0.0.X)
            dst_ip = f"10.0.{random.randint(0, 5)}.{random.randint(2, 254)}"

        # Ports and Protocol (Keep existing logic)
        dst_port = int(row.get('dst_port', 0))
        protocol = int(row.get('protocol', 0))
        
        # 2. Prepare "Model" input (Drop ID columns)
        # We convert the single row to a DataFrame
        input_df = row.to_frame().T
        # Drop the forbidden columns (ignore if they don't exist)
        features_df = input_df.drop(columns=cols_to_drop, errors='ignore')
        
        # 3. Ensure Numeric (Match training)
        features_df = features_df.apply(pd.to_numeric, errors='coerce')
        # FIX: Replace Infinity with 0 to prevent crashes
        features_df = features_df.replace([np.inf, -np.inf], 0).fillna(0)
        
        # --------------------------------------
        # B. PREDICT
        # --------------------------------------
        # Predict Class
        pred_label_code = pipeline.predict(features_df)[0]
        pred_label_name = LABEL_MAP.get(pred_label_code, f"Class-{pred_label_code}")
        
        # Predict Confidence (Probability)
        # Handle cases where model might not support predict_proba
        try:
            probs = pipeline.predict_proba(features_df)[0]
            confidence = float(np.max(probs))
        except:
            confidence = 1.0 # Fallback if proba not available
        
        # Determine if Malicious (0 is Benign)
        is_malicious = 0 if pred_label_code == 0 else 1

        # --------------------------------------
        # C. EXPLAIN (SHAP) - Only for Attacks
        # --------------------------------------
        top_features = []
        if is_malicious == 1 and pred_label_code != 10: # Skip SHAP for benign or artifacts
            try:
                # Calculate SHAP for this specific row
                shap_values = explainer.shap_values(features_df)
                
                # SHAP returns a list of arrays (one for each class).
                # We need the array index corresponding to the predicted class.
                class_idx = list(model.classes_).index(pred_label_code)
                explanation = shap_values[class_idx][0]
                
                # Get indices of top 3 features
                top_indices = np.argsort(np.abs(explanation))[-3:][::-1]
                
                feature_names = features_df.columns
                for idx in top_indices:
                    feat_name = feature_names[idx]
                    feat_val = explanation[idx]
                    top_features.append({"feature": feat_name, "shap_value": float(feat_val)})
            except Exception as e:
                # Don't crash the simulation if SHAP fails on one row
                # print(f"SHAP Error: {e}") 
                pass 

        # --------------------------------------
        # D. SEND TO ELASTICSEARCH
        # --------------------------------------
        doc = {
            "timestamp": datetime.now().isoformat(), # Use current time for "Live" dashboard
            "original_ts": str(timestamp_original),
            "src_ip": str(src_ip),
            "dst_ip": str(dst_ip),
            "dst_port": dst_port,
            "protocol": protocol,
            "prediction_label": pred_label_name,
            "confidence": confidence,
            "is_malicious": is_malicious,
            "shap_explanation": top_features, # List of dicts
            "flow_duration": float(row.get('flow_duration', 0)),
            "packet_count": float(row.get('tot_fwd_pkts', 0)) + float(row.get('tot_bwd_pkts', 0))
        }

        # Send to ES
        es.index(index=INDEX_NAME, document=doc)

        # --------------------------------------
        # E. CONSOLE OUTPUT
        # --------------------------------------
        # Color code: Red for attack, Green for benign
        if is_malicious:
            color_start = "\033[91m" # Red
            color_end = "\033[0m"
        else:
            color_start = "\033[92m" # Green
            color_end = "\033[0m"

        print(f"{color_start}{doc['timestamp']}   {str(src_ip):<16} {str(dst_ip):<16} {pred_label_name:<20} {confidence:.2f}{color_end}")
        
        # Sleep to simulate real-time traffic flow
        time.sleep(SLEEP_INTERVAL)

except KeyboardInterrupt:
    print("\n Simulation stopped by user.")
except Exception as e:
    print(f"\n Error in loop: {e}")
