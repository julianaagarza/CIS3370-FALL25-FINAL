import pickle

# 1. Load the artifacts
print("Loading project_artifacts.pkl...")
with open("project_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

# 2. Extract the mapping
# We saved this specifically in the previous step
true_mapping = artifacts.get('class_mapping')

if true_mapping is None:
    # Fallback: Reconstruct it from the LabelEncoder if the dict is missing
    le = artifacts['label_encoder']
    true_mapping = dict(zip(le.transform(le.classes_), le.classes_))

# 3. Print it in Python format
print("\n=== COPY AND PASTE THIS INTO simulate_ids.py ===")
print("LABEL_MAP = {")
for code, label in sorted(true_mapping.items()):
    print(f"    {code}: \"{label}\",")
print("}")
print("================================================")
