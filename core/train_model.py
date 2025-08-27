#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# train model
#!/usr/bin/env python
# coding: utf-8

# -----------------------------
# Train Model for Disaster Prediction
# -----------------------------
#!/usr/bin/env python
# coding: utf-8

# train model
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# LightGBM and XGBoost
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ------------------------------
# Paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "combined_disaster_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

print(f"📂 Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# ------------------------------
# Prepare features & labels
# ------------------------------
if "disaster_type" not in df.columns:
    raise ValueError("❌ No 'disaster_type' column found in dataset!")

X = df.drop(columns=["disaster_type"])
X = X.select_dtypes(include=["int64", "float64"])  # keep numeric only
y = df["disaster_type"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# ------------------------------
# Define models
# ------------------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
}

# Voting Classifier (soft voting across all models)
voting_clf = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting="soft"
)
models["VotingClassifier"] = voting_clf

# ------------------------------
# Train, evaluate, save
# ------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n📊 Classification Report for {name}:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Save model + encoder together
    save_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
    joblib.dump({"model": model, "label_encoder": label_encoder}, save_path)
    print(f"✅ Saved {name} at: {save_path}")
