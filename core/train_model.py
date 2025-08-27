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
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# -----------------------------
# Path Setup
# -----------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root of repo
DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "combined_disaster_data.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "disaster_model.pkl")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# -----------------------------
# Load Data
# -----------------------------
print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Drop completely empty columns (if any exist)
df = df.dropna(axis=1, how="all")

# -----------------------------
# Define Features and Targets
# -----------------------------
# Label columns (multi-label setup)
LABEL_COLS = ["disaster_type_encoded", "is_earthquake", "is_flood", "is_landslide", "is_cyclone"]
LABEL_COLS = [col for col in LABEL_COLS if col in df.columns]

if not LABEL_COLS:
    raise ValueError("No label columns found in dataset!")

X = df.drop(columns=LABEL_COLS, errors="ignore")
Y = df[LABEL_COLS].astype(int)   # Convert labels to integers (0/1)

# --- FIX: drop non-numeric features (e.g., date, text) ---
X = X.select_dtypes(include=[np.number]).copy()

# Handle NaN in features (basic imputation)
X = X.fillna(0)

print(f"Features: {X.shape[1]}, Labels: {len(LABEL_COLS)}")
print(f"Label columns: {LABEL_COLS}")

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# -----------------------------
# Define Models
# -----------------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
}

# -----------------------------
# Train & Evaluate
# -----------------------------
best_model = None
best_score = -1

for name, model in models.items():
    print(f"\nTraining {name}...")
    try:
        scores = cross_val_score(model, X_train, Y_train, cv=3, scoring="accuracy")
        mean_score = np.mean(scores)
        print(f"{name} CV Accuracy: {mean_score:.4f}")
    except Exception as e:
        print(f"⚠️ Skipping CV for {name} due to error:\n{e}")

    # Fit model on full training set
    try:
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(Y_test, preds)
        print(f"{name} Test Accuracy: {acc:.4f}")
        print(classification_report(Y_test, preds))
        if acc > best_score:
            best_score = acc
            best_model = model
    except Exception as e:
        print(f"⚠️ Skipping training for {name} due to error:\n{e}")

# -----------------------------
# Voting Classifier (Ensemble)
# -----------------------------
print("\nTraining Voting Classifier...")
try:
    voting_clf = VotingClassifier(
        estimators=[(name, mdl) for name, mdl in models.items()],
        voting="soft"
    )
    voting_clf.fit(X_train, Y_train)
    preds = voting_clf.predict(X_test)
    acc = accuracy_score(Y_test, preds)
    print(f"VotingClassifier Test Accuracy: {acc:.4f}")
    print(classification_report(Y_test, preds))
    if acc > best_score:
        best_model = voting_clf
        best_score = acc
except Exception as e:
    print(f"⚠️ Skipping VotingClassifier due to error:\n{e}")

# -----------------------------
# Save Best Model
# -----------------------------
if best_model is not None:
    print(f"\nBest model selected with accuracy: {best_score:.4f}")
    joblib.dump(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
else:
    print("❌ No model was successfully trained!")
