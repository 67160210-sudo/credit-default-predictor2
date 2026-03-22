# ===============================
# IMPORT
# ===============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

import joblib
import os
import json

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv("credit_default_risk.csv")

print("Shape:", df.shape)
print(df.head())

# ===============================
# PREPARE DATA
# ===============================
target_col = "default"  # ⚠️ เช็คชื่อให้ตรง

X = df.drop(target_col, axis=1)
y = df[target_col]

feature_names = X.columns.tolist()

# ===============================
# SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# PIPELINE
# ===============================
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    ))
])

# ===============================
# TRAIN
# ===============================
pipeline.fit(X_train, y_train)

# ===============================
# EVALUATE
# ===============================
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

print(classification_report(y_test, y_pred))

# ===============================
# SAVE MODEL
# ===============================
os.makedirs("model_artifacts", exist_ok=True)

joblib.dump(pipeline, "model_artifacts/model.pkl")

with open("model_artifacts/feature_names.json", "w") as f:
    json.dump(feature_names, f)

metadata = {
    "accuracy": float(acc),
    "features": feature_names
}

with open("model_artifacts/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Model saved!")