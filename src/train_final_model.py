
"""Train and save the Iris classifier for Docker deployment."""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/processed/iris_features.csv"
MODEL_DIR = "models"

FEATURE_COLS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]


def main():
    """Train the classifier and save its model artifacts."""
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLS].copy()
    X = X.fillna(X.median())

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["species"])

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )
    model.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "final_model.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    print("Training completed.")
    print(f"Rows used: {len(df)}")
    print(f"Features: {FEATURE_COLS}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print("Saved model and label encoder.")


if __name__ == "__main__":
    main()
