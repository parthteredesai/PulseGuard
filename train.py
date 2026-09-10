import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # 1. Ensure directory structure exists
    os.makedirs("models", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)

    dataset_path = os.path.join("dataset", "heart.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Please download 'heart.csv' from Kaggle (UCI Heart Disease Dataset) and place it in the 'dataset/' folder."
        )

    # 2. Load Dataset
    print("[1/5] Loading UCI Heart Disease dataset...")
    df = pd.read_csv(dataset_path)

    # Standard UCI dataset features check:
    # age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 3. Handle Missing Values
    print("[2/5] Cleaning data and checking for missing values...")
    # Replace '?' or invalid strings if using older raw UCI CSV versions
    df = df.replace("?", np.nan)
    df = df.dropna()

    # Define Feature Matrix (X) and Target Vector (y)
    # Target: 1 = Presence of Heart Disease, 0 = Absence
    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    # 4. Train/Test Split & Feature Scaling
    print("[3/5] Splitting data and applying StandardScaler...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Model Training (Random Forest Classifier)
    print("[4/5] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate Model Performance
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n--- Model Evaluation Summary ---")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 6. Save Model Artifacts
    print("[5/5] Exporting model.pkl and scaler.pkl...")
    model_path = os.path.join("models", "model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nSUCCESS: Saved trained model to '{model_path}'")
    print(f"SUCCESS: Saved feature scaler to '{scaler_path}'")


if __name__ == "__main__":
    main()