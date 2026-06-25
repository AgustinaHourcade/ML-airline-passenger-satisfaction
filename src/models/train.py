import json
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# Ensure src is in Python path for relative imports if run as a script
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.preprocessing import load_raw_data, preprocess_phase1, preprocess_phase2, PRE_FLIGHT_FEATURES, POST_FLIGHT_FEATURES
from src.models.pipeline import get_phase1_pipeline, get_phase2_pipeline

def save_metadata(path, model_name, features, acc, f1, hyperparams):
    metadata = {
        "model_name": model_name,
        "training_date": datetime.now().isoformat(),
        "features": features,
        "hyperparameters": hyperparams,
        "validation_metrics": {
            "accuracy": round(acc, 4),
            "f1_score_weighted": round(f1, 4)
        }
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {path}")

def main():
    print("Loading processed data...")
    
    processed_dir = Path("data") / "processed"
    p1_dir = processed_dir / "phase1"
    p2_dir = processed_dir / "phase2"
    
    # Check if processed data exists
    if not p1_dir.exists() or not p2_dir.exists():
        print("Processed data not found. Please run 'python -m src.data.make_dataset' first.")
        return

    # Define models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # --- Phase 1 ---
    print("\n--- Training Phase 1 ---")
    X_train_1 = pd.read_csv(p1_dir / "X_train.csv")
    X_test_1 = pd.read_csv(p1_dir / "X_test.csv")
    y_train_1 = pd.read_csv(p1_dir / "y_train.csv").squeeze()
    y_test_1 = pd.read_csv(p1_dir / "y_test.csv").squeeze()
    
    pipe_1 = get_phase1_pipeline('rf')
    
    print("Fitting model...")
    pipe_1.fit(X_train_1, y_train_1)
    
    y_pred_1 = pipe_1.predict(X_test_1)
    acc_1 = accuracy_score(y_test_1, y_pred_1)
    f1_1 = f1_score(y_test_1, y_pred_1, average='weighted')
    
    print(f"Phase 1 Metrics - Accuracy: {acc_1:.4f}, F1: {f1_1:.4f}")
    
    joblib.dump(pipe_1, models_dir / "model_phase1_v1.joblib")
    print("Model saved to models/model_phase1_v1.joblib")
    
    save_metadata(
        models_dir / "metadata_phase1_v1.json",
        "Phase 1 - Pre-flight (Random Forest)",
        PRE_FLIGHT_FEATURES,
        acc_1,
        f1_1,
        pipe_1.named_steps['clf'].get_params()
    )
    
    # --- Phase 2 ---
    print("\n--- Training Phase 2 ---")
    X_train_2 = pd.read_csv(p2_dir / "X_train.csv")
    X_test_2 = pd.read_csv(p2_dir / "X_test.csv")
    y_train_2 = pd.read_csv(p2_dir / "y_train.csv").squeeze()
    y_test_2 = pd.read_csv(p2_dir / "y_test.csv").squeeze()
    
    pipe_2 = get_phase2_pipeline('rf')
    
    print("Fitting model...")
    pipe_2.fit(X_train_2, y_train_2)
    
    y_pred_2 = pipe_2.predict(X_test_2)
    acc_2 = accuracy_score(y_test_2, y_pred_2)
    f1_2 = f1_score(y_test_2, y_pred_2, average='weighted')
    
    print(f"Phase 2 Metrics - Accuracy: {acc_2:.4f}, F1: {f1_2:.4f}")
    
    joblib.dump(pipe_2, models_dir / "model_phase2_v1.joblib")
    print("Model saved to models/model_phase2_v1.joblib")
    
    # Phase 2 includes both pre-flight and post-flight features
    all_phase2_features = PRE_FLIGHT_FEATURES + POST_FLIGHT_FEATURES
    
    save_metadata(
        models_dir / "metadata_phase2_v1.json",
        "Phase 2 - Pre and Post-flight (Random Forest)",
        all_phase2_features,
        acc_2,
        f1_2,
        pipe_2.named_steps['clf'].get_params()
    )
    
    print("\nDone! Stage 2 completed.")

if __name__ == "__main__":
    main()
