import json
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
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
    
    param_dist = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 15, 30],
        'clf__min_samples_split': [2, 5],
        'clf__min_samples_leaf': [1, 2]
    }
    
    search_1 = RandomizedSearchCV(
        pipe_1, param_distributions=param_dist, n_iter=3, cv=3, 
        scoring='f1_weighted', random_state=42, n_jobs=-1, verbose=1
    )
    
    print("Fitting model with RandomizedSearchCV (Phase 1)...")
    search_1.fit(X_train_1, y_train_1)
    
    best_pipe_1 = search_1.best_estimator_
    y_pred_1 = best_pipe_1.predict(X_test_1)
    acc_1 = accuracy_score(y_test_1, y_pred_1)
    f1_1 = f1_score(y_test_1, y_pred_1, average='weighted')
    
    print(f"Phase 1 Metrics - Accuracy: {acc_1:.4f}, F1: {f1_1:.4f}")
    print(f"Best params: {search_1.best_params_}")
    
    joblib.dump(best_pipe_1, models_dir / "model_phase1_v1.joblib")
    print("Model saved to models/model_phase1_v1.joblib")
    
    save_metadata(
        models_dir / "metadata_phase1_v1.json",
        "Phase 1 - Pre-flight (Random Forest)",
        PRE_FLIGHT_FEATURES,
        acc_1,
        f1_1,
        best_pipe_1.named_steps['clf'].get_params()
    )
    
    # --- Phase 2 ---
    print("\n--- Training Phase 2 ---")
    X_train_2 = pd.read_csv(p2_dir / "X_train.csv")
    X_test_2 = pd.read_csv(p2_dir / "X_test.csv")
    y_train_2 = pd.read_csv(p2_dir / "y_train.csv").squeeze()
    y_test_2 = pd.read_csv(p2_dir / "y_test.csv").squeeze()
    
    pipe_2 = get_phase2_pipeline('rf')
    
    search_2 = RandomizedSearchCV(
        pipe_2, param_distributions=param_dist, n_iter=3, cv=3, 
        scoring='f1_weighted', random_state=42, n_jobs=-1, verbose=1
    )
    
    print("Fitting model with RandomizedSearchCV (Phase 2)...")
    search_2.fit(X_train_2, y_train_2)
    
    best_pipe_2 = search_2.best_estimator_
    y_pred_2 = best_pipe_2.predict(X_test_2)
    acc_2 = accuracy_score(y_test_2, y_pred_2)
    f1_2 = f1_score(y_test_2, y_pred_2, average='weighted')
    
    print(f"Phase 2 Metrics - Accuracy: {acc_2:.4f}, F1: {f1_2:.4f}")
    print(f"Best params: {search_2.best_params_}")
    
    joblib.dump(best_pipe_2, models_dir / "model_phase2_v1.joblib")
    print("Model saved to models/model_phase2_v1.joblib")
    
    # Phase 2 includes both pre-flight and post-flight features
    all_phase2_features = PRE_FLIGHT_FEATURES + POST_FLIGHT_FEATURES
    
    save_metadata(
        models_dir / "metadata_phase2_v1.json",
        "Phase 2 - Pre and Post-flight (Random Forest)",
        all_phase2_features,
        acc_2,
        f1_2,
        best_pipe_2.named_steps['clf'].get_params()
    )
    
    print("\nDone! Stage 2 completed.")

if __name__ == "__main__":
    main()
