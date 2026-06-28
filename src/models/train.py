import json
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.preprocessing import PRE_FLIGHT_FEATURES, POST_FLIGHT_FEATURES
from src.models.pipeline import get_phase1_pipeline, get_phase2_pipeline

def save_metadata(path, model_name, features, acc, f1, prec, rec, auc, hyperparams, cm=None, roc=None):
    metadata = {
        "model_name": model_name,
        "training_date": datetime.now().isoformat(),
        "features": features,
        "hyperparameters": hyperparams,
        "validation_metrics": {
            "accuracy": round(acc, 4),
            "f1_score_weighted": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "roc_curve": roc
        }
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {path}")

def main():
    print("Loading processed data...")
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    processed_dir = base_dir / "data" / "processed"
    p1_dir = processed_dir / "phase1"
    p2_dir = processed_dir / "phase2"
    
    # Check if processed data exists
    if not p1_dir.exists() or not p2_dir.exists():
        print("Processed data not found. Please run 'python -m src.data.make_dataset' first.")
        return

    # Define models directory
    models_dir = base_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # --- Phase 1 ---
    print("\n--- Training Phase 1 ---")
    X_train_1 = pd.read_csv(p1_dir / "X_train.csv")
    X_test_1 = pd.read_csv(p1_dir / "X_test.csv")
    y_train_1 = pd.read_csv(p1_dir / "y_train.csv").squeeze()
    y_test_1 = pd.read_csv(p1_dir / "y_test.csv").squeeze()
    
    pipe_1 = get_phase1_pipeline('rf')
    
    # Try to load best hyperparameters, otherwise use fallback
    best_params_file = base_dir / "notebooks" / "models" / "best_hyperparameters.json"
    rf_params_1 = {
        'clf__n_estimators': 100,
        'clf__max_depth': 15,
        'clf__min_samples_split': 2,
        'clf__min_samples_leaf': 1
    }
    
    if best_params_file.exists():
        with open(best_params_file, "r", encoding="utf-8") as f:
            best_params = json.load(f)
            if 'phase1' in best_params and 'Random Forest' in best_params['phase1']:
                rf_params_1 = best_params['phase1']['Random Forest']
                print("Loaded Phase 1 parameters from best_hyperparameters.json")
    
    pipe_1.set_params(**rf_params_1)
    
    print("Fitting final model (Phase 1)...")
    pipe_1.fit(X_train_1, y_train_1)
    
    best_pipe_1 = pipe_1
    y_pred_1 = best_pipe_1.predict(X_test_1)
    y_prob_1 = best_pipe_1.predict_proba(X_test_1)[:, 1]
    
    acc_1 = accuracy_score(y_test_1, y_pred_1)
    f1_1 = f1_score(y_test_1, y_pred_1, average='weighted')
    prec_1 = precision_score(y_test_1, y_pred_1)
    rec_1 = recall_score(y_test_1, y_pred_1)
    auc_1 = roc_auc_score(y_test_1, y_prob_1)
    
    cm_1 = confusion_matrix(y_test_1, y_pred_1).tolist()
    fpr_1, tpr_1, _ = roc_curve(y_test_1, y_prob_1)
    step_1 = max(1, len(fpr_1) // 50)
    roc_1 = {"fpr": fpr_1[::step_1].tolist(), "tpr": tpr_1[::step_1].tolist()}
    
    print(f"Phase 1 Metrics - Accuracy: {acc_1:.4f}, F1: {f1_1:.4f}")
    print(f"Used params: {rf_params_1}")
    
    joblib.dump(best_pipe_1, models_dir / "model_phase1_v1.joblib")
    print("Model saved to models/model_phase1_v1.joblib")
    
    save_metadata(
        models_dir / "metadata_phase1_v1.json",
        "Phase 1 - Pre-flight (Random Forest)",
        PRE_FLIGHT_FEATURES,
        acc_1,
        f1_1,
        prec_1,
        rec_1,
        auc_1,
        best_pipe_1.named_steps['clf'].get_params(),
        cm_1,
        roc_1
    )
    
    # --- Phase 2 ---
    print("\n--- Training Phase 2 ---")
    X_train_2 = pd.read_csv(p2_dir / "X_train.csv")
    X_test_2 = pd.read_csv(p2_dir / "X_test.csv")
    y_train_2 = pd.read_csv(p2_dir / "y_train.csv").squeeze()
    y_test_2 = pd.read_csv(p2_dir / "y_test.csv").squeeze()
    
    pipe_2 = get_phase2_pipeline('rf')
    
    # Try to load best hyperparameters, otherwise use fallback
    rf_params_2 = {
        'clf__n_estimators': 100,
        'clf__max_depth': None,
        'clf__min_samples_split': 2,
        'clf__min_samples_leaf': 1
    }
    
    if best_params_file.exists():
        with open(best_params_file, "r", encoding="utf-8") as f:
            best_params = json.load(f)
            if 'phase2' in best_params and 'Random Forest' in best_params['phase2']:
                rf_params_2 = best_params['phase2']['Random Forest']
                print("Loaded Phase 2 parameters from best_hyperparameters.json")
    
    pipe_2.set_params(**rf_params_2)
    
    print("Fitting final model (Phase 2)...")
    pipe_2.fit(X_train_2, y_train_2)
    
    best_pipe_2 = pipe_2
    y_pred_2 = best_pipe_2.predict(X_test_2)
    y_prob_2 = best_pipe_2.predict_proba(X_test_2)[:, 1]
    
    acc_2 = accuracy_score(y_test_2, y_pred_2)
    f1_2 = f1_score(y_test_2, y_pred_2, average='weighted')
    prec_2 = precision_score(y_test_2, y_pred_2)
    rec_2 = recall_score(y_test_2, y_pred_2)
    auc_2 = roc_auc_score(y_test_2, y_prob_2)
    
    cm_2 = confusion_matrix(y_test_2, y_pred_2).tolist()
    fpr_2, tpr_2, _ = roc_curve(y_test_2, y_prob_2)
    step_2 = max(1, len(fpr_2) // 50)
    roc_2 = {"fpr": fpr_2[::step_2].tolist(), "tpr": tpr_2[::step_2].tolist()}
    
    print(f"Phase 2 Metrics - Accuracy: {acc_2:.4f}, F1: {f1_2:.4f}")
    print(f"Used params: {rf_params_2}")
    
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
        prec_2,
        rec_2,
        auc_2,
        best_pipe_2.named_steps['clf'].get_params(),
        cm_2,
        roc_2
    )
    
    print("\nDone! Stage 2 completed.")

if __name__ == "__main__":
    main()
