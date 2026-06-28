import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.preprocessing import load_raw_data, preprocess_phase1, preprocess_phase2

def main():
    print("Loading raw data...")
    raw_path = Path("data") / "raw" / "Invistico_Airline.csv"
    df = load_raw_data(raw_path)
    
    # Drop duplicates to prevent inflation of metrics
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    if df.shape[0] < initial_shape:
        print(f"Dropped {initial_shape - df.shape[0]} duplicate rows.")
    
    processed_dir = Path("data") / "processed"
    p1_dir = processed_dir / "phase1"
    p2_dir = processed_dir / "phase2"
    
    p1_dir.mkdir(parents=True, exist_ok=True)
    p2_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 1
    print("Processing Phase 1 data...")
    X_train_1, X_test_1, y_train_1, y_test_1 = preprocess_phase1(df)
    
    X_train_1.to_csv(p1_dir / "X_train.csv", index=False)
    X_test_1.to_csv(p1_dir / "X_test.csv", index=False)
    y_train_1.to_csv(p1_dir / "y_train.csv", index=False)
    y_test_1.to_csv(p1_dir / "y_test.csv", index=False)
    print(f"Phase 1 data saved to {p1_dir}")
    
    # Phase 2
    print("Processing Phase 2 data...")
    X_train_2, X_test_2, y_train_2, y_test_2, median_arrival = preprocess_phase2(df)
    
    X_train_2.to_csv(p2_dir / "X_train.csv", index=False)
    X_test_2.to_csv(p2_dir / "X_test.csv", index=False)
    y_train_2.to_csv(p2_dir / "y_train.csv", index=False)
    y_test_2.to_csv(p2_dir / "y_test.csv", index=False)
    
    # Save median for inference later alongside the model metadata
    imputation_params = {
        "median_arrival_delay": median_arrival
    }
    with open(Path("models") / "imputation_params.json", "w") as f:
        import json
        json.dump(imputation_params, f, indent=4)
        
    print(f"Phase 2 data saved to {p2_dir}")
    print("Data processing complete.")

if __name__ == "__main__":
    main()
