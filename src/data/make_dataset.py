import os
import sys
from pathlib import Path
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.preprocessing import load_raw_data, preprocess_phase1, preprocess_phase2

def main():
    print("Loading raw data...")
    df = load_raw_data()
    
    # Drop duplicates to prevent inflation of metrics
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    if df.shape[0] < initial_shape:
        print(f"Dropped {initial_shape - df.shape[0]} duplicate rows.")
    
    base_dir = Path(__file__).resolve().parent.parent.parent
    processed_dir = base_dir / "data" / "processed"
    p1_dir = processed_dir / "phase1"
    p2_dir = processed_dir / "phase2"
    
    p1_dir.mkdir(parents=True, exist_ok=True)
    p2_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 1
    print("Processing Phase 1 data...")
    X_train_1, X_test_1, y_train_1, y_test_1 = preprocess_phase1(df)
    
    pd.DataFrame(X_train_1).to_csv(p1_dir / "X_train.csv", index=False)
    pd.DataFrame(X_test_1).to_csv(p1_dir / "X_test.csv", index=False)
    pd.DataFrame(y_train_1).to_csv(p1_dir / "y_train.csv", index=False)
    pd.DataFrame(y_test_1).to_csv(p1_dir / "y_test.csv", index=False)
    print(f"Phase 1 data saved to {p1_dir}")
    
    # Phase 2
    print("Processing Phase 2 data...")
    X_train_2, X_test_2, y_train_2, y_test_2 = preprocess_phase2(df)
    
    pd.DataFrame(X_train_2).to_csv(p2_dir / "X_train.csv", index=False)
    pd.DataFrame(X_test_2).to_csv(p2_dir / "X_test.csv", index=False)
    pd.DataFrame(y_train_2).to_csv(p2_dir / "y_train.csv", index=False)
    pd.DataFrame(y_test_2).to_csv(p2_dir / "y_test.csv", index=False)
    
    print(f"Phase 2 data saved to {p2_dir}")
    print("Data processing complete.")

if __name__ == "__main__":
    main()
