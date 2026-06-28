import pandas as pd
from pathlib import Path
import urllib.request
from sklearn.model_selection import train_test_split

BINARY_MAP = {
    'Gender':         {'Female': 1, 'Male': 0},
    'Customer Type':  {'Loyal Customer': 1, 'disloyal Customer': 0},
    'Type of Travel': {'Business travel': 1, 'Personal Travel': 0},
}

PRE_FLIGHT_FEATURES = [
    'Age', 'Flight Distance',
    'Gender', 'Customer Type', 'Type of Travel', 'Class'
]

ORDINAL_FEATURES = [
    'Seat comfort', 'Departure/Arrival time convenient',
    'Food and drink', 'Gate location', 'Inflight wifi service',
    'Inflight entertainment', 'Online support', 'Ease of Online booking',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Cleanliness', 'Online boarding',
]

POST_FLIGHT_FEATURES = ['Arrival Delay in Minutes'] + ORDINAL_FEATURES

def load_raw_data():
    """Loads the airline dataset from local cache or downloads it."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    csv_path = base_dir / "data" / "raw" / "Invistico_Airline.csv"

    if not csv_path.is_file():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/AgustinaHourcade/airline-passenger-satisfaction/blob/main/data/raw/Invistico_Airline.csv?raw=true"
        urllib.request.urlretrieve(url, csv_path)

    return pd.read_csv(csv_path)

def preprocess_phase1(df):
    """Prepares data for Phase 1 (Pre-flight features only)."""
    df_model = df.copy()
    
    # Create target
    df_model['Satisfaction_bin'] = (df_model['satisfaction'] == 'satisfied').astype(int)
    
    # Map binary features
    for col, mapping in BINARY_MAP.items():
        df_model[col] = df_model[col].map(mapping)
        
    X = df_model[PRE_FLIGHT_FEATURES].copy()
    y = df_model['Satisfaction_bin']
    
    return train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

def preprocess_phase2(df):
    """Prepares data for Phase 2 (Pre + Post flight features)."""
    df_model = df.copy()
    
    # Create target
    df_model['Satisfaction_bin'] = (df_model['satisfaction'] == 'satisfied').astype(int)
    
    # Split first to prevent data leakage for imputation
    df_train, df_test = train_test_split(
        df_model,
        test_size=0.20,
        random_state=42,
        stratify=df_model['Satisfaction_bin']
    )
    
    # Select features
    features = PRE_FLIGHT_FEATURES + POST_FLIGHT_FEATURES
    
    X_train = df_train[features].copy()    # type: ignore
    y_train = df_train['Satisfaction_bin'] # type: ignore
    X_test = df_test[features].copy()      # type: ignore
    y_test = df_test['Satisfaction_bin']   # type: ignore
    
    # Map binary features
    for col, mapping in BINARY_MAP.items():
        X_train[col] = X_train[col].replace(mapping) # type: ignore
        X_test[col] = X_test[col].replace(mapping)   # type: ignore
        
    return X_train, X_test, y_train, y_test