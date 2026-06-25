from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_phase1_pipeline(model_type='rf'):
    """
    Creates the sklearn Pipeline for Phase 1.
    model_type: 'lr' (Logistic Regression), 'dt' (Decision Tree), 'rf' (Random Forest)
    """
    num_features = ['Age', 'Flight Distance']
    ohe_features = ['Class']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_scaler', StandardScaler(), num_features),
            ('ohe_class', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ohe_features),
        ],
        remainder='passthrough'
    )
    
    if model_type == 'lr':
        clf = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs')
    elif model_type == 'dt':
        clf = DecisionTreeClassifier(random_state=42)
    elif model_type == 'rf':
        clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    else:
        raise ValueError("model_type must be 'lr', 'dt', or 'rf'")
        
    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])

def get_phase2_pipeline(model_type='rf'):
    """
    Creates the sklearn Pipeline for Phase 2.
    model_type: 'lr' (Logistic Regression), 'dt' (Decision Tree), 'rf' (Random Forest)
    """
    # Numeric and Ordinal features that need scaling
    scaled_features = [
        'Age', 'Flight Distance', 'Arrival Delay in Minutes',
        'Seat comfort', 'Departure/Arrival time convenient',
        'Food and drink', 'Gate location', 'Inflight wifi service',
        'Inflight entertainment', 'Online support', 'Ease of Online booking',
        'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Cleanliness', 'Online boarding'
    ]
    
    ohe_features = ['Class']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_scaler', StandardScaler(), scaled_features),
            ('ohe_class', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), ohe_features),
        ],
        remainder='passthrough'
    )
    
    if model_type == 'lr':
        clf = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs')
    elif model_type == 'dt':
        clf = DecisionTreeClassifier(random_state=42)
    elif model_type == 'rf':
        clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    else:
        raise ValueError("model_type must be 'lr', 'dt', or 'rf'")
        
    return Pipeline([
        ('preprocessor', preprocessor),
        ('clf', clf)
    ])
