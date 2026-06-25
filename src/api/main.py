import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
import json
import sys
import os

# Ensure the root directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.api.schemas import Phase1Features, Phase2Features, PredictionResponse

# Global variables to hold models and config
models = {}
config = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    base_dir = Path(__file__).resolve().parent.parent.parent
    models_dir = base_dir / "models"
    
    try:
        models["phase1"] = joblib.load(models_dir / "model_phase1_v1.joblib")
        models["phase2"] = joblib.load(models_dir / "model_phase2_v1.joblib")
        
        with open(models_dir / "imputation_params.json", "r") as f:
            config["imputation"] = json.load(f)
            
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        
    yield
    # Clean up on shutdown
    models.clear()
    config.clear()

app = FastAPI(
    title="Airline Passenger Satisfaction API",
    description="API for predicting airline passenger satisfaction based on pre-flight and post-flight features.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration to allow the frontend to interact with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In a real scenario, this should be restricted
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Airline Passenger Satisfaction API"}

@app.get("/health")
def health_check():
    status = "healthy" if "phase1" in models and "phase2" in models else "degraded"
    return {"status": status}

@app.post("/predict/phase1", response_model=PredictionResponse)
def predict_phase1(features: Phase1Features):
    if "phase1" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    # Convert Pydantic model to DataFrame (respecting aliases used during training)
    data = features.model_dump(by_alias=True)
    df = pd.DataFrame([data])
    
    try:
        # Preprocessing mappings applied just like during training
        from src.data.preprocessing import BINARY_MAP
        for col, mapping in BINARY_MAP.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                
        pipeline = models["phase1"]
        pred = pipeline.predict(df)[0]
        prob = pipeline.predict_proba(df)[0][1]
        
        return PredictionResponse(
            prediction=int(pred),
            probability=float(prob),
            phase=1
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict/phase2", response_model=PredictionResponse)
def predict_phase2(features: Phase2Features):
    if "phase2" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    data = features.model_dump(by_alias=True)
    
    # Impute Arrival Delay if missing
    if data.get("Arrival Delay in Minutes") is None:
        data["Arrival Delay in Minutes"] = config["imputation"]["median_arrival_delay"]
        
    df = pd.DataFrame([data])
    
    try:
        from src.data.preprocessing import BINARY_MAP
        for col, mapping in BINARY_MAP.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                
        pipeline = models["phase2"]
        pred = pipeline.predict(df)[0]
        prob = pipeline.predict_proba(df)[0][1]
        
        return PredictionResponse(
            prediction=int(pred),
            probability=float(prob),
            phase=2
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/explain")
def explain_prediction(features: Phase2Features):
    if "phase2" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        import shap
        data = features.model_dump(by_alias=True)
        if data.get("Arrival Delay in Minutes") is None:
            data["Arrival Delay in Minutes"] = config["imputation"]["median_arrival_delay"]
            
        df = pd.DataFrame([data])
        
        from src.data.preprocessing import BINARY_MAP
        for col, mapping in BINARY_MAP.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
        
        pipeline = models["phase2"]
        # SHAP requires the preprocessed data and the underlying model separately
        preprocessor = pipeline.named_steps['preprocessor']
        clf = pipeline.named_steps['clf']
        
        X_transformed = preprocessor.transform(df)
        
        # Initialize JavaScript for SHAP visualizations if needed
        # We'll return the raw SHAP values so the frontend can plot them
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
        
        # Get feature names after transformation
        feature_names = preprocessor.get_feature_names_out()
        
        # Extract the shap values for the predicted class (assuming binary classification)
        # Random Forest shap_values is a list of arrays (one for each class)
        pred_class = int(clf.predict(X_transformed)[0])
        shap_vals_class = shap_values[pred_class][0] # first instance
        
        # Pair features with their SHAP values
        explanation = [
            {"feature": name, "value": float(val)} 
            for name, val in zip(feature_names, shap_vals_class)
        ]
        
        # Sort by absolute impact
        explanation.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        return {
            "prediction": pred_class,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explanation error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
