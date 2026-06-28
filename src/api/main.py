import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
import json
import sys
import os

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

@app.get("/root")
def root():
    return {"message": "Airline Passenger Satisfaction API is running. Go to /docs for Swagger UI."}

@app.get("/health")
def health_check():
    status = "healthy" if "phase1" in models and "phase2" in models else "degraded"
    return {"status": status}

@app.get("/api/metrics")
def get_metrics():
    base_dir = Path(__file__).resolve().parent.parent.parent
    models_dir = base_dir / "models"
    try:
        with open(models_dir / "metadata_phase1_v1.json") as f1, open(models_dir / "metadata_phase2_v1.json") as f2:
            return {"phase1": json.load(f1), "phase2": json.load(f2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading metrics: {str(e)}")

@app.post("/predict/phase1", response_model=PredictionResponse)
def predict_phase1(features: Phase1Features):
    if "phase1" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    # Convert Pydantic model to DataFrame 
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
        
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
        
        # Get feature names after transformation
        raw_names = preprocessor.get_feature_names_out()
        
        # Clean prefix and map to Spanish
        spanish_map = {
            "Age": "Edad", "Flight Distance": "Distancia", "Inflight wifi service": "WiFi a bordo",
            "Seat comfort": "Comodidad asiento", "Inflight entertainment": "Entretenimiento",
            "Online boarding": "Check-in Online", "Food and drink": "Comida y bebida",
            "Arrival Delay in Minutes": "Retraso Llegada", "Customer Type_disloyal Customer": "Cliente Desleal",
            "Customer Type_Loyal Customer": "Cliente Leal", "Class_Business": "Clase Business",
            "Class_Eco": "Clase Eco", "Class_Eco Plus": "Clase Eco Plus",
            "Type of Travel_Business travel": "Viaje Negocios", "Type of Travel_Personal Travel": "Viaje Personal",
            "Gate location": "Puerta embarque", "Departure/Arrival time convenient": "Horario conveniente",
            "Online support": "Soporte Online", "Ease of Online booking": "Facilidad Reserva",
            "On-board service": "Servicio a bordo", "Leg room service": "Espacio piernas",
            "Baggage handling": "Manejo equipaje", "Checkin service": "Servicio Check-in",
            "Cleanliness": "Limpieza", "Gender_Female": "Mujer", "Gender_Male": "Hombre"
        }
        
        friendly_names = []
        for name in raw_names:
            clean_name = name.split('__', 1)[-1]
            friendly_names.append(spanish_map.get(clean_name, clean_name))
        
        # Extract the shap values for the predicted class 
        pred_class = int(clf.predict(X_transformed)[0])
        import numpy as np
        if isinstance(shap_values, list):
            shap_vals_class = shap_values[pred_class][0]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_vals_class = shap_values[0, :, pred_class]
        else:
            shap_vals_class = shap_values[0]
            
        # Pair features with their SHAP values
        explanation = [
            {"feature": name, "value": float(val)} 
            for name, val in zip(friendly_names, shap_vals_class)
        ]
        
        # Sort by absolute impact
        explanation.sort(key=lambda x: abs(x["value"]), reverse=True)
        
        return {
            "prediction": pred_class,
            "explanation": explanation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explanation error: {str(e)}")


# Get absolute path to src/frontend
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Mount frontend static files to serve the UI directly from the API
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
