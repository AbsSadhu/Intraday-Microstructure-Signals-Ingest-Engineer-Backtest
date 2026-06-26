import os
import json
import joblib
import pandas as pd
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Quant Research API",
    description="Backend API for quantitative trading model inference and metrics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
REPORT_PATH = os.path.join(ARTIFACTS_DIR, "report.json")

# In-memory storage for lazy loading
model_cache = None

def get_model():
    global model_cache
    if model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=404, detail="Model not found. Please run training pipeline first.")
        try:
            model_cache = joblib.load(MODEL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    return model_cache

class FeaturesRequest(BaseModel):
    features: Dict[str, float]

class InferenceResponse(BaseModel):
    prediction: float

@app.get("/")
def health_check() -> Dict[str, str]:
    return {"status": "ok", "message": "Quant API is running"}

@app.get("/api/metrics")
def get_metrics() -> Dict[str, Any]:
    """Retrieve the latest walk-forward training report."""
    if not os.path.exists(REPORT_PATH):
        raise HTTPException(status_code=404, detail="Report not found. Please run training pipeline first.")
    with open(REPORT_PATH, "r") as f:
        return json.load(f)

@app.post("/api/predict", response_model=InferenceResponse)
def predict(request: FeaturesRequest) -> InferenceResponse:
    """Run inference on the provided feature vector."""
    model = get_model()
    # Convert incoming dict to DataFrame as the model expects pandas DataFrame
    df_features = pd.DataFrame([request.features])
    
    try:
        prediction = float(model.predict(df_features)[0])
        return InferenceResponse(prediction=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

# Placeholder for real-time WebSocket ingestion endpoint
# @app.websocket("/ws/stream")

if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Ensure project root is in python path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
        
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
