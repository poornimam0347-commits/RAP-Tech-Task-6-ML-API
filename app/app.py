from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="RAP Tech ML Inference API",
    description="REST API for real-time machine learning predictions.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "champion_model.joblib"

model = None

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=1,
        description="Numeric feature values in the model's expected order.",
    )


class PredictionResponse(BaseModel):
    prediction: int | float | str
    prediction_probabilities: dict[str, float] | None
    model_status: str


@app.get("/")
def home():
    return {
        "message": "RAP Tech ML Inference API is running",
        "docs": "/docs",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Add your trained model to models/champion_model.joblib.",
        )

    features = np.asarray(request.features, dtype=float).reshape(1, -1)

    if not np.isfinite(features).all():
        raise HTTPException(
            status_code=422,
            detail="Features must contain only finite numeric values.",
        )

    try:
        prediction = model.predict(features)[0]

        probabilities = None

        if hasattr(model, "predict_proba"):
            raw_probabilities = model.predict_proba(features)[0]
            classes = getattr(model, "classes_", range(len(raw_probabilities)))

            probabilities = {
                str(label): float(probability)
                for label, probability in zip(classes, raw_probabilities)
            }

        if isinstance(prediction, np.generic):
            prediction = prediction.item()

        return PredictionResponse(
            prediction=prediction,
            prediction_probabilities=probabilities,
            model_status="loaded",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {exc}",
        ) from exc

