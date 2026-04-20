"""
ETAPA 3 – API REST con FastAPI
Sirve el modelo de predicción de enfermedad cardíaca.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import os

# ─── Cargar modelo ────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")

try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Modelo cargado desde: {MODEL_PATH}")
    
except FileNotFoundError:
    raise RuntimeError(f"❌ No se encontró el modelo en: {MODEL_PATH}")

# ─── Esquema de entrada ───────────────────────────────────────────
class PatientFeatures(BaseModel):
    Age: int = Field(..., ge=1, le=120, example=54, description="Edad del paciente")
    Sex: str = Field(..., example="M", description="Sexo: M o F")
    ChestPainType: str = Field(..., example="ATA", description="Tipo de dolor: ATA, NAP, ASY, TA")
    RestingBP: int = Field(..., ge=0, le=300, example=130, description="Presión arterial en reposo (mm Hg)")
    Cholesterol: int = Field(..., ge=0, le=700, example=236, description="Colesterol sérico (mg/dl)")
    FastingBS: int = Field(..., ge=0, le=1, example=0, description="Glucosa en ayunas >120 mg/dl (1=sí, 0=no)")
    RestingECG: str = Field(..., example="Normal", description="ECG en reposo: Normal, ST, LVH")
    MaxHR: int = Field(..., ge=40, le=250, example=168, description="Frecuencia cardíaca máxima")
    ExerciseAngina: str = Field(..., example="N", description="Angina por ejercicio: Y o N")
    Oldpeak: float = Field(..., ge=-5.0, le=10.0, example=0.0, description="Depresión del segmento ST")
    ST_Slope: str = Field(..., example="Up", description="Pendiente ST: Up, Flat, Down")

    @validator("Sex")
    
    def sex_valid(cls, v):
        if v not in ("M", "F"):
            raise ValueError("Sex debe ser 'M' o 'F'")
        return v

    @validator("ChestPainType")
    
    def chest_valid(cls, v):
        if v not in ("ATA", "NAP", "ASY", "TA"):
            raise ValueError("ChestPainType debe ser ATA, NAP, ASY o TA")
        return v

    @validator("RestingECG")
    
    def ecg_valid(cls, v):
        if v not in ("Normal", "ST", "LVH"):
            raise ValueError("RestingECG debe ser Normal, ST o LVH")
        return v

    @validator("ExerciseAngina")
    
    def angina_valid(cls, v):
        if v not in ("Y", "N"):
            raise ValueError("ExerciseAngina debe ser 'Y' o 'N'")
        return v

    @validator("ST_Slope")
    
    def slope_valid(cls, v):
        if v not in ("Up", "Flat", "Down"):
            raise ValueError("ST_Slope debe ser Up, Flat o Down")
        return v


# ─── Esquema de respuesta ─────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_disease: float
    probability_no_disease: float
    risk_level: str


# ─── App FastAPI ──────────────────────────────────────────────────
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API REST para predicción de enfermedad cardíaca usando GradientBoostingClassifier.",
    version="1.0.0",
)

CAT_FEATURES = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUM_FEATURES = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]


@app.get("/", tags=["Health"])

def root():
    return {"status": "ok", "message": "Heart Disease API activa. Ve a /docs para la documentación."}


@app.get("/health", tags=["Health"])

def health():
    return {"status": "healthy", "model": MODEL_PATH}


@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])

def predict(patient: PatientFeatures):
    """
    Recibe las características del paciente y retorna:
    - **prediction**: 0 (sin enfermedad) o 1 (con enfermedad)
    - **label**: etiqueta legible
    - **probability_disease**: probabilidad de enfermedad cardíaca
    - **probability_no_disease**: probabilidad de no tener enfermedad
    - **risk_level**: Bajo / Moderado / Alto
    """
    try:
        data = pd.DataFrame([patient.dict()])[CAT_FEATURES + NUM_FEATURES]
        pred = int(model.predict(data)[0])
        proba = model.predict_proba(data)[0]
        p_disease = float(proba[1])
        p_no_disease = float(proba[0])

        if p_disease < 0.35:
            risk = "Bajo"
        elif p_disease < 0.65:
            risk = "Moderado"
        else:
            risk = "Alto"

        return PredictionResponse(
            prediction=pred,
            label="Con enfermedad cardíaca" if pred == 1 else "Sin enfermedad cardíaca",
            probability_disease=round(p_disease, 4),
            probability_no_disease=round(p_no_disease, 4),
            risk_level=risk,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Predicción"])

def predict_batch(patients: list[PatientFeatures]):
    """Predicción en lote (múltiples pacientes a la vez)."""
    if len(patients) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 pacientes por lote.")
    results = []
    for p in patients:
        data = pd.DataFrame([p.dict()])[CAT_FEATURES + NUM_FEATURES]
        pred = int(model.predict(data)[0])
        proba = model.predict_proba(data)[0]
        results.append({
            "prediction": pred,
            "label": "Con enfermedad" if pred == 1 else "Sin enfermedad",
            "probability_disease": round(float(proba[1]), 4),
        })
    return {"total": len(results), "predictions": results}
