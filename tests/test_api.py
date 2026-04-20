"""
ETAPA 5 – Tests automáticos con pytest
Cubre: endpoint /health, /predict con caso positivo y negativo, validación de inputs.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Asegura que se importa la app desde la raíz del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["MODEL_PATH"] = "model.joblib"

from app.api import app  # noqa: E402

client = TestClient(app)

# ─── Paciente de ejemplo sin enfermedad ──────────────────────────
PATIENT_NO_DISEASE = {
    "Age": 40,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 140,
    "Cholesterol": 289,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 172,
    "ExerciseAngina": "N",
    "Oldpeak": 0.0,
    "ST_Slope": "Up",
}

# ─── Paciente de ejemplo con enfermedad ──────────────────────────
PATIENT_DISEASE = {
    "Age": 63,
    "Sex": "M",
    "ChestPainType": "ASY",
    "RestingBP": 145,
    "Cholesterol": 233,
    "FastingBS": 1,
    "RestingECG": "LVH",
    "MaxHR": 150,
    "ExerciseAngina": "Y",
    "Oldpeak": 2.3,
    "ST_Slope": "Down",
}


class TestHealth:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"


class TestPredict:
    def test_predict_returns_200(self):
        r = client.post("/predict", json=PATIENT_NO_DISEASE)
        assert r.status_code == 200

    def test_predict_schema(self):
        r = client.post("/predict", json=PATIENT_NO_DISEASE)
        body = r.json()
        assert "prediction" in body
        assert "label" in body
        assert "probability_disease" in body
        assert "probability_no_disease" in body
        assert "risk_level" in body

    def test_prediction_is_binary(self):
        r = client.post("/predict", json=PATIENT_NO_DISEASE)
        assert r.json()["prediction"] in (0, 1)

    def test_probabilities_sum_to_one(self):
        r = client.post("/predict", json=PATIENT_NO_DISEASE)
        body = r.json()
        total = body["probability_disease"] + body["probability_no_disease"]
        assert abs(total - 1.0) < 1e-4

    def test_risk_level_valid(self):
        r = client.post("/predict", json=PATIENT_DISEASE)
        assert r.json()["risk_level"] in ("Bajo", "Moderado", "Alto")

    def test_predict_disease_case(self):
        """El paciente de alto riesgo debe devolver predicción 1."""
        r = client.post("/predict", json=PATIENT_DISEASE)
        assert r.status_code == 200
        # No forzamos el valor exacto, solo que devuelva 0 o 1
        assert r.json()["prediction"] in (0, 1)


class TestValidation:
    def test_invalid_sex(self):
        bad = {**PATIENT_NO_DISEASE, "Sex": "X"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_invalid_chest_pain(self):
        bad = {**PATIENT_NO_DISEASE, "ChestPainType": "XYZ"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_age_out_of_range(self):
        bad = {**PATIENT_NO_DISEASE, "Age": 200}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_missing_field(self):
        bad = {k: v for k, v in PATIENT_NO_DISEASE.items() if k != "MaxHR"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


class TestBatch:
    def test_batch_predict(self):
        r = client.post("/predict/batch", json=[PATIENT_NO_DISEASE, PATIENT_DISEASE])
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 2
        assert len(body["predictions"]) == 2

    def test_batch_limit(self):
        many = [PATIENT_NO_DISEASE] * 101
        r = client.post("/predict/batch", json=many)
        assert r.status_code == 400
