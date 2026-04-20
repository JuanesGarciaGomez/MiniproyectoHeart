# Heart Disease MLOps – Proyecto Integrador

Predicción de falla cardíaca con control de calidad y monitoreo integrado, aplicando prácticas de MLOps en entorno local.

---

## Estructura del Proyecto

```
heart-disease-mlops/
├── app/
│   └── api.py                  # ETAPA 3 – API REST con FastAPI
├── docker/
│   ├── Dockerfile              # ETAPA 3 – Contenedorización
│   └── requirements.txt
├── k8s/
│   ├── deployment.yaml         # ETAPA 4 – Orquestación Kubernetes
│   └── service.yaml
├── notebooks/
│   └── 3_drift_monitoring.py   # ETAPA 6 – Monitoreo con Evidently
├── tests/
│   └── test_api.py             # ETAPA 5 – Tests automáticos
├── .github/
│   └── workflows/
│       └── ci.yml              # ETAPA 5 – GitHub Actions CI/CD
├── model.joblib                # Modelo entrenado (GradientBoosting)
├── heartminiproyecto.csv       # Dataset original
└── drift_report.html           # Reporte de drift generado
```

---

## Etapas

###  ETAPA 0 – Estructura de carpetas
La estructura modular mostrada arriba.

###  ETAPA 1 – Análisis Exploratorio & Preprocesamiento
Ver `Proyecto_Heart_Disease_Completo.ipynb`.

###  ETAPA 2 – Entrenamiento Seguro
Pipeline con `GridSearchCV` + validación cruzada estratificada. Mejor modelo: **GradientBoostingClassifier**.

---

###  ETAPA 3 – Despliegue Local (FastAPI + Docker)

**Ejecutar la API directamente:**
```bash
cd heart-disease-mlops
pip install fastapi uvicorn scikit-learn pandas joblib pydantic
uvicorn app.api:app --reload --port 8000
```

Accede a la documentación interactiva en: http://localhost:8000/docs

**Ejemplo de predicción:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 54, "Sex": "M", "ChestPainType": "ATA",
    "RestingBP": 130, "Cholesterol": 236, "FastingBS": 0,
    "RestingECG": "Normal", "MaxHR": 168,
    "ExerciseAngina": "N", "Oldpeak": 0.0, "ST_Slope": "Up"
  }'
```

**Construir y correr con Docker:**
```bash
# Construir imagen
docker build -f docker/Dockerfile -t heart-disease-api:latest .

# Correr contenedor
docker run -d -p 8000:8000 --name heart-api heart-disease-api:latest

# Verificar
curl http://localhost:8000/health
```

---

###  ETAPA 4 – Orquestación (Kubernetes)

Requiere [minikube](https://minikube.sigs.k8s.io/) o [kind](https://kind.sigs.k8s.io/).

```bash
# Iniciar minikube
minikube start

# Cargar imagen local en minikube
minikube image load heart-disease-api:latest

# Aplicar manifiestos
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Verificar pods y servicio
kubectl get pods
kubectl get svc

# Obtener URL de acceso
minikube service heart-disease-service --url
```

El servicio queda expuesto en `<minikube-ip>:30080`.

---

###  ETAPA 5 – Integración Continua (GitHub Actions)

El workflow `.github/workflows/ci.yml` se ejecuta automáticamente en cada push a `main`:

1. **Lint** con `flake8` (estilo de código)
2. **Tests** con `pytest` (17 pruebas automáticas)
3. **Build** de la imagen Docker
4. **Smoke test** del contenedor

**Correr tests localmente:**
```bash
pip install pytest httpx
pytest tests/ -v
```

---

###  ETAPA 6 – Monitoreo con Evidently

```bash
pip install evidently
python notebooks/3_drift_monitoring.py
```

Abre `drift_report.html` en el navegador para ver el reporte completo de deriva de datos.

El script detecta drift comparando datos de entrenamiento (referencia) vs. datos de producción (nuevas predicciones).

---

## Dataset

[Heart Failure Prediction – Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
918 registros | 11 features | Variable objetivo: `HeartDisease` (0/1)

## Tecnologías

| Componente | Tecnología |
|---|---|
| Modelo | scikit-learn GradientBoostingClassifier |
| API | FastAPI + Uvicorn |
| Contenedor | Docker |
| Orquestación | Kubernetes (minikube) |
| CI/CD | GitHub Actions |
| Monitoreo | Evidently AI |
