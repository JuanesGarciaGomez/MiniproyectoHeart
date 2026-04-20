# %% [markdown]
# # ETAPA 6 – Monitoreo con Evidently
# **Detección de Data Drift entre datos de entrenamiento y predicción**
#
# Pasos:
# 1. Cargar datos de referencia (entrenamiento) y datos de producción (nuevos)
# 2. Generar reporte de deriva con Evidently
# 3. Exportar `drift_report.html` para revisión visual

# %% Importaciones
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# %% Cargar datos de referencia (train)
df = pd.read_csv("../heartminiproyecto.csv", sep=";")

# Imputar Colesterol = 0 (igual que en Etapa 1)
mediana_0 = df[df["HeartDisease"] == 0]["Cholesterol"].replace(0, np.nan).median()
mediana_1 = df[df["HeartDisease"] == 1]["Cholesterol"].replace(0, np.nan).median()
df.loc[(df["Cholesterol"] == 0) & (df["HeartDisease"] == 0), "Cholesterol"] = mediana_0
df.loc[(df["Cholesterol"] == 0) & (df["HeartDisease"] == 1), "Cholesterol"] = mediana_1

from sklearn.model_selection import train_test_split
X = df.drop(columns=["HeartDisease"])
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

reference_data = X_train.copy()
reference_data["HeartDisease"] = y_train.values

# %% Simular datos de producción (con leve drift artificial)
# En un proyecto real, esto sería el log de predicciones en producción.
production_data = X_test.copy()
production_data["HeartDisease"] = y_test.values

# Simulamos drift: envejecimiento de población y mayor proporción de fumar
production_data_drift = production_data.copy()
production_data_drift["Age"] = (production_data_drift["Age"] * 1.08).clip(upper=100).astype(int)
production_data_drift["Cholesterol"] = (production_data_drift["Cholesterol"] * 1.12).clip(upper=700).astype(int)
production_data_drift["MaxHR"] = (production_data_drift["MaxHR"] * 0.93).astype(int)

print(f"Referencia (train): {reference_data.shape}")
print(f"Producción (test):  {production_data_drift.shape}")

# %% Generar reporte de Data Drift con Evidently
report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
])

report.run(
    reference_data=reference_data,
    current_data=production_data_drift
)

# Exportar reporte HTML
report.save_html("../drift_report.html")
print("✅ Reporte exportado → drift_report.html")

# %% Revisar métricas de drift en consola
result = report.as_dict()

dataset_drift = result["metrics"][0]["result"]
print("\n=== RESUMEN DE DRIFT ===")
print(f"Número de columnas con drift: {dataset_drift['number_of_drifted_columns']}")
print(f"Total columnas analizadas   : {dataset_drift['number_of_columns']}")
print(f"¿Drift detectado?           : {dataset_drift['dataset_drift']}")

print("\n=== DRIFT POR COLUMNA ===")
for col_result in result["metrics"][1:]:
    if "column_name" in col_result.get("result", {}):
        col = col_result["result"]["column_name"]
        drift = col_result["result"].get("drift_detected", "N/A")
        score = col_result["result"].get("drift_score", "N/A")
        print(f"  {col:<20} drift={drift}  score={score:.4f}" if isinstance(score, float) else f"  {col}")

# %% [markdown]
# ## Interpretación
# - Si `dataset_drift = True`, el modelo puede estar viendo datos fuera de distribución.
# - Las columnas con mayor drift deberían revisarse manualmente.
# - Si el drift es persistente, considerar **reentrenamiento** del modelo con datos recientes.
#
# **Acción recomendada:**
# 1. Revisar `drift_report.html` en el navegador.
# 2. Si >30% de columnas tienen drift → activar pipeline de reentrenamiento.
# 3. Documentar el drift en el registro de experimentos (MLflow, W&B, etc.).
