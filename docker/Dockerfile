# ─── ETAPA 3: Dockerfile ─────────────────────────────────────────
# Imagen base liviana de Python
FROM python:3.11-slim

# Evitar prompts interactivos durante el build
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar dependencias primero (cache de capas)
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y el modelo
COPY app/api.py .
COPY model.joblib .

# Variables de entorno de la app
ENV MODEL_PATH=model.joblib \
    PORT=8000

EXPOSE 8000

# Comando de arranque
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
