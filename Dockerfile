FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Europe/Berlin

# System-Abhängigkeiten inkl. ffmpeg für verlustfreies Schneiden & Video-Verarbeitung
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python-Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Standard-KI-Modelle vorab herunterladen, damit der Container sofort einsatzbereit ist
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true
RUN python -c "from fast_alpr import ALPR; ALPR(detector_model='yolo-v9-t-384-license-plate-end2end', ocr_model='cct-s-v2-global-model')" || true

# Anwendungs-Code kopieren
COPY app/ ./app/
COPY config.yaml ./config.yaml

# Standard-Datenordner anlegen
RUN mkdir -p /app/data/media /app/data/seg

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/status || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
