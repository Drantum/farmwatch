# Eigene KI-Modelle (Optional)

In diesem Ordner können Sie selbst trainierte oder feingetunte Modelle ablegen.

## 1. LKW-Erkennung (Truck Detection)
- **Empfohlenes Modell:** `yolo11n.pt` oder `yolov8n.pt` (oder exportiert als `.onnx`).
- **Dateiname:** z.B. `lkw_custom.pt`
- **Konfiguration in `config.yaml`:**
  ```yaml
  detector:
    yolo_model: "models/lkw_custom.pt"
  ```

## 2. Kennzeichen-Detektor (Plate Detection)
- **Empfohlenes Modell:** YOLO11n trainiert auf Kennzeichen-Bounding-Boxes, exportiert als ONNX.
- **Dateiname:** z.B. `plate_detector.onnx`
- **Konfiguration in `config.yaml`:**
  ```yaml
  alpr:
    detector_model: "models/plate_detector.onnx"
  ```

## 3. Kennzeichen-OCR (Plate Text Recognition)
- **Empfohlenes Modell:** CCT oder CRNN ONNX-Modell trainiert auf deutscher FE-Schrift.
- **Dateiname:** z.B. `plate_ocr_de.onnx`
- **Konfiguration in `config.yaml`:**
  ```yaml
  alpr:
    ocr_model: "models/plate_ocr_de.onnx"
  ```

## 4. Kuh-Erkennung & Tracking für Abkalbebox (COLO / Eigene Modelle)
- **Datensatz & Vorlage:** [Niche-Lab/COLO](https://github.com/Niche-Lab/COLO) (COw LOcalization Dataset für Stall- und Rinderhaltung)
- **Unterstützte Formate:** PyTorch Checkpoints (`.pt`, z.B. YOLOv8 / YOLOv9 / YOLO11) oder exportiert als `.onnx`.
- **Dateiname:** z.B. `colo_cow.pt` oder `cow_tracker.pt`
- **Automatische Erkennung:**
  Legen Sie die Datei einfach hier in den Ordner `models/` ab. Das System erkennt Dateien mit `cow`, `colo` oder `kuh` im Namen automatisch!
- **Manuelle Konfiguration in `config.yaml`:**
  ```yaml
  cow_detector:
    enabled: true
    rtsp_url: "rtsp://go2rtc:8554/kalb"
    model_path: "models/colo_cow.pt"
    confidence_threshold: 0.35
    target_fps: 5.0
    device: "cpu"
    track: true       # Aktiviert ByteTrack-Objektverfolgung mit konsistenten IDs
  ```
- **Hinweis:** Liegt noch kein eigenes Modell im Ordner, verwendet das System automatisch das Standardmodell `yolov8n.pt` (COCO-Klasse "cow").
- **Live-Ansicht im Dashboard:** Über die Schaltfläche `🎯 KI Boxen` bei der Abkalbebox können Sie die Live-Ansicht mit Bounding-Boxes und Track-IDs direkt im Browser betrachten.
