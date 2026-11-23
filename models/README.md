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
