"""
Beispielskript zum Trainieren / Feintunen eines eigenen LKW-Modells mit YOLO11.

Voraussetzungen:
    pip install ultralytics

Verwendung:
    python scripts/train_lkw_yolo.py
"""

from ultralytics import YOLO

def train():
    # 1. Kleinstes Modell (Nano) laden - perfekt für CPU / Intel iGPU
    model = YOLO("yolo11n.pt")

    # 2. Modell auf eigenen Hofkamera-Bildern trainieren
    # dataset.yaml enthält Pfade zu Bildern und Klassen ('truck')
    results = model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        device="cpu",      # oder "0" falls CUDA GPU vorhanden
        batch=16,
        project="runs/lkw_training",
        name="lkw_hof",
    )

    # 3. Bestes Modell nach ONNX exportieren (schnellste CPU-Inferenz)
    best_model = YOLO("runs/lkw_training/lkw_hof/weights/best.pt")
    best_model.export(format="onnx")

    print("\nTraining abgeschlossen!")
    print("Kopieren Sie die Datei 'best.pt' oder 'best.onnx' in den Ordner 'models/'")

if __name__ == "__main__":
    train()
