import os
from pathlib import Path
import tempfile
import unittest

from app.cow_detector import (
    resolve_cow_model,
    list_available_models,
    CowDetector,
)


class TestCowDetector(unittest.TestCase):
    def test_resolve_cow_model_fallback(self):
        # Wenn kein Ordner und keine Datei existiert
        model, is_custom = resolve_cow_model("models/nonexistent.pt", models_dir="/nonexistent/dir")
        self.assertEqual(model, "yolov8n.pt")
        self.assertFalse(is_custom)

    def test_resolve_cow_model_custom_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_model = Path(tmpdir) / "colo_cow.pt"
            custom_model.write_bytes(b"dummy_weights")

            # 1. Direkter Pfad
            model, is_custom = resolve_cow_model(str(custom_model), models_dir=tmpdir)
            self.assertEqual(model, str(custom_model))
            self.assertTrue(is_custom)

            # 2. Automatische Erkennung in models_dir
            model, is_custom = resolve_cow_model("auto", models_dir=tmpdir)
            self.assertEqual(model, str(custom_model))
            self.assertTrue(is_custom)

    def test_list_available_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "colo_cow.pt"
            f1.write_bytes(b"x" * 1024)
            f2 = Path(tmpdir) / "custom.onnx"
            f2.write_bytes(b"y" * 2048)
            f3 = Path(tmpdir) / "ignore.txt"
            f3.write_bytes(b"test")

            models = list_available_models(tmpdir)
            names = [m["name"] for m in models]
            self.assertIn("colo_cow.pt", names)
            self.assertIn("custom.onnx", names)
            self.assertNotIn("ignore.txt", names)

    def test_cow_detector_status(self):
        detector = CowDetector(
            rtsp_url="rtsp://localhost:8554/kalb",
            model_path="yolov8n.pt",
            target_fps=5.0,
        )
        status = detector.get_status()
        self.assertIn("running", status)
        self.assertIn("cows_detected", status)
        self.assertIn("model_path", status)
        self.assertIn("is_custom_model", status)
        self.assertEqual(status["cows_detected"], 0)

    def test_cow_detector_placeholder_jpeg(self):
        detector = CowDetector(
            rtsp_url="rtsp://localhost:8554/kalb",
            model_path="yolov8n.pt",
        )
        jpeg = detector.get_latest_jpeg()
        # Bei get_latest_jpeg wird entweder bytes oder leer geliefert
        self.assertIsInstance(jpeg, bytes)


if __name__ == "__main__":
    unittest.main()
