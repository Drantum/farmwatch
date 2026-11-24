from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


def extract_plate_data(item: Any) -> tuple[str, float, Any]:
    """
    Liest Text, Confidence und Box aus fast-alpr Rückgabe-Objekten.
    Unterstützt sowohl ALPRResult(detection, ocr), AlprPrediction als auch dicts.
    """
    text = ""
    conf = 0.0
    box = None

    if isinstance(item, dict):
        text = item.get("text") or (item.get("ocr", {}).get("text") if isinstance(item.get("ocr"), dict) else "") or ""
        conf = item.get("confidence") or (item.get("ocr", {}).get("confidence") if isinstance(item.get("ocr"), dict) else 0.0) or 0.0
        box = item.get("box") or (item.get("detection", {}).get("box") if isinstance(item.get("detection"), dict) else None)
    else:
        # 1. Text
        if hasattr(item, "text") and getattr(item, "text"):
            text = str(getattr(item, "text"))
        elif hasattr(item, "ocr") and hasattr(item.ocr, "text"):
            text = str(getattr(item.ocr, "text"))
        elif hasattr(item, "plate") and getattr(item, "plate"):
            text = str(getattr(item.plate))

        # 2. Confidence
        if hasattr(item, "confidence") and getattr(item, "confidence") is not None:
            conf = float(getattr(item, "confidence"))
        elif hasattr(item, "ocr") and hasattr(item.ocr, "confidence") and item.ocr.confidence is not None:
            conf = float(getattr(item.ocr, "confidence"))
        elif hasattr(item, "detection") and hasattr(item.detection, "confidence") and item.detection.confidence is not None:
            conf = float(getattr(item.detection, "confidence"))

        # 3. Bounding Box
        if hasattr(item, "box"):
            box = getattr(item, "box")
        elif hasattr(item, "detection") and hasattr(item.detection, "box"):
            box = getattr(item.detection, "box")

    return text.strip(), conf, box


class StreamDetector:
    """
    Holt Video-Frames vom RTSP-Stream (go2rtc) und wertet diese mit ~5 FPS aus:
    1. LKW-Erkennung mittels YOLO (COCO-Klasse 'truck')
    2. Wenn LKW erkannt: Kennzeichenerkennung mit fast-alpr
    3. Weiterleitung der Ergebnisse an den EventManager
    """

    def __init__(
        self,
        rtsp_url: str,
        event_manager: Any,
        target_fps: float = 5.0,
        yolo_model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.45,
        device: str = "cpu",
        target_classes: Optional[List[str]] = None,
        alpr_detector_model: str = "yolo-v9-t-384-license-plate-end2end",
        alpr_ocr_model: str = "cct-s-v2-global-model",
        **kwargs: Any,
    ):
        self.rtsp_url = rtsp_url
        self.event_manager = event_manager
        self.target_fps = target_fps
        self.frame_interval = 1.0 / max(1.0, target_fps)
        self.yolo_model_name = yolo_model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.target_classes = [c.lower() for c in (target_classes or ["truck"])]
        self.alpr_detector_model = alpr_detector_model
        self.alpr_ocr_model = alpr_ocr_model

        self._stop_event = threading.Event()
        self._grabber_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._last_frame_time: float = 0.0

        self.yolo_model = None
        self.alpr = None
        self.yolo_load_error: Optional[str] = None
        self.alpr_load_error: Optional[str] = None

        self.fps_measured: float = 0.0
        self.is_connected: bool = False

    def start(self) -> None:
        """Initialisiert Modelle und startet Frame-Grabber & Worker."""
        if self.is_running():
            return

        self._load_models()

        self._stop_event.clear()
        self._grabber_thread = threading.Thread(target=self._run_frame_grabber, daemon=True)
        self._worker_thread = threading.Thread(target=self._run_detection_loop, daemon=True)

        self._grabber_thread.start()
        self._worker_thread.start()
        logger.info(f"Stream-Detektor gestartet (Ziel-FPS: {self.target_fps}, Device: {self.device})")

    def stop(self) -> None:
        self._stop_event.set()
        self.is_connected = False
        logger.info("Stream-Detektor gestoppt.")

    def is_running(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def get_status(self) -> dict:
        return {
            "running": self.is_running(),
            "connected": self.is_connected,
            "fps_measured": round(self.fps_measured, 1),
            "device": self.device,
            "yolo_model": self.yolo_model_name,
            "yolo_loaded": self.yolo_model is not None,
            "yolo_error": self.yolo_load_error,
            "alpr_loaded": self.alpr is not None,
            "alpr_error": self.alpr_load_error,
        }

    def _load_models(self) -> None:
        """Lädt YOLO und fast-alpr fehlertolerant."""
        self.yolo_load_error = None
        self.alpr_load_error = None

        try:
            from ultralytics import YOLO
            logger.info(f"Lade YOLO-Modell ({self.yolo_model_name}) auf {self.device}...")
            self.yolo_model = YOLO(self.yolo_model_name)
            logger.info("YOLO-Modell erfolgreich geladen!")
        except Exception as e:
            self.yolo_load_error = str(e)
            logger.error(f"❌ Konnte YOLO nicht laden ({e}). LKW-Erkennung inaktiv!", exc_info=True)

        try:
            import os
            from fast_alpr import ALPR
            detector_model = self.alpr_detector_model
            # Automatische Ergänzung bei Kurznamen wie "yolo-v9-t-384"
            if detector_model and not os.path.exists(detector_model):
                if detector_model in ("yolo-v9-t-384", "yolov9-t-384"):
                    detector_model = "yolo-v9-t-384-license-plate-end2end"
                elif not detector_model.endswith("-license-plate-end2end") and detector_model.startswith("yolo-v9-"):
                    detector_model = f"{detector_model}-license-plate-end2end"

            logger.info(f"Lade fast-alpr ONNX-Modelle (Detector: {detector_model}, OCR: {self.alpr_ocr_model})...")
            try:
                self.alpr = ALPR(
                    detector_model=detector_model,
                    ocr_model=self.alpr_ocr_model,
                )
            except FileNotFoundError as fnf_err:
                fallback_detector = "yolo-v9-t-384-license-plate-end2end"
                if detector_model != fallback_detector:
                    logger.warning(f"Konnte Detector '{detector_model}' nicht finden ({fnf_err}). Versuche Standard-Modell '{fallback_detector}'...")
                    self.alpr = ALPR(
                        detector_model=fallback_detector,
                        ocr_model=self.alpr_ocr_model,
                    )
                else:
                    raise
            logger.info("fast-alpr erfolgreich initialisiert und bereit!")
        except Exception as e:
            self.alpr_load_error = str(e)
            logger.error(f"❌ Konnte fast-alpr nicht laden: {e}. ALPR-Erkennung ist inaktiv!", exc_info=True)

    def _run_frame_grabber(self) -> None:
        """
        Liest den RTSP-Stream kontinuierlich aus, um Paketstau zu vermeiden,
        und speichert stets das neueste Bild ab.
        """
        while not self._stop_event.is_set():
            logger.info(f"Verbinde RTSP Frame-Grabber mit: {self.rtsp_url}")
            cap = cv2.VideoCapture(self.rtsp_url)
            
            # Puffer auf 1 Frame minimieren
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                logger.warning(f"Konnte RTSP-Stream nicht öffnen: {self.rtsp_url}. Warte 3s...")
                self.is_connected = False
                self._stop_event.wait(timeout=3.0)
                continue

            self.is_connected = True
            logger.info(f"RTSP-Stream erfolgreich geöffnet: {self.rtsp_url}")

            consecutive_failures = 0
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        logger.warning("Zu viele aufeinanderfolgende Lesefehler vom RTSP-Stream. Reconnect...")
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0
                with self._frame_lock:
                    self._latest_frame = frame
                    self._last_frame_time = time.time()

            self.is_connected = False
            cap.release()
            if not self._stop_event.is_set():
                self._stop_event.wait(timeout=2.0)

    def _run_detection_loop(self) -> None:
        """Verarbeitet Frames im festen Intervall (~5 FPS)."""
        fps_counter = 0
        fps_start = time.time()

        while not self._stop_event.is_set():
            loop_start = time.time()

            # Neuesten Frame holen
            frame = None
            with self._frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()

            if frame is not None and self.yolo_model is not None:
                try:
                    truck_detected, truck_conf = self._detect_truck(frame)
                    plates = []

                    if truck_detected and self.alpr is not None:
                        plates = self._detect_plates(frame)

                    # An EventManager übergeben
                    self.event_manager.on_frame_analyzed(
                        truck_detected=truck_detected,
                        truck_conf=truck_conf,
                        plates=plates,
                        frame=frame,
                        timestamp=loop_start,
                    )

                except Exception as e:
                    logger.error(f"Fehler bei Frame-Auswertung: {e}")

            fps_counter += 1
            now = time.time()
            if now - fps_start >= 2.0:
                self.fps_measured = fps_counter / (now - fps_start)
                fps_counter = 0
                fps_start = now

            # Intervall-Wartezeit für ~5 FPS
            elapsed = time.time() - loop_start
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

    def _detect_truck(self, frame: np.ndarray) -> tuple[bool, float]:
        """Prüft mit YOLO, ob ein LKW im Bild ist."""
        results = self.yolo_model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        max_conf = 0.0
        truck_found = False

        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = r.names.get(cls_id, "").lower()
                conf = float(box.conf[0])

                if cls_name in self.target_classes and conf >= self.confidence_threshold:
                    truck_found = True
                    if conf > max_conf:
                        max_conf = conf

        return truck_found, max_conf

    def _detect_plates(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Führt fast-alpr Kennzeichenerkennung auf dem Frame durch."""
        plates = []
        try:
            results = self.alpr.predict(frame)
            if results:
                logger.info(f"🔍 fast-alpr: {len(results)} Kennzeichen-Kandidat(en) im Frame gefunden.")
            for item in results:
                text, conf, box = extract_plate_data(item)
                logger.info(f"   📋 fast-alpr Roh-Erkennung: '{text}' (Confidence: {conf:.2f}, Box: {box})")
                plates.append({"text": text, "confidence": conf, "box": box})
        except Exception as e:
            logger.error(f"Fehler bei fast-alpr Erkennung: {e}", exc_info=True)
        return plates
