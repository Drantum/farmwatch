import logging
import threading
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


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
        alpr_detector_model: str = "yolo-v9-t-384",
        alpr_ocr_model: str = "cct-s-v2-global-model",
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
        }

    def _load_models(self) -> None:
        """Lädt YOLO und fast-alpr fehlertolerant."""
        try:
            from ultralytics import YOLO
            logger.info(f"Lade YOLO-Modell ({self.yolo_model_name}) auf {self.device}...")
            self.yolo_model = YOLO(self.yolo_model_name)
        except Exception as e:
            logger.error(f"Konnte YOLO nicht laden ({e}). Erkennung wird pausiert.")

        try:
            from fast_alpr import ALPR
            logger.info("Lade fast-alpr ONNX-Modelle...")
            self.alpr = ALPR(
                detector_model=self.alpr_detector_model,
                ocr_model=self.alpr_ocr_model,
            )
        except Exception as e:
            logger.warning(f"Konnte fast-alpr nicht laden ({e}). ALPR wird simuliert oder übersprungen.")

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
            for item in results:
                # Fast-ALPR liefert Vorhersagen mit text und confidence
                text = getattr(item, "text", "") or ""
                conf = float(getattr(item, "confidence", 0.0) or 0.0)
                box = getattr(item, "box", None)
                plates.append({"text": text, "confidence": conf, "box": box})
        except Exception as e:
            logger.warning(f"Fehler bei fast-alpr Erkennung: {e}")
        return plates
