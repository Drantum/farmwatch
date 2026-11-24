from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


def resolve_cow_model(
    configured_path: str,
    models_dir: Path | str = "./models",
) -> Tuple[str, bool]:
    """
    Sucht nach dem gewünschten Modell für die Kuh-Erkennung.
    Gibt (model_path_or_name, is_custom) zurück.

    Prioritäten:
    1. Wenn configured_path existiert -> verwenden
    2. Wenn in models_dir eine .pt oder .onnx Datei mit 'cow', 'colo', 'kuh' liegt -> verwenden
    3. Wenn in models_dir irgendeine .pt oder .onnx Datei liegt -> verwenden
    4. Fallback: 'yolov8n.pt' (nutzt COCO-Klasse 19 'cow')
    """
    # 1. Konfigurierter Pfad existiert
    if configured_path and os.path.exists(configured_path):
        return configured_path, True

    # Auch im Docker /app/models bzw. relativem Ordner suchen
    models_path = Path(models_dir)
    if not models_path.exists():
        alt_path = Path("/app/models")
        if alt_path.exists():
            models_path = alt_path

    if models_path.exists():
        # Suche nach spezifischen Namen (COLO, Cow, Kuh)
        candidates = list(models_path.glob("*.pt")) + list(models_path.glob("*.onnx"))
        for cand in candidates:
            c_name = cand.name.lower()
            if any(term in c_name for term in ("cow", "colo", "kuh", "kalb")):
                return str(cand), True

        # Andere pt / onnx Modelle (z.B. best.pt)
        for cand in candidates:
            if cand.name.lower() in ("best.pt", "last.pt", "custom.pt"):
                return str(cand), True

    # Wenn configured_path ein Standard-Modellname ohne Verzeichnispfad ist (z.B. yolov8n.pt, yolo11n.pt)
    if configured_path and configured_path.endswith(".pt") and Path(configured_path).name == configured_path:
        return configured_path, False

    # Fallback
    return "yolov8n.pt", False


def list_available_models(models_dir: Path | str = "./models") -> List[Dict[str, Any]]:
    """Listet alle im models-Ordner gefundenen Modelle auf."""
    results = []
    models_path = Path(models_dir)
    if not models_path.exists():
        alt_path = Path("/app/models")
        if alt_path.exists():
            models_path = alt_path

    if models_path.exists():
        for file in models_path.glob("*"):
            if file.suffix.lower() in (".pt", ".onnx"):
                size_mb = round(file.stat().st_size / (1024 * 1024), 2)
                results.append({
                    "name": file.name,
                    "path": str(file),
                    "size_mb": size_mb,
                    "type": file.suffix.lower().replace(".", ""),
                })
    return results


class CowDetector:
    """
    Erkennt und trackt Kühe auf dem RTSP-Stream der Abkalbebox.
    Zeichnet Bounding-Boxes, Konfidenz und Track-IDs ein
    und stellt einen MJPEG-Stream für das Web-Dashboard bereit.
    """

    def __init__(
        self,
        rtsp_url: str = "rtsp://go2rtc:8554/kalb",
        model_path: str = "models/colo_cow.pt",
        confidence_threshold: float = 0.35,
        target_fps: float = 5.0,
        device: str = "cpu",
        track: bool = True,
        target_classes: Optional[List[str]] = None,
        models_dir: str = "./models",
        **kwargs: Any,
    ):
        self.rtsp_url = rtsp_url
        self.configured_model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_fps = max(1.0, min(15.0, target_fps))
        self.frame_interval = 1.0 / self.target_fps
        self.device = device
        self.track = track
        self.target_classes = [c.lower() for c in (target_classes or ["cow"])]
        self.models_dir = models_dir

        # Modell-Pfad auflösen
        self.resolved_model_path, self.is_custom_model = resolve_cow_model(
            model_path, models_dir=self.models_dir
        )

        self.model: Any = None
        self.model_load_error: Optional[str] = None
        self.model_classes: Dict[int, str] = {}
        self.cow_class_ids: List[int] = []

        # Threading & Status
        self._stop_event = threading.Event()
        self._grabber_thread: Optional[threading.Thread] = None
        self._detection_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._latest_raw_frame: Optional[Any] = None
        self._latest_annotated_jpeg: Optional[bytes] = None
        self._last_detection_time: float = 0.0

        # Statistik
        self.is_connected = False
        self.cows_detected = 0
        self.active_tracks: List[int] = []
        self.fps_measured = 0.0

    def start(self) -> None:
        """Startet den Modell-Load, Frame-Grabber und Detection-Thread."""
        self._stop_event.clear()
        self._load_model()

        self._grabber_thread = threading.Thread(
            target=self._run_frame_grabber,
            name="CowFrameGrabber",
            daemon=True,
        )
        self._grabber_thread.start()

        self._detection_thread = threading.Thread(
            target=self._run_detection_loop,
            name="CowDetectionLoop",
            daemon=True,
        )
        self._detection_thread.start()

        logger.info(
            f"🐮 CowDetector gestartet für Stream: {self.rtsp_url} "
            f"(Modell: {self.resolved_model_path}, Custom: {self.is_custom_model}, Ziel-FPS: {self.target_fps})"
        )

    def stop(self) -> None:
        """Stoppt alle Threads."""
        self._stop_event.set()
        if self._grabber_thread and self._grabber_thread.is_alive():
            self._grabber_thread.join(timeout=2.0)
        if self._detection_thread and self._detection_thread.is_alive():
            self._detection_thread.join(timeout=2.0)
        logger.info("🐮 CowDetector gestoppt.")

    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self.is_running(),
            "connected": self.is_connected,
            "cows_detected": self.cows_detected,
            "active_tracks": self.active_tracks,
            "fps_measured": round(self.fps_measured, 1),
            "model_path": self.resolved_model_path,
            "is_custom_model": self.is_custom_model,
            "model_loaded": self.model is not None,
            "model_error": self.model_load_error,
            "available_models": [m["name"] for m in list_available_models(self.models_dir)],
        }

    def _load_model(self) -> None:
        """Lädt Ultralytics YOLO-Modell fehlertolerant."""
        self.model = None
        self.model_load_error = None
        try:
            from ultralytics import YOLO

            logger.info(f"🐮 Lade Kuh-Detektor Modell: {self.resolved_model_path} auf {self.device}...")
            self.model = YOLO(self.resolved_model_path)
            self.model_classes = getattr(self.model, "names", {})

            # Relevante Klassen-IDs ermitteln
            if self.is_custom_model:
                # Bei dedizierten Kuh-Modellen (z.B. COLO) oft nur 1 Klasse oder spezifische Klassen
                matched = [
                    cls_id
                    for cls_id, name in self.model_classes.items()
                    if any(term in name.lower() for term in ("cow", "cattle", "kuh", "kalb"))
                ]
                if matched:
                    self.cow_class_ids = matched
                elif len(self.model_classes) == 1:
                    # Einzige Klasse im Modell
                    self.cow_class_ids = [0]
                else:
                    # Alle Klassen des Custom-Modells erkennen
                    self.cow_class_ids = list(self.model_classes.keys())
            else:
                # Standard-YOLO (COCO): Klasse 'cow' (ID 19)
                self.cow_class_ids = [
                    cls_id
                    for cls_id, name in self.model_classes.items()
                    if name.lower() in self.target_classes
                ]
                if not self.cow_class_ids and 19 in self.model_classes:
                    self.cow_class_ids = [19]

            logger.info(
                f"🐮 Kuh-Modell erfolgreich geladen! Ziel-Klassen-IDs: {self.cow_class_ids} "
                f"({[self.model_classes.get(cid, str(cid)) for cid in self.cow_class_ids]})"
            )
        except Exception as e:
            self.model_load_error = str(e)
            logger.error(f"❌ Fehler beim Laden des Kuh-Modells: {e}", exc_info=True)

    def _run_frame_grabber(self) -> None:
        """Holt kontinuierlich die neuesten Frames vom RTSP-Stream."""
        if cv2 is None:
            logger.warning("OpenCV nicht verfügbar. Frame-Grabber inaktiv.")
            return

        while not self._stop_event.is_set():
            logger.info(f"Verbinde Kuh-Kamera Frame-Grabber mit: {self.rtsp_url}")
            cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self.is_connected = False
                logger.warning(f"Konnte Stream Abkalbebox nicht öffnen: {self.rtsp_url}. Warte 3s...")
                self._stop_event.wait(timeout=3.0)
                continue

            self.is_connected = True
            consecutive_failures = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        logger.warning("Zu viele Lesefehler auf Stream Abkalbebox. Reconnect...")
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0
                with self._lock:
                    self._latest_raw_frame = frame

            self.is_connected = False
            cap.release()
            if not self._stop_event.is_set():
                self._stop_event.wait(timeout=2.0)

    def _run_detection_loop(self) -> None:
        """Führt zyklisch Inferenz & Tracking durch und zeichnet Bounding Boxes ein."""
        if cv2 is None or np is None:
            return

        fps_counter = 0
        fps_start = time.time()

        while not self._stop_event.is_set():
            loop_start = time.time()
            frame = None

            with self._lock:
                if self._latest_raw_frame is not None:
                    frame = self._latest_raw_frame.copy()

            if frame is not None and self.model is not None:
                try:
                    annotated_frame, count, tracks = self._process_frame(frame)
                    jpeg_bytes = self._encode_jpeg(annotated_frame)

                    with self._lock:
                        self._latest_annotated_jpeg = jpeg_bytes
                        self.cows_detected = count
                        self.active_tracks = tracks
                        self._last_detection_time = time.time()
                except Exception as e:
                    logger.error(f"Fehler bei Kuh-Erkennung: {e}")
            elif frame is not None:
                # Kein Modell geladen: Rohes Frame mit Hinweis ausgeben
                annotated = frame.copy()
                self._draw_overlay_banner(annotated, "KI-Modell nicht aktiv", (0, 0, 255))
                jpeg_bytes = self._encode_jpeg(annotated)
                with self._lock:
                    self._latest_annotated_jpeg = jpeg_bytes

            fps_counter += 1
            now = time.time()
            if now - fps_start >= 2.0:
                self.fps_measured = fps_counter / (now - fps_start)
                fps_counter = 0
                fps_start = now

            elapsed = time.time() - loop_start
            sleep_time = self.frame_interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, List[int]]:
        """Führt Tracking / Erkennung aus und annotiert das Bild mit Boxen."""
        count = 0
        tracks = []
        classes_filter = self.cow_class_ids if self.cow_class_ids else None

        # 1. Inferenz mit Ultralytics
        results = None
        if self.track:
            try:
                results = self.model.track(
                    source=frame,
                    conf=self.confidence_threshold,
                    classes=classes_filter,
                    device=self.device,
                    verbose=False,
                    persist=True,
                )
            except Exception:
                # Fallback auf einfaches predict falls tracker fehlschlägt
                pass

        if results is None:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                classes=classes_filter,
                device=self.device,
                verbose=False,
            )

        # 2. Bounding Boxes einzeichnen
        annotated = frame.copy()
        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes

            if boxes is not None and len(boxes) > 0:
                count = len(boxes)
                # Ultralytics plot() für professionelle Bounding Boxes & Labels
                try:
                    annotated = res.plot(line_width=2, font_size=0.6)
                except Exception:
                    annotated = frame.copy()

                for box in boxes:
                    if box.id is not None:
                        try:
                            tid = int(box.id[0])
                            tracks.append(tid)
                        except Exception:
                            pass

        # 3. Status-Banner oben links einblenden
        model_name = Path(self.resolved_model_path).name
        banner_text = f"🐮 {count} Kuh/Kühe erfasst"
        if tracks:
            banner_text += f" (IDs: {', '.join(str(t) for t in tracks[:5])})"
        banner_text += f" | {model_name} | {self.fps_measured:.1f} FPS"

        color = (34, 197, 94) if count > 0 else (100, 116, 139)  # Grün wenn Kuh da, sonst Grau
        self._draw_overlay_banner(annotated, banner_text, color)

        return annotated, count, tracks

    def _draw_overlay_banner(self, img: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
        """Zeichnet ein lesbares HUD-Banner im oberen Bereich des Bildes."""
        if cv2 is None or img is None:
            return
        h, w = img.shape[:2]
        banner_h = max(32, int(h * 0.06))
        # Halbtransparenter Balken
        sub = img[0:banner_h, 0:w]
        black = np.zeros_like(sub)
        cv2.addWeighted(sub, 0.4, black, 0.6, 0, sub)
        img[0:banner_h, 0:w] = sub

        # Farbiger Indikator-Punkt links
        cv2.circle(img, (18, banner_h // 2), 6, color, -1)

        # Text zeichnen
        font_scale = max(0.45, min(0.8, w / 1200.0))
        cv2.putText(
            img,
            text,
            (32, int(banner_h * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _encode_jpeg(self, img: np.ndarray) -> bytes:
        """Komprimiert das annotierte Bild als JPEG für die Web-Auslieferung."""
        if cv2 is None:
            return b""
        ret, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return buf.tobytes() if ret else b""

    def get_latest_jpeg(self) -> bytes:
        """Gibt das neueste annotierte JPEG oder ein Platzhalter-Bild zurück."""
        with self._lock:
            if self._latest_annotated_jpeg is not None:
                return self._latest_annotated_jpeg

        # Platzhalter erzeugen
        return self._create_placeholder_jpeg()

    def _create_placeholder_jpeg(self) -> bytes:
        """Erzeugt ein informatives Platzhalter-Bild, falls der Stream noch lädt."""
        if cv2 is None or np is None:
            return b""
        img = np.zeros((360, 640, 3), dtype=np.uint8)
        img[:] = (30, 41, 59)  # Dunkles Schieferblau
        text = "Abkalbebox: Verbinde Stream & KI..." if self.is_running() else "Abkalbebox: KI inaktiv"
        cv2.putText(
            img,
            text,
            (80, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (226, 232, 240),
            2,
            cv2.LINE_AA,
        )
        return self._encode_jpeg(img)

    def generate_mjpeg(self):
        """Generator für HTTP MJPEG Video-Stream (Browser <img> kompatibel)."""
        while self.is_running():
            frame_bytes = self.get_latest_jpeg()
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(self.frame_interval)
