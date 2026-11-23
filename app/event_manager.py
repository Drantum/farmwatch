import logging
from pathlib import Path
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import cv2
except ImportError:
    cv2 = None

from app.alpr_logic import PlateVoteCollector
from app.clip_maker import build_event_clip
from app.database import Database
from app.vision_client import VisionClient

logger = logging.getLogger(__name__)


class EventManager:
    """
    Steuert den Lebenszyklus von LKW-Ereignissen:
    - Erkennt Start des Ereignisses beim ersten LKW-Frame
    - Sammelt kontinuierlich Kennzeichen und Snapshots
    - Schließt das Ereignis ab, wenn 8 Sekunden lang kein LKW mehr erkannt wird
    - Wartet 12 Sekunden vor dem Schneiden, um alle Segmente zu sichern
    - Schneidet den Clip (3s Vorlauf, 2s Nachlauf) und speichert Snapshot & DB-Eintrag
    """

    def __init__(
        self,
        db: Database,
        media_dir: Path,
        seg_dir: Path,
        vision_client: VisionClient,
        idle_timeout: float = 8.0,
        segment_flush_wait: float = 12.0,
        pre_buffer: float = 3.0,
        post_buffer: float = 2.0,
        alpr_min_confidence: float = 0.35,
    ):
        self.db = db
        self.media_dir = Path(media_dir)
        self.seg_dir = Path(seg_dir)
        self.vision_client = vision_client
        self.idle_timeout = idle_timeout
        self.segment_flush_wait = segment_flush_wait
        self.pre_buffer = pre_buffer
        self.post_buffer = post_buffer
        self.alpr_min_confidence = alpr_min_confidence

        self._lock = threading.Lock()
        self.is_active: bool = False
        self.start_time: float = 0.0
        self.last_seen_time: float = 0.0

        self.vote_collector: PlateVoteCollector = PlateVoteCollector(self.alpr_min_confidence)
        self.best_truck_frame: Optional[Any] = None
        self.best_truck_conf: float = 0.0

    def on_frame_analyzed(
        self,
        truck_detected: bool,
        truck_conf: float,
        plates: List[Dict[str, Any]],
        frame: Optional[Any],
        timestamp: Optional[float] = None,
    ) -> None:
        """Wird für jeden analysierten Video-Frame (~5 FPS) aufgerufen."""
        ts = timestamp or time.time()

        with self._lock:
            if truck_detected:
                if not self.is_active:
                    self.is_active = True
                    self.start_time = ts
                    self.vote_collector = PlateVoteCollector(self.alpr_min_confidence)
                    self.best_truck_frame = None
                    self.best_truck_conf = 0.0
                    logger.info(f"🚚 LKW erkannt! Ereignis gestartet um {datetime.fromtimestamp(ts).strftime('%H:%M:%S')}")

                self.last_seen_time = ts

                if frame is not None and truck_conf > self.best_truck_conf:
                    self.best_truck_conf = truck_conf
                    self.best_truck_frame = frame.copy() if hasattr(frame, "copy") else frame

                if frame is not None and plates:
                    for p in plates:
                        raw_text = p.get("text", "")
                        conf = float(p.get("confidence", 0.0))
                        self.vote_collector.add_detection(raw_text, conf, frame)

            else:
                if self.is_active:
                    elapsed_without_truck = ts - self.last_seen_time
                    if elapsed_without_truck >= self.idle_timeout:
                        logger.info(
                            f"LKW seit {self.idle_timeout}s nicht mehr im Bild. Ereignis wird abgeschlossen..."
                        )
                        self._trigger_finalization()

    def _trigger_finalization(self) -> None:
        """Kopiert den aktuellen Zustand und stößt die Finalisierung im Hintergrund an."""
        start_ts = self.start_time
        end_ts = self.last_seen_time
        collector = self.vote_collector
        truck_frame = (
            self.best_truck_frame.copy()
            if (self.best_truck_frame is not None and hasattr(self.best_truck_frame, "copy"))
            else self.best_truck_frame
        )

        self.is_active = False
        self.start_time = 0.0
        self.last_seen_time = 0.0
        self.best_truck_frame = None
        self.best_truck_conf = 0.0

        thread = threading.Thread(
            target=self._finalize_event_worker,
            args=(start_ts, end_ts, collector, truck_frame),
            daemon=True,
        )
        thread.start()

    def _finalize_event_worker(
        self,
        start_ts: float,
        end_ts: float,
        collector: PlateVoteCollector,
        truck_frame: Optional[Any],
    ) -> None:
        """Wartet auf das finale 10s Segment und verarbeitet das fertige Ereignis."""
        try:
            logger.info(f"Warte {self.segment_flush_wait}s auf Abschluss des letzten Video-Segments...")
            time.sleep(self.segment_flush_wait)

            dt_event = datetime.fromtimestamp(start_ts)
            timestamp_str = dt_event.strftime("%Y%m%d_%H%M%S")
            duration_seconds = max(0.0, end_ts - start_ts)

            plate_formatted, plate_raw, plate_conf, plate_frame = collector.get_winner()
            best_snapshot = plate_frame if plate_frame is not None else truck_frame

            snapshot_filename = f"snapshot_{timestamp_str}.jpg"
            snapshot_path = self.media_dir / snapshot_filename

            if best_snapshot is not None:
                if cv2 is not None:
                    cv2.imwrite(str(snapshot_path), best_snapshot, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    # Fallback ohne OpenCV (z.B. Mock in Tests)
                    snapshot_path.write_text("dummy_snapshot")
                logger.info(f"Snapshot gespeichert: {snapshot_path}")
            else:
                snapshot_filename = ""

            clip_filename = f"clip_{timestamp_str}.mp4"
            clip_path = self.media_dir / clip_filename

            clip_success = build_event_clip(
                seg_dir=self.seg_dir,
                output_path=clip_path,
                start_time=start_ts,
                end_time=end_ts,
                pre_buffer=self.pre_buffer,
                post_buffer=self.post_buffer,
            )
            if not clip_success:
                clip_filename = ""

            company_name = None
            if self.vision_client.enabled and best_snapshot is not None:
                logger.info("Frage Speditionsnamen über Vision API an...")
                company_name = self.vision_client.extract_company_from_frame(best_snapshot)
                if company_name:
                    logger.info(f"Spedition erkannt: {company_name}")

            event_id = self.db.insert_event(
                created_at=dt_event,
                plate=plate_formatted or "Unbekannt",
                plate_raw=plate_raw or "",
                plate_confidence=plate_conf if plate_formatted else 0.0,
                company=company_name,
                snapshot_filename=snapshot_filename,
                clip_filename=clip_filename,
                duration_seconds=round(duration_seconds, 1),
            )

            logger.info(
                f"✅ Ereignis #{event_id} erfolgreich gespeichert! Kennzeichen: {plate_formatted or 'Unbekannt'}, "
                f"Dauer: {duration_seconds:.1f}s, Clip: {clip_filename}"
            )

        except Exception as e:
            logger.error(f"Fehler bei der Ereignis-Finalisierung: {e}", exc_info=True)

    def get_status(self) -> dict:
        with self._lock:
            return {
                "active_event": self.is_active,
                "start_time": self.start_time if self.is_active else None,
                "duration_so_far": round(time.time() - self.start_time, 1) if self.is_active else 0,
            }
