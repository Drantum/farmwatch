import logging
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class StreamRecorder:
    """
    Überwacht den kontinuierlichen ffmpeg-Aufnahme-Prozess, der den RTSP-Stream
    ohne Transkodierung (-c copy) in 10-Sekunden-Segmente zerlegt.
    Startet den Prozess bei Stream-Abbrüchen automatisch neu und löscht alte Segmente (> 10 Min).
    """

    def __init__(
        self,
        rtsp_url: str,
        seg_dir: Path,
        segment_time: int = 10,
        retention_minutes: int = 10,
        **kwargs,
    ):
        self.rtsp_url = rtsp_url
        self.seg_dir = Path(seg_dir)
        self.segment_time = segment_time
        self.retention_seconds = retention_minutes * 60

        self.seg_dir.mkdir(parents=True, exist_ok=True)

        self._stop_event = threading.Event()
        self._record_thread: Optional[threading.Thread] = None
        self._cleaner_thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen] = None
        self._last_segment_time: float = 0.0

    def start(self) -> None:
        """Startet den Recorder und den Cleaner in separaten Hintergrund-Threads."""
        if self.is_running():
            logger.warning("Recorder läuft bereits.")
            return

        self._stop_event.clear()
        self._record_thread = threading.Thread(target=self._run_recorder_supervisor, daemon=True)
        self._cleaner_thread = threading.Thread(target=self._run_cleaner_loop, daemon=True)

        self._record_thread.start()
        self._cleaner_thread.start()
        logger.info(f"Stream-Recorder gestartet für: {self.rtsp_url}")

    def stop(self) -> None:
        """Beendet den Aufnahme-Prozess und die Threads sauber."""
        self._stop_event.set()
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
        logger.info("Stream-Recorder gestoppt.")

    def is_running(self) -> bool:
        return self._record_thread is not None and self._record_thread.is_alive()

    def get_status(self) -> dict:
        """Liefert Diagnose-Daten für den Recorder-Status."""
        seg_files = list(self.seg_dir.glob("seg_*.mp4"))
        return {
            "running": self.is_running(),
            "active_process": self._process is not None and self._process.poll() is None,
            "segment_count": len(seg_files),
            "rtsp_url": self.rtsp_url,
        }

    def _run_recorder_supervisor(self) -> None:
        """Supervisor-Schleife: Hält den ffmpeg-Prozess am Laufen."""
        output_pattern = str(self.seg_dir / "seg_%Y%m%d_%H%M%S.mp4")

        while not self._stop_event.is_set():
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "warning",
                "-rtsp_transport", "tcp",
                "-i", self.rtsp_url,
                "-c", "copy",
                "-f", "segment",
                "-segment_time", str(self.segment_time),
                "-segment_format", "mp4",
                "-reset_timestamps", "1",
                "-strftime", "1",
                output_pattern,
            ]

            try:
                logger.info(f"Starte ffmpeg-Segmenter für {self.rtsp_url}...")
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Warten bis ffmpeg beendet oder Stop signalisiert wird
                while self._process.poll() is None:
                    if self._stop_event.wait(timeout=1.0):
                        break

                if self._process.poll() is not None:
                    _, stderr = self._process.communicate()
                    if not self._stop_event.is_set():
                        logger.warning(
                            f"ffmpeg-Segmenter unerwartet beendet (Exit-Code {self._process.returncode}): {stderr.strip()}"
                        )
            except FileNotFoundError:
                logger.error("ffmpeg nicht im Systempfad gefunden. Recording pausiert.")
                self._stop_event.wait(timeout=10.0)
            except Exception as e:
                logger.error(f"Fehler im Recorder-Supervisor: {e}")

            # Wartepause vor Neustart, um CPU-Schleifen bei Verbindungsausfall zu verhindern
            if not self._stop_event.is_set():
                self._stop_event.wait(timeout=3.0)

    def _run_cleaner_loop(self) -> None:
        """Bereinigt alle 30 Sekunden Segmente, die älter als retention_seconds sind."""
        while not self._stop_event.is_set():
            try:
                now = time.time()
                cutoff = now - self.retention_seconds
                deleted = 0

                if self.seg_dir.exists():
                    for file in self.seg_dir.glob("seg_*.mp4"):
                        try:
                            if file.stat().st_mtime < cutoff:
                                file.unlink()
                                deleted += 1
                        except OSError:
                            pass

                if deleted > 0:
                    logger.debug(f"{deleted} veraltete Video-Segmente (> {self.retention_seconds // 60} Min) gelöscht.")
            except Exception as e:
                logger.error(f"Fehler im Segment-Cleaner: {e}")

            self._stop_event.wait(timeout=30.0)
