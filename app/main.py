from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import events, status, cows
from app.config import load_config
from app.cow_detector import CowDetector
from app.database import Database
from app.detector import StreamDetector
from app.event_manager import EventManager
from app.recorder import StreamRecorder
from app.vision_client import VisionClient

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisiert und beendet alle Hintergrund-Dienste sauber."""
    logger.info("Starte LKW-Erkennungs- und Hofkamera-System...")
    config = load_config()
    app.state.config = config

    # 1. Datenbank initialisieren
    db = Database(config.storage.db_path)
    app.state.db = db

    # 2. Vision Client (OpenAI-kompatibel)
    vision_client = VisionClient(
        enabled=config.vision.enabled,
        api_base_url=config.vision.api_base_url,
        api_key=config.vision.api_key,
        model=config.vision.model,
        timeout_seconds=config.vision.timeout_seconds,
        prompt=config.vision.prompt,
    )
    app.state.vision_client = vision_client

    # 3. Event Manager
    event_manager = EventManager(
        db=db,
        media_dir=config.storage.media_dir,
        seg_dir=config.storage.seg_dir,
        vision_client=vision_client,
        idle_timeout=config.event.idle_timeout_seconds,
        segment_flush_wait=config.event.segment_flush_wait_seconds,
        pre_buffer=config.recorder.pre_buffer_seconds,
        post_buffer=config.recorder.post_buffer_seconds,
        video_codec=getattr(config.recorder, "video_codec", "copy"),
        alpr_min_confidence=config.alpr.min_confidence,
    )
    app.state.event_manager = event_manager

    # 4. ffmpeg-Segment Recorder (Hintergrund-Supervisor)
    try:
        recorder = StreamRecorder(
            rtsp_url=config.stream.rtsp_url,
            seg_dir=config.storage.seg_dir,
            segment_time=config.recorder.segment_time,
            retention_minutes=config.recorder.retention_minutes,
        )
        recorder.start()
        app.state.recorder = recorder
    except Exception as e:
        logger.error(f"Fehler beim Starten des Recorders: {e}")
        app.state.recorder = None

    # 5. Detektor & ALPR
    try:
        detector = StreamDetector(
            rtsp_url=config.stream.rtsp_url,
            event_manager=event_manager,
            target_fps=config.detector.target_fps,
            yolo_model_name=config.detector.yolo_model,
            confidence_threshold=config.detector.confidence_threshold,
            device=config.detector.device,
            target_classes=config.detector.target_classes,
            alpr_detector_model=config.alpr.detector_model,
            alpr_ocr_model=config.alpr.ocr_model,
        )
        detector.start()
        app.state.detector = detector
    except Exception as e:
        logger.error(f"Fehler beim Starten des Detektors: {e}")
        app.state.detector = None

    # 6. Kuh-Detektor für Abkalbebox (COLO / Eigene Modelle & Bounding-Boxes)
    if getattr(config, "cow_detector", None) and config.cow_detector.enabled:
        try:
            cow_detector = CowDetector(
                rtsp_url=config.cow_detector.rtsp_url,
                model_path=config.cow_detector.model_path,
                confidence_threshold=config.cow_detector.confidence_threshold,
                target_fps=config.cow_detector.target_fps,
                device=config.cow_detector.device,
                track=config.cow_detector.track,
                target_classes=config.cow_detector.target_classes,
                models_dir=config.cow_detector.models_dir,
            )
            cow_detector.start()
            app.state.cow_detector = cow_detector
        except Exception as e:
            logger.error(f"Fehler beim Starten des Kuh-Detektors: {e}")
            app.state.cow_detector = None
    else:
        app.state.cow_detector = None

    logger.info("System bereit und aktiv.")
    yield

    # Shutdown
    logger.info("Fahre System herunter...")
    if getattr(app.state, "cow_detector", None):
        app.state.cow_detector.stop()
    if getattr(app.state, "detector", None):
        app.state.detector.stop()
    if getattr(app.state, "recorder", None):
        app.state.recorder.stop()
    logger.info("System beendet.")


app = FastAPI(
    title="LKW-Erkennung & Hofkameras",
    version="1.0.0",
    lifespan=lifespan,
)

# Statische Verzeichnisse mounten
BASE_DIR = Path(__file__).resolve().parent

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Media-Ordner (/media/) für Video-Clips und Snapshots (unterstützt HTTP 206 Partial Content / Range Requests)
cfg = load_config()
cfg.storage.media_dir.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(cfg.storage.media_dir)), name="media")

# API Router einbinden
app.include_router(events.router)
app.include_router(status.router)
app.include_router(cows.router)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Liefert das Haupt-Dashboard mit Live-Ansicht und Ereignissen."""
    index_file = BASE_DIR / "templates" / "index.html"
    return FileResponse(str(index_file), media_type="text/html")
