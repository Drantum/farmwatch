import os
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    data_dir: Path = Path("./data")
    db_path: Path = Path("./data/events.db")
    media_dir: Path = Path("./data/media")
    seg_dir: Path = Path("./data/seg")


class StreamConfig(BaseModel):
    rtsp_url: str = "rtsp://go2rtc:8554/lkw"
    go2rtc_host: str = "go2rtc"
    go2rtc_http_port: int = 1984
    go2rtc_rtsp_port: int = 8554


class RecorderConfig(BaseModel):
    segment_time: int = 10
    retention_minutes: int = 10
    pre_buffer_seconds: int = 3
    post_buffer_seconds: int = 2


class DetectorConfig(BaseModel):
    target_fps: float = 5.0
    yolo_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.45
    device: str = "cpu"
    target_classes: List[str] = Field(default_factory=lambda: ["truck"])


class ALPRConfig(BaseModel):
    detector_model: str = "yolo-v9-t-384"
    ocr_model: str = "cct-s-v2-global-model"
    min_confidence: float = 0.35
    format: str = "DE"


class EventConfig(BaseModel):
    idle_timeout_seconds: float = 8.0
    segment_flush_wait_seconds: float = 12.0


class VisionConfig(BaseModel):
    enabled: bool = False
    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 15.0
    prompt: str = (
        "Identifiziere den Namen der Spedition oder des Transportunternehmens auf diesem Tank-LKW "
        "(z.B. Aufschrift auf dem Führerhaus oder Tank). Antworte ausschließlich mit dem Firmennamen "
        "oder 'Unbekannt', falls keine Firmenaufschrift lesbar ist."
    )


class AppConfig(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    stream: StreamConfig = Field(default_factory=StreamConfig)
    recorder: RecorderConfig = Field(default_factory=RecorderConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    alpr: ALPRConfig = Field(default_factory=ALPRConfig)
    event: EventConfig = Field(default_factory=EventConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Lädt die Konfiguration aus YAML und überschreibt mit Umgebungsvariablen."""
    path = Path(config_path or os.getenv("CONFIG_PATH", "config.yaml"))
    data = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    config = AppConfig(**data)

    # Umgebungsvariablen überschreiben
    if os.getenv("DATA_DIR"):
        config.storage.data_dir = Path(os.getenv("DATA_DIR"))
        config.storage.db_path = config.storage.data_dir / "events.db"
        config.storage.media_dir = config.storage.data_dir / "media"
        config.storage.seg_dir = config.storage.data_dir / "seg"

    if os.getenv("RTSP_URL"):
        config.stream.rtsp_url = os.getenv("RTSP_URL")
    elif os.getenv("GO2RTC_HOST") and os.getenv("GO2RTC_RTSP_PORT"):
        host = os.getenv("GO2RTC_HOST")
        port = os.getenv("GO2RTC_RTSP_PORT")
        config.stream.rtsp_url = f"rtsp://{host}:{port}/lkw"

    if os.getenv("GO2RTC_HOST"):
        config.stream.go2rtc_host = os.getenv("GO2RTC_HOST")
    if os.getenv("GO2RTC_HTTP_PORT"):
        config.stream.go2rtc_http_port = int(os.getenv("GO2RTC_HTTP_PORT"))
    if os.getenv("GO2RTC_RTSP_PORT"):
        config.stream.go2rtc_rtsp_port = int(os.getenv("GO2RTC_RTSP_PORT"))

    # Vision API Umgebungsvariablen
    if os.getenv("VISION_ENABLED") is not None:
        config.vision.enabled = os.getenv("VISION_ENABLED").strip().lower() in ("true", "1", "yes")
    if os.getenv("VISION_API_BASE_URL"):
        config.vision.api_base_url = os.getenv("VISION_API_BASE_URL")
    if os.getenv("VISION_API_KEY"):
        config.vision.api_key = os.getenv("VISION_API_KEY")
    if os.getenv("VISION_MODEL"):
        config.vision.model = os.getenv("VISION_MODEL")

    # Verzeichnisse anlegen
    config.storage.data_dir.mkdir(parents=True, exist_ok=True)
    config.storage.media_dir.mkdir(parents=True, exist_ok=True)
    config.storage.seg_dir.mkdir(parents=True, exist_ok=True)

    return config
