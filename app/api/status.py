import shutil
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/status", tags=["status"])


@router.get("", response_model=Dict[str, Any])
async def get_system_status(request: Request):
    """Liefert Diagnose- und Gesundheitsdaten aller beteiligten Komponenten."""
    app = request.app
    config = app.state.config
    db = app.state.db
    recorder = getattr(app.state, "recorder", None)
    detector = getattr(app.state, "detector", None)
    event_manager = getattr(app.state, "event_manager", None)

    # Speicherplatz auf der Festplatte prüfen
    try:
        data_path = Path(config.storage.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        disk_usage = shutil.disk_usage(data_path)
        disk_free_gb = round(disk_usage.free / (1024**3), 2)
        disk_total_gb = round(disk_usage.total / (1024**3), 2)
    except Exception:
        disk_free_gb, disk_total_gb = 0.0, 0.0

    return {
        "status": "online",
        "events_count": db.count_events(),
        "recorder": recorder.get_status() if recorder else {"running": False},
        "detector": detector.get_status() if detector else {"running": False},
        "event_manager": event_manager.get_status() if event_manager else {"active_event": False},
        "vision": {
            "enabled": config.vision.enabled,
            "model": config.vision.model,
            "api_base_url": config.vision.api_base_url,
        },
        "storage": {
            "free_gb": disk_free_gb,
            "total_gb": disk_total_gb,
            "data_dir": str(config.storage.data_dir),
        },
    }
