import asyncio
import logging
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cows", tags=["cows"])


@router.get("/live")
async def get_live_annotated_feed(request: Request):
    """
    Liefert einen MJPEG-Videostream der Abkalbebox mit eingezeichneten Bounding-Boxes,
    Kuh-Zählung und Tracking-IDs.
    Direkt kompatibel mit Standard-HTML <img> Tags im Browser.
    """
    cow_detector = getattr(request.app.state, "cow_detector", None)
    if not cow_detector:
        raise HTTPException(status_code=503, detail="Kuh-Detektor ist nicht aktiv.")

    async def generate():
        """Async-Wrapper um den synchronen MJPEG-Generator, damit der Event-Loop nicht blockiert."""
        while cow_detector.is_running():
            frame_bytes = cow_detector.get_latest_jpeg()
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            await asyncio.sleep(1.0 / max(1.0, cow_detector.target_fps))

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/snapshot")
async def get_snapshot(request: Request):
    """Liefert ein einzelnes JPEG-Snapshot der annotierten Ansicht."""
    cow_detector = getattr(request.app.state, "cow_detector", None)
    if not cow_detector:
        raise HTTPException(status_code=503, detail="Kuh-Detektor ist nicht aktiv.")

    jpeg = cow_detector.get_latest_jpeg()
    if not jpeg:
        raise HTTPException(status_code=503, detail="Noch kein Frame verfügbar.")

    from fastapi.responses import Response
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/status")
async def get_cows_status(request: Request):
    """Liefert den aktuellen Status der Kuh-Erkennung und der geladenen Modelle."""
    cow_detector = getattr(request.app.state, "cow_detector", None)
    if not cow_detector:
        return {
            "running": False,
            "connected": False,
            "cows_detected": 0,
            "active_tracks": [],
            "fps_measured": 0.0,
            "model_path": None,
            "is_custom_model": False,
            "model_loaded": False,
            "model_error": "CowDetector nicht initialisiert",
            "available_models": [],
        }
    return cow_detector.get_status()


@router.get("/models")
async def get_available_models(request: Request):
    """Listet alle im models/-Ordner abgelegten Modelle auf."""
    from app.cow_detector import list_available_models
    config = getattr(request.app.state, "config", None)
    models_dir = getattr(config.cow_detector, "models_dir", "./models") if config else "./models"
    return {
        "models": list_available_models(models_dir),
    }
