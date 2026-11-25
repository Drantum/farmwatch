import logging
from pathlib import Path
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/events", tags=["events"])


@router.get("", response_model=List[Dict[str, Any]])
async def list_events(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Liefert die letzten Ereignisse (Standard: 100, neueste zuerst)."""
    db = request.app.state.db
    events = db.get_events(limit=limit, offset=offset)

    # Bild- und Videopfad für das Frontend mit /media/ URL anreichern
    for ev in events:
        if ev.get("snapshot_filename"):
            ev["snapshot_url"] = f"/media/{ev['snapshot_filename']}"
        else:
            ev["snapshot_url"] = None

        if ev.get("clip_filename"):
            ev["clip_url"] = f"/media/{ev['clip_filename']}"
        else:
            ev["clip_url"] = None

    return events


@router.get("/{event_id}", response_model=Dict[str, Any])
async def get_event(request: Request, event_id: int):
    """Liefert ein bestimmtes Ereignis anhand seiner ID."""
    db = request.app.state.db
    ev = db.get_event_by_id(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Ereignis nicht gefunden")

    if ev.get("snapshot_filename"):
        ev["snapshot_url"] = f"/media/{ev['snapshot_filename']}"
    if ev.get("clip_filename"):
        ev["clip_url"] = f"/media/{ev['clip_filename']}"

    return ev


@router.delete("/{event_id}")
async def delete_event(request: Request, event_id: int):
    """Löscht ein Ereignis und zugehörige Mediendateien."""
    db = request.app.state.db
    config = request.app.state.config
    ev = db.get_event_by_id(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail="Ereignis nicht gefunden")

    # Mediendateien löschen
    media_dir = Path(config.storage.media_dir)
    if ev.get("snapshot_filename"):
        snap_path = media_dir / ev["snapshot_filename"]
        if snap_path.exists():
            try:
                snap_path.unlink()
            except OSError as e:
                logger.warning(f"Konnte Snapshot nicht löschen: {e}")

    if ev.get("clip_filename"):
        clip_path = media_dir / ev["clip_filename"]
        if clip_path.exists():
            try:
                clip_path.unlink()
            except OSError as e:
                logger.warning(f"Konnte Clip nicht löschen: {e}")

    success = db.delete_event(event_id)
    return {"success": success, "deleted_id": event_id}
