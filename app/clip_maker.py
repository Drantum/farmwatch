import logging
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_segment_timestamp(file_path: Path) -> Optional[float]:
    """
    Extrahiert den Start-Zeitstempel eines Segments aus dem Dateinamen
    (z.B. seg_20261121_083000.mp4) oder nutzt mtime als Fallback.
    """
    stem = file_path.stem
    # Format: seg_YYYYMMDD_HHMMSS
    if stem.startswith("seg_"):
        raw_ts = stem[4:]
        try:
            dt = datetime.strptime(raw_ts, "%Y%m%d_%H%M%S")
            return dt.timestamp()
        except ValueError:
            pass

    # Fallback auf Datei-Änderungszeitpunkt minus geschätzte Segmentdauer
    try:
        return file_path.stat().st_mtime - 10.0
    except OSError:
        return None


def find_matching_segments(
    seg_dir: Path,
    start_time: float,
    end_time: float,
    segment_duration: float = 10.0,
    pre_buffer: float = 3.0,
    post_buffer: float = 2.0,
) -> List[Path]:
    """
    Findet alle Segmente in seg_dir, die das Zeitfenster
    [start_time - pre_buffer, end_time + post_buffer] abdecken.
    """
    target_start = start_time - pre_buffer
    target_end = end_time + post_buffer

    if not seg_dir.exists():
        return []

    segment_files = list(seg_dir.glob("seg_*.mp4"))
    matched: List[Tuple[float, Path]] = []

    for f in segment_files:
        seg_start = parse_segment_timestamp(f)
        if seg_start is None:
            continue
        seg_end = seg_start + segment_duration

        # Überschneidung zweier Intervalle prüfen
        if seg_start <= target_end and seg_end >= target_start:
            matched.append((seg_start, f))

    # Chronologisch sortieren
    matched.sort(key=lambda x: x[0])
    return [path for _, path in matched]


def build_event_clip(
    seg_dir: Path,
    output_path: Path,
    start_time: float,
    end_time: float,
    segment_duration: float = 10.0,
    pre_buffer: float = 3.0,
    post_buffer: float = 2.0,
) -> bool:
    """
    Kopiert und fügt die relevanten Segmente verlustfrei mittels ffmpeg concat zusammen.
    Gibt True bei Erfolg zurück.
    """
    matching_segments = find_matching_segments(
        seg_dir=seg_dir,
        start_time=start_time,
        end_time=end_time,
        segment_duration=segment_duration,
        pre_buffer=pre_buffer,
        post_buffer=post_buffer,
    )

    if not matching_segments:
        logger.warning(
            f"Keine passenden Segmente für Zeitraum {start_time:.1f} bis {end_time:.1f} in {seg_dir} gefunden."
        )
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Concat-Demuxer Datei temporär anlegen
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        temp_list_path = Path(f.name)
        for seg in matching_segments:
            # ffmpeg concat erfordert vorwärtsgerichtete Slashes oder Escaping
            clean_path = str(seg.resolve()).replace("\\", "/")
            f.write(f"file '{clean_path}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(temp_list_path),
            "-c", "copy",
            str(output_path),
        ]
        logger.info(f"Führe ffmpeg Concat für {len(matching_segments)} Segmente aus...")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(f"ffmpeg Concat fehlgeschlagen: {result.stderr}")
            return False

        logger.info(f"Clip erfolgreich erstellt: {output_path}")
        return True
    except FileNotFoundError:
        logger.error("ffmpeg Befehl nicht gefunden. Bitte ffmpeg auf dem Host oder im Container installieren.")
        return False
    except Exception as e:
        logger.error(f"Fehler beim Erstellen des Videoclips: {e}")
        return False
    finally:
        if temp_list_path.exists():
            temp_list_path.unlink()
