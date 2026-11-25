from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional


class Database:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Kontextmanager, der die Verbindung nach Benutzung garantiert schließt."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Erstellt die Tabellen und Indizes, falls nicht vorhanden."""
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    plate TEXT,
                    plate_raw TEXT,
                    plate_confidence REAL,
                    company TEXT,
                    snapshot_filename TEXT NOT NULL,
                    clip_filename TEXT NOT NULL,
                    duration_seconds REAL NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_created_at
                ON events(created_at DESC);
                """
            )
            conn.commit()

    def insert_event(
        self,
        created_at: Optional[datetime] = None,
        plate: Optional[str] = None,
        plate_raw: Optional[str] = None,
        plate_confidence: Optional[float] = None,
        company: Optional[str] = None,
        snapshot_filename: str = "",
        clip_filename: str = "",
        duration_seconds: float = 0.0,
    ) -> int:
        """Fügt ein neues LKW-Ereignis in die Datenbank ein."""
        ts_str = (created_at or datetime.now()).isoformat()
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (
                    created_at, plate, plate_raw, plate_confidence,
                    company, snapshot_filename, clip_filename, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_str,
                    plate,
                    plate_raw,
                    plate_confidence,
                    company,
                    snapshot_filename,
                    clip_filename,
                    duration_seconds,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_events(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Gibt die neuesten Ereignisse zurück (Standard: letzte 100)."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, created_at, plate, plate_raw, plate_confidence,
                       company, snapshot_filename, clip_filename, duration_seconds
                FROM events
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_event_by_id(self, event_id: int) -> Optional[Dict[str, Any]]:
        """Gibt ein bestimmtes Ereignis anhand der ID zurück."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, created_at, plate, plate_raw, plate_confidence,
                       company, snapshot_filename, clip_filename, duration_seconds
                FROM events
                WHERE id = ?
                """,
                (event_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_event(self, event_id: int) -> bool:
        """Löscht einen Eintrag aus der Datenbank."""
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            conn.commit()
            return cursor.rowcount > 0

    def count_events(self) -> int:
        """Gibt die Gesamtzahl der gespeicherten Ereignisse zurück."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM events")
            return cursor.fetchone()[0]
