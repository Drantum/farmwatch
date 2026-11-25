import unittest
import tempfile
from pathlib import Path
from datetime import datetime
from app.database import Database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_events.db"
        self.db = Database(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_insert_and_get_events(self):
        self.assertEqual(self.db.count_events(), 0)

        # 1. Event einfügen
        dt1 = datetime(2026, 11, 21, 8, 15, 0)
        id1 = self.db.insert_event(
            created_at=dt1,
            plate="B-MW 1234",
            plate_raw="BMW1234",
            plate_confidence=0.92,
            company="Hoyer Spedition",
            snapshot_filename="snapshot_01.jpg",
            clip_filename="clip_01.mp4",
            duration_seconds=45.5,
        )
        self.assertEqual(id1, 1)

        # 2. Zweites Event einfügen
        dt2 = datetime(2026, 11, 21, 8, 30, 0)
        id2 = self.db.insert_event(
            created_at=dt2,
            plate="M-AB 123 E",
            plate_raw="MAB123E",
            plate_confidence=0.88,
            company=None,
            snapshot_filename="snapshot_02.jpg",
            clip_filename="clip_02.mp4",
            duration_seconds=20.0,
        )
        self.assertEqual(id2, 2)
        self.assertEqual(self.db.count_events(), 2)

        # Events abrufen (neueste zuerst)
        events = self.db.get_events(limit=10)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["id"], id2)
        self.assertEqual(events[0]["plate"], "M-AB 123 E")
        self.assertEqual(events[1]["id"], id1)
        self.assertEqual(events[1]["plate"], "B-MW 1234")
        self.assertEqual(events[1]["company"], "Hoyer Spedition")

        # Einzelnes Event abrufen
        single = self.db.get_event_by_id(id1)
        self.assertIsNotNone(single)
        self.assertEqual(single["plate_raw"], "BMW1234")
        self.assertEqual(single["duration_seconds"], 45.5)

        # Event löschen
        deleted = self.db.delete_event(id1)
        self.assertTrue(deleted)
        self.assertEqual(self.db.count_events(), 1)
        self.assertIsNone(self.db.get_event_by_id(id1))


if __name__ == "__main__":
    unittest.main()
