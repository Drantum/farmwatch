import unittest
import tempfile
import time
from pathlib import Path
from app.database import Database
from app.event_manager import EventManager
from app.vision_client import VisionClient


class DummyFrame:
    def __init__(self, val: int = 1):
        self.val = val

    def copy(self):
        return DummyFrame(self.val)


class TestEventManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.db = Database(base / "events.db")
        self.media_dir = base / "media"
        self.seg_dir = base / "seg"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.seg_dir.mkdir(parents=True, exist_ok=True)

        self.vision_client = VisionClient(enabled=False)

        # Schnelle Timeouts für Unit-Tests
        self.mgr = EventManager(
            db=self.db,
            media_dir=self.media_dir,
            seg_dir=self.seg_dir,
            vision_client=self.vision_client,
            idle_timeout=1.0,
            segment_flush_wait=0.2,
            pre_buffer=1.0,
            post_buffer=1.0,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_event_lifecycle(self):
        self.assertFalse(self.mgr.is_active)

        # 1. Frame: LKW erkannt
        t0 = time.time()
        self.mgr.on_frame_analyzed(
            truck_detected=True,
            truck_conf=0.85,
            plates=[{"text": "B-MW 1234", "confidence": 0.88}],
            frame=DummyFrame(1),
            timestamp=t0,
        )
        self.assertTrue(self.mgr.is_active)
        self.assertEqual(self.mgr.start_time, t0)

        # 2. Frame: LKW weiterhin im Bild mit zweiter Kennzeichen-Messung
        t1 = t0 + 0.5
        self.mgr.on_frame_analyzed(
            truck_detected=True,
            truck_conf=0.90,
            plates=[{"text": "BMW1234", "confidence": 0.95}],
            frame=DummyFrame(2),
            timestamp=t1,
        )
        self.assertTrue(self.mgr.is_active)
        self.assertEqual(self.mgr.last_seen_time, t1)

        # 3. Frame: LKW weg, aber noch innerhalb des 1.0s Idle-Timeouts
        t2 = t1 + 0.5
        self.mgr.on_frame_analyzed(
            truck_detected=False,
            truck_conf=0.0,
            plates=[],
            frame=None,
            timestamp=t2,
        )
        self.assertTrue(self.mgr.is_active)

        # 4. Frame: LKW weg, mehr als 1.0s verstrichen -> Trigger Finalization
        t3 = t1 + 1.2
        self.mgr.on_frame_analyzed(
            truck_detected=False,
            truck_conf=0.0,
            plates=[],
            frame=None,
            timestamp=t3,
        )
        self.assertFalse(self.mgr.is_active)

        # Warte kurz auf den Finalize-Worker (0.2s Karenz + Puffer)
        time.sleep(0.5)

        # Prüfe, ob Eintrag in der Datenbank angelegt wurde
        events = self.db.get_events(limit=10)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["plate"], "B-MW 1234")
        self.assertEqual(ev["plate_raw"], "BMW1234")
        self.assertAlmostEqual(ev["duration_seconds"], 0.5, places=1)


if __name__ == "__main__":
    unittest.main()
