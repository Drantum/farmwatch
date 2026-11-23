import unittest
import tempfile
from pathlib import Path
from datetime import datetime
from app.clip_maker import parse_segment_timestamp, find_matching_segments


class TestClipMaker(unittest.TestCase):
    def test_parse_segment_timestamp(self):
        # seg_20261121_083000.mp4
        expected_dt = datetime(2026, 11, 21, 8, 30, 0)
        file_path = Path("seg_20261121_083000.mp4")
        parsed = parse_segment_timestamp(file_path)
        self.assertEqual(parsed, expected_dt.timestamp())

    def test_find_matching_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_path = Path(tmpdir)
            base_dt = datetime(2026, 11, 21, 10, 0, 0)
            base_ts = base_dt.timestamp()

            # Erstelle 6 Dummy-Segmente à 10 Sekunden:
            # 10:00:00 (0-10s), 10:00:10 (10-20s), 10:00:20 (20-30s), 10:00:30 (30-40s), 10:00:40 (40-50s), 10:00:50 (50-60s)
            created_files = []
            for i in range(6):
                seg_time = datetime.fromtimestamp(base_ts + i * 10)
                name = f"seg_{seg_time.strftime('%Y%m%d_%H%M%S')}.mp4"
                file_path = dir_path / name
                file_path.write_text("dummy")
                created_files.append(file_path)

            # Ein Ereignis dauerte von Sekunde 15 bis Sekunde 32
            # Event start = base_ts + 15, Event end = base_ts + 32
            # Mit Vorlauf (3s): Startfenster = base_ts + 12
            # Mit Nachlauf (2s): Endfenster = base_ts + 34
            # Passende Segmente:
            # Seg 0 (0-10s): endet vor 12 -> Nein
            # Seg 1 (10-20s): überlappt 12..34 -> Ja
            # Seg 2 (20-30s): überlappt 12..34 -> Ja
            # Seg 3 (30-40s): überlappt 12..34 -> Ja (da Start 30 <= 34)
            # Seg 4 (40-50s): startet bei 40 > 34 -> Nein

            matched = find_matching_segments(
                seg_dir=dir_path,
                start_time=base_ts + 15,
                end_time=base_ts + 32,
                segment_duration=10.0,
                pre_buffer=3.0,
                post_buffer=2.0,
            )

            matched_names = [f.name for f in matched]
            expected_names = [created_files[1].name, created_files[2].name, created_files[3].name]
            self.assertEqual(matched_names, expected_names)


if __name__ == "__main__":
    unittest.main()
