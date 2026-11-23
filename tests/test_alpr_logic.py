import unittest
from app.alpr_logic import (
    clean_plate_text,
    is_valid_german_plate,
    format_german_plate,
    PlateVoteCollector,
)

try:
    import numpy as np
except ImportError:
    np = None


class DummyFrame:
    def __init__(self, val: int):
        self.val = val

    def copy(self):
        return DummyFrame(self.val)

    def __eq__(self, other):
        return isinstance(other, DummyFrame) and self.val == other.val


class TestALPRLogic(unittest.TestCase):
    def test_clean_plate_text(self):
        self.assertEqual(clean_plate_text("b-mw 1234"), "BMW1234")
        self.assertEqual(clean_plate_text(" M AB 123 E "), "MAB123E")
        self.assertEqual(clean_plate_text("wü-xy:99"), "WÜXY99")
        self.assertEqual(clean_plate_text(""), "")

    def test_is_valid_german_plate(self):
        valid_cases = [
            "BMW1234",     # B-MW 1234
            "MAB12",       # M-AB 12
            "HHXY9999",    # HH-XY 9999
            "WÜT1",        # WÜ-T 1
            "DAHXY12E",    # DAH-XY 12 E (Elektro)
            "ROLA123H",    # ROL-A 123 H (Oldtimer)
            "KLL1",        # K-LL 1
        ]
        for plate in valid_cases:
            self.assertTrue(is_valid_german_plate(plate), f"Sollte gültig sein: {plate}")

        invalid_cases = [
            "BMW0123",     # Ziffernteil beginnt mit 0
            "123BMW",      # Ziffern vor Buchstaben
            "B",           # Zu kurz
            "ABCDEFGHI",   # Zu lang (> 8 Zeichen)
            "B1234",       # Keine Erkennungsbuchstaben
            "B12AB",       # Ziffern vor Buchstaben
            "XYZW1234",    # 4 Buchstaben Ortskürzel gibt es in DE nicht
        ]
        for plate in invalid_cases:
            self.assertFalse(is_valid_german_plate(plate), f"Sollte ungültig sein: {plate}")

    def test_format_german_plate(self):
        self.assertEqual(format_german_plate("BMW1234"), "B-MW 1234")
        self.assertEqual(format_german_plate("MAB123E"), "M-AB 123 E")
        self.assertEqual(format_german_plate("HHXY999H"), "HH-XY 999 H")
        self.assertEqual(format_german_plate("BMW1234", hint_raw="B-MW 1234"), "B-MW 1234")

    def test_plate_vote_collector(self):
        collector = PlateVoteCollector(min_confidence=0.35)

        if np is not None:
            frame1 = np.ones((10, 10, 3), dtype=np.uint8) * 100
            frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 200
        else:
            frame1 = DummyFrame(100)
            frame2 = DummyFrame(200)

        # Erkennungen hinzufügen
        collector.add_detection("B-MW 1234", confidence=0.75, frame=frame1)
        collector.add_detection("BMW1234", confidence=0.92, frame=frame2)  # Höhere Confidence
        collector.add_detection("B-MW 1231", confidence=0.50, frame=frame1)
        collector.add_detection("BMW1234", confidence=0.20, frame=frame1)

        formatted, raw, conf, best_frame = collector.get_winner()

        self.assertEqual(raw, "BMW1234")
        self.assertEqual(formatted, "B-MW 1234")
        self.assertAlmostEqual(conf, 0.835, places=3)
        self.assertEqual(best_frame, frame2)


if __name__ == "__main__":
    unittest.main()
