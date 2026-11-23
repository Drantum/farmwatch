import unittest
from app.vision_client import VisionClient


class TestVisionClient(unittest.TestCase):
    def test_disabled_by_default(self):
        client = VisionClient(enabled=False)
        result = client.extract_company_from_frame(b"dummy_image_data")
        self.assertIsNone(result)

    def test_clean_company_response(self):
        client = VisionClient(enabled=True)

        # Reiner Name
        self.assertEqual(client._clean_company_response('"Hoyer"'), "Hoyer")
        self.assertEqual(client._clean_company_response("Spedition Bischoff"), "Spedition Bischoff")

        # Unerwünschte Phrasen / Nicht erkannt
        self.assertIsNone(client._clean_company_response("Unbekannt"))
        self.assertIsNone(client._clean_company_response("Keine Beschriftung erkennbar."))
        self.assertIsNone(client._clean_company_response("Nicht lesbar"))
        self.assertIsNone(client._clean_company_response(""))

        # Ganze Sätze bereinigen
        self.assertEqual(
            client._clean_company_response("Die Spedition heißt Schenker."),
            "Schenker",
        )
        self.assertEqual(
            client._clean_company_response("Der Name lautet: Nagel Group"),
            "Nagel Group",
        )


if __name__ == "__main__":
    unittest.main()
