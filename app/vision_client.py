import base64
import logging
from pathlib import Path
from typing import Any, Optional, Union

try:
    import httpx
except ImportError:
    httpx = None

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class VisionClient:
    """
    Optionaler Client zur Erkennung der Speditionsaufschrift auf LKWs
    über eine beliebige OpenAI-kompatible Vision-API (z.B. OpenAI GPT-4o-mini,
    OpenRouter, vLLM, Ollama mit OpenAI-Adapter etc.).
    """

    def __init__(
        self,
        enabled: bool = False,
        api_base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 15.0,
        prompt: Optional[str] = None,
    ):
        self.enabled = enabled
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.prompt = prompt or (
            "Identifiziere den Namen der Spedition oder des Transportunternehmens auf diesem Tank-LKW "
            "(z.B. Aufschrift auf dem Führerhaus oder Tank). Antworte ausschließlich mit dem Firmennamen "
            "oder 'Unbekannt', falls keine Firmenaufschrift lesbar ist."
        )

    def extract_company_from_frame(
        self,
        image_or_path: Union[Any, bytes, Path, str],
    ) -> Optional[str]:
        """
        Analysiert das übergebene Bild (Frame oder Dateipfad) und extrahiert
        den Namen der Spedition. Gibt None zurück, wenn deaktiviert oder nicht erkennbar.
        """
        if not self.enabled:
            return None

        if httpx is None:
            logger.warning("httpx ist nicht installiert. Vision API kann nicht aufgerufen werden.")
            return None

        try:
            if isinstance(image_or_path, (str, Path)):
                with open(image_or_path, "rb") as f:
                    jpeg_bytes = f.read()
            elif isinstance(image_or_path, bytes):
                jpeg_bytes = image_or_path
            else:
                if cv2 is None:
                    logger.warning("cv2 ist nicht installiert, kann Bildarray nicht konvertieren.")
                    return None
                success, encoded = cv2.imencode(".jpg", image_or_path, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not success:
                    logger.warning("Konnte Frame nicht als JPEG kodieren.")
                    return None
                jpeg_bytes = encoded.tobytes()

            base64_img = base64.b64encode(jpeg_bytes).decode("utf-8")
            return self._call_vision_api(base64_img)

        except Exception as e:
            logger.error(f"Fehler bei der Speditionserkennung: {e}")
            return None

    def _call_vision_api(self, base64_img: str) -> Optional[str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_img}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 60,
            "temperature": 0.1,
        }

        url = f"{self.api_base_url}/chat/completions"

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                logger.warning(
                    f"Vision API Aufruf fehlgeschlagen ({response.status_code}): {response.text}"
                )
                return None

            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            return self._clean_company_response(content)

    def _clean_company_response(self, text: str) -> Optional[str]:
        if not text:
            return None

        cleaned = text.strip().strip('"').strip("'").strip(".")
        lower = cleaned.lower()

        if any(
            marker in lower
            for marker in [
                "unbekannt",
                "keine",
                "nicht erkennbar",
                "nicht lesbar",
                "unknown",
                "n/a",
                "none",
            ]
        ):
            return None

        if "heißt" in lower:
            parts = cleaned.split("heißt")
            cleaned = parts[-1].strip().strip(":").strip()
        elif "lautet" in lower:
            parts = cleaned.split("lautet")
            cleaned = parts[-1].strip().strip(":").strip()

        return cleaned if len(cleaned) <= 60 else cleaned[:60]
