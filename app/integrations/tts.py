from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.settings import Settings

logger = logging.getLogger(__name__)

_AUDIO_DIR = Path(__file__).resolve().parent.parent / "static" / "audio"
_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


class TTS:
    """Text-to-speech with ElevenLabs primary, Polly fallback, local file serving."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._eleven_ok = bool(settings.elevenlabs_api_key)
        self._polly = None

        if not self._eleven_ok and settings.enable_polly:
            try:
                import boto3
                kwargs = {"region_name": settings.aws_region}
                ak = settings.aws_access_key_id or settings.aws_access_key
                if ak and settings.aws_secret_access_key:
                    kwargs["aws_access_key_id"] = ak
                    kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
                self._polly = boto3.client("polly", **kwargs)
            except Exception as exc:
                logger.warning("Polly init failed: %s", exc)

    def synthesize(self, text: str) -> str | None:
        if not text:
            return None

        if self._eleven_ok:
            url = self._elevenlabs_synthesize(text)
            if url:
                return url

        if self._polly:
            return self._polly_synthesize(text)

        return None

    def _elevenlabs_synthesize(self, text: str) -> str | None:
        try:
            import httpx

            resp = httpx.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.settings.elevenlabs_voice_id}?output_format=mp3_44100_128",
                headers={
                    "xi-api-key": self.settings.elevenlabs_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": self.settings.elevenlabs_model_id,
                    "voice_settings": {
                        "stability": 0.4,
                        "similarity_boost": 0.8,
                        "style": 0.15,
                        "speed": 1.2,
                    },
                },
                timeout=15,
            )

            if resp.status_code != 200:
                logger.warning("ElevenLabs TTS failed (status %s): %s", resp.status_code, resp.text[:200])
                return None

            filename = f"{uuid4().hex}.mp3"
            path = _AUDIO_DIR / filename
            path.write_bytes(resp.content)

            self._cleanup_old_files()

            return f"/static/audio/{filename}"
        except Exception as exc:
            logger.warning("ElevenLabs TTS error: %s", exc)
            return None

    def _polly_synthesize(self, text: str) -> str | None:
        try:
            speech = self._polly.synthesize_speech(
                Text=text,
                OutputFormat="mp3",
                VoiceId=self.settings.polly_voice_id,
                Engine=self.settings.polly_engine,
            )
            stream = speech.get("AudioStream")
            if not stream:
                return None
            audio_bytes = stream.read()

            filename = f"{uuid4().hex}.mp3"
            path = _AUDIO_DIR / filename
            path.write_bytes(audio_bytes)

            self._cleanup_old_files()

            return f"/static/audio/{filename}"
        except Exception as exc:
            logger.warning("Polly TTS error: %s", exc)
            return None

    @staticmethod
    def _cleanup_old_files(max_files: int = 50) -> None:
        files = sorted(_AUDIO_DIR.glob("*.mp3"), key=lambda f: f.stat().st_mtime)
        while len(files) > max_files:
            files.pop(0).unlink(missing_ok=True)


# Keep backward compat with existing imports
PollyTTS = TTS
