from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

import boto3
from botocore.client import BaseClient

from app.settings import Settings

logger = logging.getLogger(__name__)


class PollyTTS:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = settings.enable_polly

        kwargs = {"region_name": settings.aws_region}
        aws_access_key = settings.aws_access_key_id or settings.aws_access_key
        if aws_access_key and settings.aws_secret_access_key:
            kwargs["aws_access_key_id"] = aws_access_key
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

        self._polly: BaseClient | None = None
        self._s3: BaseClient | None = None

        if self.enabled:
            try:
                self._polly = boto3.client("polly", **kwargs)
                if settings.s3_tts_bucket:
                    self._s3 = boto3.client("s3", **kwargs)
            except Exception as exc:
                logger.warning("Polly initialization failed. Falling back to text-only response: %s", exc)
                self._polly = None
                self._s3 = None

    def synthesize(self, text: str) -> str | None:
        if not self.enabled or not text or not self._polly:
            return None

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

            if not self.settings.s3_tts_bucket or not self._s3:
                return None

            key = (
                f"aura-tts/{datetime.now(timezone.utc).strftime('%Y%m%d')}/"
                f"{datetime.now(timezone.utc).strftime('%H%M%S')}-{uuid4().hex}.mp3"
            )
            self._s3.put_object(
                Bucket=self.settings.s3_tts_bucket,
                Key=key,
                Body=audio_bytes,
                ContentType="audio/mpeg",
            )

            if self.settings.s3_tts_public_base_url:
                base = self.settings.s3_tts_public_base_url.rstrip("/")
                return f"{base}/{key}"

            return self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.settings.s3_tts_bucket, "Key": key},
                ExpiresIn=3600,
            )
        except Exception as exc:
            logger.warning("Polly synthesis failed: %s", exc)
            return None
