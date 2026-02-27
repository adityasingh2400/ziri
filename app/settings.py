from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Aura Hub"
    app_env: str = "dev"
    aura_port: int = 8000
    log_level: str = "INFO"
    cors_allow_origins: str = "*"

    aws_region: str = "us-east-1"
    aws_access_key: Optional[str] = None
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    bedrock_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    polly_voice_id: str = "Joanna"
    polly_engine: str = "neural"
    s3_tts_bucket: Optional[str] = None
    s3_tts_public_base_url: Optional[str] = None

    supabase_url: Optional[str] = None
    supabase_service_role_key: Optional[str] = None

    spotify_client_id: Optional[str] = None
    spotify_client_secret: Optional[str] = None
    spotify_redirect_uri: str = "http://127.0.0.1:8000/spotify/callback"
    spotify_refresh_token: Optional[str] = None
    spotify_user_access_token: Optional[str] = None
    spotify_default_device_id: Optional[str] = None

    google_service_account_file: Optional[str] = None
    google_calendar_id: str = "primary"

    device_map_path: str = "app/config/device_map.yaml"
    scene_map_path: str = "app/config/scenes.yaml"
    return_audio_url: bool = True
    enable_polly: bool = True
    memory_window: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
