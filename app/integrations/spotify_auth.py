from __future__ import annotations

import logging
from typing import Any

from app.settings import Settings

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except Exception:
    spotipy = None  # type: ignore[assignment]
    SpotifyOAuth = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

SCOPES = (
    "user-read-playback-state "
    "user-modify-playback-state "
    "user-read-currently-playing "
    "user-library-modify "
    "user-library-read "
    "user-read-recently-played "
    "playlist-read-private"
)


class SpotifyAuthHelper:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _build_oauth(self) -> Any | None:
        if not SpotifyOAuth:
            return None
        if not self.settings.spotify_client_id or not self.settings.spotify_client_secret:
            return None
        return SpotifyOAuth(
            client_id=self.settings.spotify_client_id,
            client_secret=self.settings.spotify_client_secret,
            redirect_uri=self.settings.spotify_redirect_uri,
            scope=SCOPES,
            open_browser=False,
            cache_path=None,
        )

    def get_authorize_url(self) -> str | None:
        oauth = self._build_oauth()
        if not oauth:
            return None
        return oauth.get_authorize_url()

    def exchange_code(self, code: str) -> dict[str, Any] | None:
        oauth = self._build_oauth()
        if not oauth:
            return None
        try:
            return oauth.get_access_token(code, as_dict=True, check_cache=False)
        except Exception as exc:
            logger.exception("Spotify token exchange failed")
            return None

    def list_devices(self) -> list[dict[str, Any]]:
        client = self._get_authed_client()
        if not client:
            return []
        try:
            payload = client.devices()
            return payload.get("devices", []) if isinstance(payload, dict) else []
        except Exception:
            logger.exception("Failed to list Spotify devices")
            return []

    def _get_authed_client(self) -> Any | None:
        if not spotipy:
            return None
        if (
            self.settings.spotify_refresh_token
            and self.settings.spotify_client_id
            and self.settings.spotify_client_secret
        ):
            oauth = self._build_oauth()
            if not oauth:
                return None
            try:
                token_info = oauth.refresh_access_token(self.settings.spotify_refresh_token)
                return spotipy.Spotify(auth=token_info["access_token"])
            except Exception:
                logger.exception("Could not refresh token")
        if self.settings.spotify_user_access_token:
            return spotipy.Spotify(auth=self.settings.spotify_user_access_token)
        return None

    def get_now_playing(self) -> dict[str, Any] | None:
        client = self._get_authed_client()
        if not client:
            return None
        try:
            playback = client.current_playback()
            if not playback or not playback.get("item"):
                return None

            item = playback["item"]
            images = item.get("album", {}).get("images", [])
            album_art = images[0]["url"] if images else None

            return {
                "track": item.get("name", ""),
                "artist": ", ".join(a.get("name", "") for a in item.get("artists", [])),
                "album": item.get("album", {}).get("name", ""),
                "album_art": album_art,
                "is_playing": playback.get("is_playing", False),
                "progress_ms": playback.get("progress_ms", 0),
                "duration_ms": item.get("duration_ms", 0),
                "shuffle_state": playback.get("shuffle_state", False),
                "repeat_state": playback.get("repeat_state", "off"),
                "spotify_uri": item.get("uri"),
            }
        except Exception:
            logger.exception("Failed to get now playing")
            return None
