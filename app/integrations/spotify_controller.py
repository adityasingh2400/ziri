from __future__ import annotations

import logging
from typing import Any

from app.settings import Settings
from app.schemas import ToolResult

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
except Exception:  # pragma: no cover - optional import for local dev
    spotipy = None  # type: ignore[assignment]
    SpotifyClientCredentials = None  # type: ignore[assignment]
    SpotifyOAuth = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class SpotifyController:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._cached_client: Any = None
        self._cached_can_control: bool = False
        self._cache_expires: float = 0

    def _build_client(self) -> tuple[Any, bool]:
        import time
        now = time.time()
        if self._cached_client and now < self._cache_expires:
            return self._cached_client, self._cached_can_control

        if not spotipy:
            return None, False

        client = None
        can_control = False

        if (
            self.settings.spotify_refresh_token
            and self.settings.spotify_client_id
            and self.settings.spotify_client_secret
            and SpotifyOAuth
        ):
            try:
                oauth = SpotifyOAuth(
                    client_id=self.settings.spotify_client_id,
                    client_secret=self.settings.spotify_client_secret,
                    redirect_uri=self.settings.spotify_redirect_uri,
                    scope=(
                        "user-read-playback-state "
                        "user-modify-playback-state "
                        "user-read-currently-playing "
                        "user-library-modify "
                        "user-library-read "
                        "user-read-recently-played "
                        "playlist-read-private"
                    ),
                    open_browser=False,
                    cache_path=None,
                )
                token_info = oauth.refresh_access_token(self.settings.spotify_refresh_token)
                access_token = token_info.get("access_token")
                if access_token:
                    client = spotipy.Spotify(auth=access_token)
                    can_control = True
            except Exception as exc:
                logger.warning("Spotify refresh-token flow failed, falling back: %s", exc)

        if not client and self.settings.spotify_user_access_token:
            client = spotipy.Spotify(auth=self.settings.spotify_user_access_token)
            can_control = True

        if not client and self.settings.spotify_client_id and self.settings.spotify_client_secret:
            creds = SpotifyClientCredentials(
                client_id=self.settings.spotify_client_id,
                client_secret=self.settings.spotify_client_secret,
            )
            client = spotipy.Spotify(auth_manager=creds)
            can_control = False

        if client:
            self._cached_client = client
            self._cached_can_control = can_control
            self._cache_expires = now + 3000  # ~50 minutes

        return client, can_control

    def _resolve_target_device(
        self,
        client: Any,
        explicit_device_id: str | None,
        speaker_name: str | None,
    ) -> str | None:
        if explicit_device_id:
            return explicit_device_id

        try:
            payload = client.devices()
            devices = payload.get("devices", []) if isinstance(payload, dict) else []
        except Exception:
            return self.settings.spotify_default_device_id

        if speaker_name:
            lowered = speaker_name.lower()
            for device in devices:
                name = str(device.get("name", "")).lower()
                if lowered in name or name in lowered:
                    return device.get("id")

        for device in devices:
            if device.get("is_active"):
                return device.get("id")

        return self.settings.spotify_default_device_id

    def _wake_device(self, client: Any, device_id: str | None) -> bool:
        """Transfer playback to a device to wake it from dormant state."""
        if not device_id:
            return False
        try:
            client.transfer_playback(device_id, force_play=False)
            import time
            time.sleep(0.3)
            return True
        except Exception as exc:
            logger.debug("Could not wake device %s: %s", device_id, exc)
            return False

    def _ensure_active_device(self, client: Any, explicit_device_id: str | None, speaker_name: str | None) -> str | None:
        """Get a device ID and make sure it's awake."""
        target = self._resolve_target_device(client, explicit_device_id, speaker_name)

        try:
            payload = client.devices()
            devices = payload.get("devices", []) if isinstance(payload, dict) else []
            has_active = any(d.get("is_active") for d in devices)

            if has_active:
                return target

            wake_id = target
            if not wake_id and devices:
                wake_id = devices[0].get("id")
            if wake_id:
                self._wake_device(client, wake_id)
                return wake_id
        except Exception:
            pass

        return target

    @staticmethod
    def _build_search_query(raw_query: str) -> str:
        """Parse natural language into Spotify field-filtered search.

        'money longer by Uzi'  -> 'track:money longer artist:Uzi'
        'XO Tour Life Lil Uzi' -> 'XO Tour Life Lil Uzi' (no 'by', pass as-is)
        """
        lowered = raw_query.lower()

        by_separators = [" by ", " from "]
        for sep in by_separators:
            idx = lowered.rfind(sep)
            if idx > 0:
                track_part = raw_query[:idx].strip()
                artist_part = raw_query[idx + len(sep):].strip()
                if track_part and artist_part:
                    return f"track:{track_part} artist:{artist_part}"

        dash_idx = raw_query.find(" - ")
        if dash_idx > 0:
            track_part = raw_query[:dash_idx].strip()
            artist_part = raw_query[dash_idx + 3:].strip()
            if track_part and artist_part:
                return f"track:{track_part} artist:{artist_part}"

        return raw_query

    @staticmethod
    def _pick_best_track(tracks: list[dict], query: str) -> dict:
        """Score tracks against the original query and return the best match.

        Scoring: +3 if track name appears in query, +3 if any artist name
        appears in query, +1 for popularity/10. Ties broken by Spotify's
        default relevance order.
        """
        if len(tracks) == 1:
            return tracks[0]

        q_lower = query.lower()

        # Extract artist hint if "by" is present
        artist_hint = ""
        for sep in [" by ", " from "]:
            idx = q_lower.rfind(sep)
            if idx > 0:
                artist_hint = q_lower[idx + len(sep):].strip()
                break

        best_track = tracks[0]
        best_score = -1

        for track in tracks:
            score = 0
            track_name = track.get("name", "").lower()
            track_artists = [a.get("name", "").lower() for a in track.get("artists", [])]

            if track_name in q_lower or q_lower in track_name:
                score += 3

            for ta in track_artists:
                if artist_hint and (artist_hint in ta or ta in artist_hint):
                    score += 4
                elif ta in q_lower:
                    score += 2

            score += track.get("popularity", 0) / 100

            if score > best_score:
                best_score = score
                best_track = track

        return best_track

    def play_query(
        self,
        query: str,
        speaker_name: str | None = None,
        spotify_device_id: str | None = None,
    ) -> ToolResult:
        client, can_control_playback = self._build_client()
        if not client:
            return ToolResult(
                ok=False,
                action_code="MUSIC_ERROR",
                speak_text="",
                private_note="Spotify is not configured yet. Add credentials in .env and retry.",
                error="spotify_not_configured",
            )

        try:
            structured_q = self._build_search_query(query)
            search = client.search(q=structured_q, type="track", limit=5)
            tracks = search.get("tracks", {}).get("items", [])

            if not tracks and structured_q != query:
                search = client.search(q=query, type="track", limit=5)
                tracks = search.get("tracks", {}).get("items", [])

            if not tracks:
                search = client.search(q=query, type="artist", limit=1)
                artists = search.get("artists", {}).get("items", [])
                if artists:
                    artist = artists[0]
                    return ToolResult(
                        ok=True,
                        action_code="MUSIC_RESULT",
                        speak_text=f"I found {artist.get('name', 'that artist')} on Spotify.",
                        private_note="Open the Spotify URL from the response metadata to start playback.",
                        payload={
                            "spotify_url": artist.get("external_urls", {}).get("spotify"),
                            "artist_id": artist.get("id"),
                        },
                    )
                return ToolResult(
                    ok=False,
                    action_code="MUSIC_NOT_FOUND",
                    speak_text="I could not find that on Spotify.",
                    private_note="Try a more specific song or artist name.",
                    error="not_found",
                )

            top_track = self._pick_best_track(tracks, query)
            uri = top_track["uri"]
            title = top_track.get("name", "that track")
            artist_name = ", ".join(a.get("name", "") for a in top_track.get("artists", []))
            external_url = top_track.get("external_urls", {}).get("spotify")

            if can_control_playback:
                target_device = self._ensure_active_device(
                    client=client,
                    explicit_device_id=spotify_device_id,
                    speaker_name=speaker_name,
                )
                if target_device:
                    client.start_playback(device_id=target_device, uris=[uri])
                else:
                    client.start_playback(uris=[uri])
                return ToolResult(
                    ok=True,
                    action_code="MUSIC_START",
                    speak_text=f"Playing {title} by {artist_name}.",
                    private_note="",
                    payload={
                        "spotify_url": external_url,
                        "spotify_uri": uri,
                        "title": title,
                        "artist": artist_name,
                        "spotify_device_id": target_device,
                    },
                )

            return ToolResult(
                ok=True,
                action_code="MUSIC_RESULT",
                speak_text="",
                private_note="I found a Spotify result. Use the provided URL to open playback on your phone.",
                payload={"spotify_url": external_url, "spotify_uri": uri, "title": title},
            )

        except Exception as exc:
            logger.exception("Spotify playback failed")
            return ToolResult(
                ok=False,
                action_code="MUSIC_ERROR",
                speak_text="I had trouble talking to Spotify.",
                private_note=str(exc),
                error="spotify_exception",
            )

    def adjust_volume(
        self,
        delta_percent: int,
        speaker_name: str | None = None,
        spotify_device_id: str | None = None,
    ) -> ToolResult:
        client, can_control_playback = self._build_client()
        if not client or not can_control_playback:
            return ToolResult(
                ok=False,
                action_code="MUSIC_ERROR",
                speak_text="",
                private_note="Volume control requires Spotify playback scopes and user auth.",
                error="missing_user_token",
            )

        try:
            playback = client.current_playback() or {}
            current_volume = (
                playback.get("device", {}).get("volume_percent") if playback.get("device") else None
            )
            if current_volume is None:
                current_volume = 50
            new_volume = max(0, min(100, int(current_volume) + int(delta_percent)))
            target_device = self._ensure_active_device(
                client=client,
                explicit_device_id=spotify_device_id,
                speaker_name=speaker_name,
            )
            if target_device:
                client.volume(new_volume, device_id=target_device)
            else:
                client.volume(new_volume)

            direction = "up" if delta_percent >= 0 else "down"
            return ToolResult(
                ok=True,
                action_code="MUSIC_VOLUME",
                speak_text=f"Volume {direction} to {new_volume} percent.",
                private_note="",
                payload={"volume_percent": new_volume},
            )
        except Exception as exc:
            logger.exception("Spotify volume adjustment failed")
            return ToolResult(
                ok=False,
                action_code="MUSIC_ERROR",
                speak_text="I could not change Spotify volume.",
                private_note=str(exc),
                error="spotify_volume_exception",
            )

    def pause(
        self,
        speaker_name: str | None = None,
        spotify_device_id: str | None = None,
    ) -> ToolResult:
        client, can_control_playback = self._build_client()
        if not client or not can_control_playback:
            return ToolResult(
                ok=False,
                action_code="MUSIC_ERROR",
                speak_text="",
                private_note="Pause requires Spotify playback scopes and user auth.",
                error="missing_user_token",
            )

        try:
            target_device = self._ensure_active_device(
                client=client,
                explicit_device_id=spotify_device_id,
                speaker_name=speaker_name,
            )
            if target_device:
                client.pause_playback(device_id=target_device)
            else:
                client.pause_playback()

            return ToolResult(
                ok=True,
                action_code="MUSIC_PAUSE",
                speak_text="Paused.",
                private_note="",
            )
        except Exception as exc:
            logger.exception("Spotify pause failed")
            return ToolResult(
                ok=False,
                action_code="MUSIC_ERROR",
                speak_text="I could not pause Spotify.",
                private_note=str(exc),
                error="spotify_pause_exception",
            )

    def skip_next(
        self,
        speaker_name: str | None = None,
        spotify_device_id: str | None = None,
    ) -> ToolResult:
        client, can_control_playback = self._build_client()
        if not client or not can_control_playback:
            return ToolResult(
                ok=False,
                action_code="MUSIC_ERROR",
                speak_text="",
                private_note="Skip requires Spotify playback scopes and user auth.",
                error="missing_user_token",
            )

        try:
            target_device = self._ensure_active_device(
                client=client,
                explicit_device_id=spotify_device_id,
                speaker_name=speaker_name,
            )
            if target_device:
                client.next_track(device_id=target_device)
            else:
                client.next_track()

            return ToolResult(
                ok=True,
                action_code="MUSIC_SKIP",
                speak_text="Skipped to the next track.",
                private_note="",
            )
        except Exception as exc:
            logger.exception("Spotify skip failed")
            return ToolResult(
                ok=False,
                action_code="MUSIC_ERROR",
                speak_text="I could not skip the track.",
                private_note=str(exc),
                error="spotify_skip_exception",
            )

    def get_currently_playing(self) -> ToolResult:
        client, can_control_playback = self._build_client()
        if not client or not can_control_playback:
            return ToolResult(
                ok=False,
                action_code="MUSIC_ERROR",
                speak_text="",
                private_note="Currently-playing requires Spotify user auth.",
                error="missing_user_token",
            )

        try:
            playback = client.current_playback()
            if not playback or not playback.get("item"):
                return ToolResult(
                    ok=True,
                    action_code="MUSIC_INFO",
                    speak_text="Nothing is playing right now.",
                    private_note="",
                )

            item = playback["item"]
            title = item.get("name", "Unknown track")
            artist_name = ", ".join(a.get("name", "") for a in item.get("artists", []))
            album = item.get("album", {}).get("name", "")
            is_playing = playback.get("is_playing", False)
            status = "Playing" if is_playing else "Paused"

            return ToolResult(
                ok=True,
                action_code="MUSIC_INFO",
                speak_text=f"{status}: {title} by {artist_name}.",
                private_note="",
                payload={
                    "title": title,
                    "artist": artist_name,
                    "album": album,
                    "is_playing": is_playing,
                    "spotify_uri": item.get("uri"),
                    "spotify_url": item.get("external_urls", {}).get("spotify"),
                },
            )
        except Exception as exc:
            logger.exception("Spotify current-track lookup failed")
            return ToolResult(
                ok=False,
                action_code="MUSIC_ERROR",
                speak_text="I could not check what is playing.",
                private_note=str(exc),
                error="spotify_current_exception",
            )

    # ------------------------------------------------------------------
    # New commands: resume, previous, shuffle, repeat, set_volume,
    #               add_to_queue, play_artist, play_playlist, like
    # ------------------------------------------------------------------

    def _require_playback_client(self, action_label: str) -> tuple[Any, str | None]:
        """Return (client, error_result). If error_result is not None, return it."""
        client, can_control = self._build_client()
        if not client or not can_control:
            return None, "missing_user_token"
        return client, None

    def resume(
        self,
        speaker_name: str | None = None,
        spotify_device_id: str | None = None,
    ) -> ToolResult:
        client, err = self._require_playback_client("resume")
        if err:
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="", private_note="Resume requires user auth.", error=err)

        try:
            target = self._ensure_active_device(client, spotify_device_id, speaker_name)
            if target:
                client.start_playback(device_id=target)
            else:
                client.start_playback()
            return ToolResult(ok=True, action_code="MUSIC_RESUME", speak_text="Resumed.", private_note="")
        except Exception as exc:
            logger.exception("Spotify resume failed")
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="I could not resume playback.", private_note=str(exc), error="spotify_resume_exception")

    def previous_track(
        self,
        speaker_name: str | None = None,
        spotify_device_id: str | None = None,
    ) -> ToolResult:
        client, err = self._require_playback_client("previous")
        if err:
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="", private_note="Previous requires user auth.", error=err)

        try:
            target = self._ensure_active_device(client, spotify_device_id, speaker_name)
            if target:
                client.previous_track(device_id=target)
            else:
                client.previous_track()
            return ToolResult(ok=True, action_code="MUSIC_PREVIOUS", speak_text="Playing the previous track.", private_note="")
        except Exception as exc:
            logger.exception("Spotify previous failed")
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="I could not go back.", private_note=str(exc), error="spotify_previous_exception")

    def shuffle(self, state: bool) -> ToolResult:
        client, err = self._require_playback_client("shuffle")
        if err:
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="", private_note="Shuffle requires user auth.", error=err)

        try:
            client.shuffle(state)
            label = "on" if state else "off"
            return ToolResult(ok=True, action_code="MUSIC_SHUFFLE", speak_text=f"Shuffle {label}.", private_note="")
        except Exception as exc:
            logger.exception("Spotify shuffle failed")
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="I could not change shuffle.", private_note=str(exc), error="spotify_shuffle_exception")

    def repeat(self, mode: str) -> ToolResult:
        client, err = self._require_playback_client("repeat")
        if err:
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="", private_note="Repeat requires user auth.", error=err)

        valid_modes = {"off", "track", "context"}
        if mode not in valid_modes:
            mode = "off"
        try:
            client.repeat(mode)
            label = {"off": "off", "track": "this track", "context": "this playlist"}.get(mode, mode)
            return ToolResult(ok=True, action_code="MUSIC_REPEAT", speak_text=f"Repeat set to {label}.", private_note="")
        except Exception as exc:
            logger.exception("Spotify repeat failed")
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="I could not change repeat.", private_note=str(exc), error="spotify_repeat_exception")

    def set_volume(
        self,
        percent: int,
        speaker_name: str | None = None,
        spotify_device_id: str | None = None,
    ) -> ToolResult:
        client, err = self._require_playback_client("set_volume")
        if err:
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="", private_note="Volume requires user auth.", error=err)

        percent = max(0, min(100, int(percent)))
        try:
            target = self._ensure_active_device(client, spotify_device_id, speaker_name)
            if target:
                client.volume(percent, device_id=target)
            else:
                client.volume(percent)
            return ToolResult(ok=True, action_code="MUSIC_VOLUME", speak_text=f"Volume set to {percent} percent.", private_note="", payload={"volume_percent": percent})
        except Exception as exc:
            logger.exception("Spotify set_volume failed")
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="I could not set the volume.", private_note=str(exc), error="spotify_set_volume_exception")

    # ------------------------------------------------------------------
    # Audio ducking – lower volume while Ziri is listening/thinking/speaking
    # ------------------------------------------------------------------

    _pre_duck_volume: int | None = None

    def duck(self, duck_percent: int = 25) -> dict[str, Any]:
        """Lower Spotify volume for voice interaction. Returns the pre-duck volume."""
        client, can_control = self._build_client()
        if not client or not can_control:
            return {"ok": False, "error": "no_client"}

        try:
            playback = client.current_playback() or {}
            device = playback.get("device") or {}
            current_vol = device.get("volume_percent")
            if current_vol is None:
                return {"ok": False, "error": "no_playback"}

            if current_vol <= duck_percent:
                self._pre_duck_volume = current_vol
                return {"ok": True, "previous_volume": current_vol, "ducked_to": current_vol, "skipped": True}

            self._pre_duck_volume = current_vol
            target_vol = max(0, min(100, duck_percent))
            client.volume(target_vol)
            logger.info("Ducked Spotify volume %d → %d", current_vol, target_vol)
            return {"ok": True, "previous_volume": current_vol, "ducked_to": target_vol}
        except Exception as exc:
            logger.exception("Spotify duck failed")
            return {"ok": False, "error": str(exc)}

    def unduck(self) -> dict[str, Any]:
        """Gradually restore Spotify volume to the pre-duck level."""
        import time as _time

        client, can_control = self._build_client()
        if not client or not can_control:
            return {"ok": False, "error": "no_client"}

        restore_vol = self._pre_duck_volume
        if restore_vol is None:
            return {"ok": False, "error": "not_ducked"}

        try:
            playback = client.current_playback() or {}
            current_vol = (playback.get("device") or {}).get("volume_percent", 25)

            steps = 4
            diff = restore_vol - current_vol
            if diff <= 0:
                self._pre_duck_volume = None
                return {"ok": True, "restored_volume": restore_vol}

            for i in range(1, steps + 1):
                step_vol = current_vol + int(diff * (i / steps))
                client.volume(min(step_vol, 100))
                if i < steps:
                    _time.sleep(0.3)

            logger.info("Unducked Spotify volume %d → %d (ramped)", current_vol, restore_vol)
            self._pre_duck_volume = None
            return {"ok": True, "restored_volume": restore_vol}
        except Exception as exc:
            logger.exception("Spotify unduck failed")
            return {"ok": False, "error": str(exc)}

    def add_to_queue(
        self,
        query: str,
        speaker_name: str | None = None,
        spotify_device_id: str | None = None,
    ) -> ToolResult:
        client, err = self._require_playback_client("queue")
        if err:
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="", private_note="Queue requires user auth.", error=err)

        try:
            search = client.search(q=query, type="track", limit=1)
            tracks = search.get("tracks", {}).get("items", [])
            if not tracks:
                return ToolResult(ok=False, action_code="MUSIC_NOT_FOUND", speak_text=f"I could not find {query} to queue.", private_note="", error="not_found")

            track = tracks[0]
            uri = track["uri"]
            title = track.get("name", "that track")
            artist_name = ", ".join(a.get("name", "") for a in track.get("artists", []))

            target = self._ensure_active_device(client, spotify_device_id, speaker_name)
            if target:
                client.add_to_queue(uri, device_id=target)
            else:
                client.add_to_queue(uri)

            return ToolResult(ok=True, action_code="MUSIC_QUEUE", speak_text=f"Added {title} by {artist_name} to the queue.", private_note="", payload={"title": title, "artist": artist_name, "spotify_uri": uri})
        except Exception as exc:
            logger.exception("Spotify queue failed")
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="I could not add that to the queue.", private_note=str(exc), error="spotify_queue_exception")

    def play_artist(
        self,
        query: str,
        speaker_name: str | None = None,
        spotify_device_id: str | None = None,
        shuffle: bool = False,
    ) -> ToolResult:
        client, can_control = self._build_client()
        if not client:
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="", private_note="Spotify not configured.", error="spotify_not_configured")

        try:
            search = client.search(q=f"artist:{query}", type="artist", limit=5)
            artists = search.get("artists", {}).get("items", [])
            if not artists:
                search = client.search(q=query, type="artist", limit=5)
                artists = search.get("artists", {}).get("items", [])
            if not artists:
                return ToolResult(ok=False, action_code="MUSIC_NOT_FOUND", speak_text=f"I could not find the artist {query}.", private_note="", error="not_found")

            q_lower = query.lower()
            best = artists[0]
            for a in artists:
                if a.get("name", "").lower() == q_lower:
                    best = a
                    break
                if q_lower in a.get("name", "").lower():
                    best = a
                    break

            artist = best
            artist_uri = artist["uri"]
            artist_name = artist.get("name", query)

            if can_control:
                if shuffle:
                    client.shuffle(True)
                target = self._ensure_active_device(client, spotify_device_id, speaker_name)
                if target:
                    client.start_playback(device_id=target, context_uri=artist_uri)
                else:
                    client.start_playback(context_uri=artist_uri)
                label = "Shuffling" if shuffle else "Playing"
                return ToolResult(ok=True, action_code="MUSIC_START", speak_text=f"{label} {artist_name}.", private_note="", payload={"artist": artist_name, "spotify_uri": artist_uri})

            return ToolResult(ok=True, action_code="MUSIC_RESULT", speak_text=f"I found {artist_name} on Spotify.", private_note="Open Spotify to start playback.", payload={"artist": artist_name, "spotify_uri": artist_uri})
        except Exception as exc:
            logger.exception("Spotify play_artist failed")
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="I had trouble playing that artist.", private_note=str(exc), error="spotify_artist_exception")

    def play_playlist(
        self,
        query: str,
        speaker_name: str | None = None,
        spotify_device_id: str | None = None,
        shuffle: bool = False,
    ) -> ToolResult:
        client, can_control = self._build_client()
        if not client:
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="", private_note="Spotify not configured.", error="spotify_not_configured")

        try:
            playlist_uri = None
            playlist_name = query

            try:
                user_playlists = client.current_user_playlists(limit=50)
                items = user_playlists.get("items", []) if user_playlists else []
                q_lower = query.lower()
                for pl in items:
                    pl_name = pl.get("name", "")
                    if pl_name.lower() == q_lower:
                        playlist_uri = pl["uri"]
                        playlist_name = pl_name
                        break
                if not playlist_uri:
                    for pl in items:
                        pl_name = pl.get("name", "")
                        if q_lower in pl_name.lower() or pl_name.lower() in q_lower:
                            playlist_uri = pl["uri"]
                            playlist_name = pl_name
                            break
            except Exception:
                pass

            if not playlist_uri:
                search = client.search(q=query, type="playlist", limit=1)
                playlists = search.get("playlists", {}).get("items", [])
                if not playlists:
                    return ToolResult(ok=False, action_code="MUSIC_NOT_FOUND", speak_text=f"I could not find a playlist called {query}.", private_note="", error="not_found")
                playlist = playlists[0]
                playlist_uri = playlist["uri"]
                playlist_name = playlist.get("name", query)

            if can_control:
                if shuffle:
                    client.shuffle(True)
                target = self._ensure_active_device(client, spotify_device_id, speaker_name)
                if target:
                    client.start_playback(device_id=target, context_uri=playlist_uri)
                else:
                    client.start_playback(context_uri=playlist_uri)
                place = ""
                label = "Shuffling" if shuffle else "Playing"
                return ToolResult(ok=True, action_code="MUSIC_START", speak_text=f"{label} {playlist_name}.", private_note="", payload={"playlist": playlist_name, "spotify_uri": playlist_uri})

            return ToolResult(ok=True, action_code="MUSIC_RESULT", speak_text=f"I found the playlist {playlist_name}.", private_note="Open Spotify to start playback.", payload={"playlist": playlist_name, "spotify_uri": playlist_uri})
        except Exception as exc:
            logger.exception("Spotify play_playlist failed")
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="I had trouble playing that playlist.", private_note=str(exc), error="spotify_playlist_exception")

    def like_current_track(self) -> ToolResult:
        client, err = self._require_playback_client("like")
        if err:
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="", private_note="Like requires user auth.", error=err)

        try:
            playback = client.current_playback()
            if not playback or not playback.get("item"):
                return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="Nothing is playing to like.", private_note="", error="nothing_playing")

            item = playback["item"]
            track_id = item.get("id")
            title = item.get("name", "this track")
            artist_name = ", ".join(a.get("name", "") for a in item.get("artists", []))

            if not track_id:
                return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="I could not identify the current track.", private_note="", error="no_track_id")

            client.current_user_saved_tracks_add([track_id])
            return ToolResult(ok=True, action_code="MUSIC_LIKE", speak_text=f"Liked {title} by {artist_name}.", private_note="", payload={"title": title, "artist": artist_name})
        except Exception as exc:
            logger.exception("Spotify like failed")
            return ToolResult(ok=False, action_code="MUSIC_ERROR", speak_text="I could not like that track.", private_note=str(exc), error="spotify_like_exception")
