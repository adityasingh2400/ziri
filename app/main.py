from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import Response

from app.settings import get_settings
from app.hub import ZiriHub
from app.integrations.spotify_auth import SpotifyAuthHelper
from app.schemas import IntentRequest, IntentResponse, StatusResponse
from app.core.metrics import REQUEST_DURATION, PROMETHEUS_AVAILABLE

from datetime import datetime, timezone

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

app = FastAPI(title=settings.app_name, version="0.1.0")
hub = ZiriHub(settings=settings)
spotify_auth = SpotifyAuthHelper(settings=settings)

_static_dir = Path(__file__).resolve().parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

allowed_origins = [v.strip() for v in settings.cors_allow_origins.split(",") if v.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if PROMETHEUS_AVAILABLE:
    from app.core.metrics import make_asgi_app as _make_metrics_app
    _metrics_app = _make_metrics_app()
    app.mount("/metrics/", _metrics_app)

    @app.get("/metrics")
    async def _metrics_redirect():
        from starlette.responses import RedirectResponse as _RR
        return _RR("/metrics/", status_code=301)


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next) -> Response:
    req_id = str(uuid4())
    started = time.monotonic()
    response = await call_next(request)
    latency_ms = int((time.monotonic() - started) * 1000)
    response.headers["X-Request-Id"] = req_id
    response.headers["X-Process-Time-Ms"] = str(latency_ms)
    endpoint = request.url.path
    REQUEST_DURATION.labels(
        method=request.method, endpoint=endpoint, status=response.status_code,
    ).observe(latency_ms / 1000)
    return response


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": settings.app_name, "status_endpoint": "/status", "intent_endpoint": "/intent"}


@app.post("/intent", response_model=IntentResponse)
async def post_intent(payload: IntentRequest) -> IntentResponse:
    return await hub.handle_intent(payload)


@app.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    return hub.status()


# ---------------------------------------------------------------------------
# Spotify OAuth flow
# ---------------------------------------------------------------------------


@app.get("/spotify/auth")
async def spotify_auth_redirect() -> Response:
    url = spotify_auth.get_authorize_url()
    if not url:
        return HTMLResponse(
            "<h3>Spotify not configured</h3>"
            "<p>Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env, then restart.</p>",
            status_code=400,
        )
    return RedirectResponse(url)


@app.get("/spotify/callback")
async def spotify_callback(code: str = Query(...)) -> HTMLResponse:
    token_info = spotify_auth.exchange_code(code)
    if not token_info:
        return HTMLResponse("<h3>Token exchange failed.</h3><p>Check server logs.</p>", status_code=400)

    refresh_token = token_info.get("refresh_token", "")
    access_token = token_info.get("access_token", "")
    return HTMLResponse(
        "<h2>Spotify connected!</h2>"
        f"<p><b>Refresh token</b> (paste into .env as SPOTIFY_REFRESH_TOKEN):</p>"
        f"<pre style='background:#111;color:#0f0;padding:12px;border-radius:6px;'>{refresh_token}</pre>"
        f"<p><b>Access token</b> (temporary, for testing):</p>"
        f"<pre style='background:#111;color:#0f0;padding:12px;border-radius:6px;'>{access_token}</pre>"
        f"<p>Next: <a href='/spotify/devices'>List your Spotify devices</a></p>"
    )


@app.get("/spotify/devices")
async def spotify_devices() -> dict[str, Any]:
    devices = spotify_auth.list_devices()
    return {"devices": devices}


@app.get("/spotify/now-playing")
async def spotify_now_playing() -> dict[str, Any]:
    data = spotify_auth.get_now_playing()
    if not data:
        return {"playing": False}
    return {"playing": True, **data}


@app.get("/spotify/art-proxy")
async def spotify_art_proxy(url: str = Query("")) -> Response:
    """Proxy Spotify album art to avoid CORS issues with canvas color extraction."""
    if not url or "scdn.co" not in url and "spotify" not in url:
        return Response(status_code=400, content=b"bad url")
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=8)
    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "image/jpeg"),
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.post("/spotify/duck")
async def spotify_duck() -> dict[str, Any]:
    """Lower Spotify volume while Ziri is listening/thinking/speaking."""
    return hub.orchestrator.tool_runner.spotify.duck()


@app.post("/spotify/unduck")
async def spotify_unduck() -> dict[str, Any]:
    """Restore Spotify volume after Ziri finishes responding."""
    return hub.orchestrator.tool_runner.spotify.unduck()


@app.post("/spotify/play")
async def spotify_play() -> dict[str, Any]:
    result = hub.orchestrator.tool_runner.spotify.resume()
    return {"ok": result.ok, "error": result.error}


@app.post("/spotify/pause")
async def spotify_pause() -> dict[str, Any]:
    result = hub.orchestrator.tool_runner.spotify.pause()
    return {"ok": result.ok, "error": result.error}


@app.post("/spotify/next")
async def spotify_next() -> dict[str, Any]:
    result = hub.orchestrator.tool_runner.spotify.skip_next()
    return {"ok": result.ok, "error": result.error}


@app.post("/spotify/prev")
async def spotify_prev() -> dict[str, Any]:
    result = hub.orchestrator.tool_runner.spotify.previous_track()
    return {"ok": result.ok, "error": result.error}


@app.get("/debug/connections")
async def debug_connections() -> dict[str, Any]:
    """Quick status of all connected services for the debug panel."""
    s = settings
    eleven_ok = bool(s.elevenlabs_api_key)
    spotify_ok = bool(s.spotify_refresh_token and s.spotify_client_id)
    bedrock_ok = hub.brain._bedrock is not None

    devices = []
    try:
        devices = spotify_auth.list_devices()
    except Exception:
        pass

    return {
        "spotify": {"connected": spotify_ok, "devices": devices},
        "bedrock": {"connected": bedrock_ok, "model": s.bedrock_model_id if bedrock_ok else None},
        "tts": {"engine": "elevenlabs" if eleven_ok else ("polly" if s.enable_polly else "browser"), "voice_id": s.elevenlabs_voice_id if eleven_ok else s.polly_voice_id},
        "memory": hub.memory_store.__class__.__name__,
    }


# ---------------------------------------------------------------------------
# Browser voice client
# ---------------------------------------------------------------------------

_listen_html = _static_dir / "listen.html"


@app.get("/listen", response_class=HTMLResponse)
async def listen_page() -> HTMLResponse:
    if _listen_html.exists():
        return HTMLResponse(_listen_html.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>listen.html not found in app/static/</h3>", status_code=404)


# ---------------------------------------------------------------------------
# Siri Shortcuts endpoint
# ---------------------------------------------------------------------------


@app.post("/siri")
async def siri_intent(text: str = Query(..., min_length=1, max_length=3000)) -> dict[str, Any]:
    """Simplified endpoint for Siri Shortcuts.

    Usage from Shortcuts app:
        POST http://<mac-ip>:8000/siri?text=play+some+jazz
    Returns JSON with speak_text and optional audio_url.
    """
    request = IntentRequest(
        user_id=settings.siri_user_id,
        device_id=settings.siri_device_id,
        room=settings.siri_room,
        raw_text=text,
        timestamp=datetime.now(timezone.utc),
    )
    response = await hub.handle_intent(request)
    return {
        "speak_text": response.speak_text,
        "private_note": response.private_note,
        "action_code": response.action_code,
        "audio_url": response.audio_url,
    }


# ---------------------------------------------------------------------------
# Listener dashboard
# ---------------------------------------------------------------------------

_listener_ref: Any = None
_vision_ref: Any = None


def set_listener(listener: Any) -> None:
    """Called by run_listener.py to make the listener accessible to the dashboard."""
    global _listener_ref
    _listener_ref = listener


def set_vision(vision: Any) -> None:
    """Called by run_listener.py to make the gesture recognizer accessible."""
    global _vision_ref
    _vision_ref = vision


@app.get("/dashboard/api")
async def dashboard_api() -> dict[str, Any]:
    """JSON API for the listener dashboard."""
    listener_status = _listener_ref.get_status() if _listener_ref else {"running": False, "state": "not_started", "history": []}
    connections = {}
    try:
        s = settings
        eleven_ok = bool(s.elevenlabs_api_key)
        spotify_ok = bool(s.spotify_refresh_token and s.spotify_client_id)
        bedrock_ok = hub.brain._bedrock is not None
        connections = {
            "spotify": spotify_ok,
            "bedrock": bedrock_ok,
            "tts": "elevenlabs" if eleven_ok else ("polly" if s.enable_polly else "none"),
        }
    except Exception:
        pass
    return {"listener": listener_status, "connections": connections}


_dashboard_html = _static_dir / "dashboard.html"


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page() -> HTMLResponse:
    """Redirect to /listen which serves as the combined dashboard."""
    return HTMLResponse('<script>location.href="/listen"</script>')


# ---------------------------------------------------------------------------
# Vision gesture status + MJPEG stream
# ---------------------------------------------------------------------------


@app.get("/vision/status")
async def vision_status() -> dict[str, Any]:
    """Status and recent gesture history from the vision module."""
    if _vision_ref is None:
        return {"running": False, "active": False, "gesture_count": 0, "history": []}
    return _vision_ref.get_status()


@app.get("/vision/feed")
async def vision_feed() -> Response:
    """MJPEG stream of the camera with landmark overlay."""
    if _vision_ref is None:
        return Response(status_code=503, content=b"Vision module not running")

    import asyncio as _aio

    async def _generate():
        while True:
            frame = _vision_ref.get_jpeg_frame()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            await _aio.sleep(0.04)

    from starlette.responses import StreamingResponse
    return StreamingResponse(
        _generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
