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
from app.hub import AuraHub
from app.integrations.spotify_auth import SpotifyAuthHelper
from app.schemas import IntentRequest, IntentResponse, StatusResponse

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

app = FastAPI(title=settings.app_name, version="0.1.0")
hub = AuraHub(settings=settings)
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


@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next) -> Response:
    req_id = str(uuid4())
    started = time.monotonic()
    response = await call_next(request)
    latency_ms = int((time.monotonic() - started) * 1000)
    response.headers["X-Request-Id"] = req_id
    response.headers["X-Process-Time-Ms"] = str(latency_ms)
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


# ---------------------------------------------------------------------------
# Browser voice client
# ---------------------------------------------------------------------------

_listen_html = _static_dir / "listen.html"


@app.get("/listen", response_class=HTMLResponse)
async def listen_page() -> HTMLResponse:
    if _listen_html.exists():
        return HTMLResponse(_listen_html.read_text(encoding="utf-8"))
    return HTMLResponse("<h3>listen.html not found in app/static/</h3>", status_code=404)
