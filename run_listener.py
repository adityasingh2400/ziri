#!/usr/bin/env python3
"""Start the Ziri FastAPI server and always-on wake word listener together.

Usage:
    python3 run_listener.py          # default: host 0.0.0.0, port from .env
    python3 run_listener.py --no-listener   # server only, no wake word
"""

from __future__ import annotations

import os
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import argparse
import asyncio
import logging
import signal
import sys

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Ziri: always-on voice agent")
    parser.add_argument("--no-listener", action="store_true", help="Start server without the wake word listener")
    parser.add_argument("--no-vision", action="store_true", help="Start server without the gesture vision module")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: from .env ZIRI_PORT)")
    args = parser.parse_args()

    from app.settings import get_settings
    settings = get_settings()
    port = args.port or settings.ziri_port

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("ziri.runner")

    # Suppress noisy polling endpoints from uvicorn access log
    _QUIET_PATHS = {"/spotify/now-playing", "/dashboard/api", "/debug/connections",
                    "/spotify/art-proxy", "/favicon.ico", "/vision/status", "/vision/feed"}

    class _QuietAccessFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not any(p in msg for p in _QUIET_PATHS)

    logging.getLogger("uvicorn.access").addFilter(_QuietAccessFilter())

    if args.no_listener:
        logger.info("Starting server only (no listener) on %s:%d", args.host, port)
        uvicorn.run("app.main:app", host=args.host, port=port, log_level=settings.log_level.lower())
        return

    # Run both: uvicorn server + wake word listener sharing one event loop
    logger.info("Starting Ziri server + listener on %s:%d", args.host, port)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Import after settings are loaded
    from app.main import app, hub
    from app.core.listener import Listener

    listener = Listener(settings=settings, hub=hub)

    from app.main import set_listener
    set_listener(listener)

    vision = None
    if not args.no_vision:
        try:
            from app.core.vision import GestureRecognizer
            from app.main import set_vision
            vision = GestureRecognizer(hub=hub, settings=settings)
            set_vision(vision)
            logger.info("Vision gesture module enabled")
        except Exception as exc:
            logger.warning("Vision module unavailable (camera/mediapipe): %s", exc)

    config = uvicorn.Config(
        app,
        host=args.host,
        port=port,
        log_level=settings.log_level.lower(),
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    _shutting_down = False

    def _shutdown(sig, frame):
        nonlocal _shutting_down
        if _shutting_down:
            logger.info("Force quit")
            os._exit(1)
        _shutting_down = True
        logger.info("Shutting down... (Ctrl+C again to force quit)")
        listener.stop()
        if vision:
            vision.stop()
        server.should_exit = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    async def _run():
        listener.start(loop=asyncio.get_running_loop())
        if vision:
            vision.start(loop=asyncio.get_running_loop())
        await server.serve()
        if vision:
            vision.stop()
        listener.stop()

    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
