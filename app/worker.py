"""Lightweight FastAPI service for the LangGraph worker.

The API container forwards intent-processing requests here over the
internal Docker/Kubernetes network.  This keeps the heavy LangGraph
orchestration in its own container.
"""
from __future__ import annotations

import logging

from fastapi import FastAPI

from app.settings import get_settings
from app.hub import ZiriHub
from app.schemas import IntentRequest, IntentResponse

settings = get_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

worker_app = FastAPI(title="Ziri Worker", version="0.1.0")
hub = ZiriHub(settings=settings)


@worker_app.post("/process", response_model=IntentResponse)
async def process_intent(payload: IntentRequest) -> IntentResponse:
    return await hub.handle_intent(payload)


@worker_app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
