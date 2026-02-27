from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover
    Langfuse = None  # type: ignore[assignment,misc]

from app.settings import Settings

logger = logging.getLogger(__name__)

_langfuse_instance: Langfuse | None = None
_initialised = False


def get_langfuse(settings: Settings | None = None) -> Langfuse | None:
    global _langfuse_instance, _initialised
    if _initialised:
        return _langfuse_instance
    _initialised = True

    if Langfuse is None:
        logger.debug("langfuse package not installed; tracing disabled")
        return None

    if settings is None:
        from app.settings import get_settings
        settings = get_settings()

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.debug("Langfuse keys not configured; tracing disabled")
        return None

    try:
        _langfuse_instance = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse tracing initialised (host=%s)", settings.langfuse_host)
    except Exception as exc:
        logger.warning("Langfuse init failed: %s", exc)
    return _langfuse_instance


def create_trace(
    *,
    name: str,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    settings: Settings | None = None,
) -> Any | None:
    """Create a top-level Langfuse trace. Returns the trace object or None."""
    lf = get_langfuse(settings)
    if lf is None:
        return None
    try:
        return lf.trace(
            name=name,
            user_id=user_id,
            metadata=metadata or {},
            tags=tags or [],
        )
    except Exception as exc:
        logger.debug("Failed to create Langfuse trace: %s", exc)
        return None


def trace_llm_call(
    *,
    trace: Any | None,
    name: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    bedrock_call: Any,
    tool_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a Bedrock converse() call, wrapping it with a Langfuse generation span.

    Returns the raw Bedrock response dict.
    """
    start = time.perf_counter()
    response = bedrock_call()
    elapsed_ms = (time.perf_counter() - start) * 1000

    if trace is None:
        return response

    try:
        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

        content_blocks = response.get("output", {}).get("message", {}).get("content", [])
        output_text = ""
        for block in content_blocks:
            if "text" in block:
                output_text += block["text"]
            elif "toolUse" in block:
                import json
                output_text += json.dumps(block["toolUse"])

        trace.generation(
            name=name,
            model=model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            output=output_text,
            usage={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            },
            metadata={
                "latency_ms": round(elapsed_ms, 1),
                "has_tool_config": tool_config is not None,
            },
        )
    except Exception as exc:
        logger.debug("Failed to log Langfuse generation: %s", exc)

    return response


@contextmanager
def trace_tts_span(
    *,
    trace: Any | None,
    text: str,
    voice_id: str,
    model_id: str,
) -> Generator[dict[str, float], None, None]:
    """Context manager that yields a timing dict for TTS. Records TTFT and total latency."""
    timing: dict[str, float] = {"ttfb_ms": 0.0, "total_ms": 0.0}
    start = time.perf_counter()
    try:
        yield timing
    finally:
        timing["total_ms"] = (time.perf_counter() - start) * 1000
        if trace is not None:
            try:
                trace.span(
                    name="tts_synthesis",
                    input={"text": text, "text_length": len(text)},
                    output={
                        "voice_id": voice_id,
                        "model_id": model_id,
                    },
                    metadata={
                        "ttfb_ms": round(timing["ttfb_ms"], 1),
                        "total_ms": round(timing["total_ms"], 1),
                    },
                )
            except Exception as exc:
                logger.debug("Failed to log Langfuse TTS span: %s", exc)


def flush() -> None:
    """Flush pending Langfuse events (call at shutdown)."""
    lf = get_langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception:
            pass
