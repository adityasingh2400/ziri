"""Prometheus metrics for Ziri observability.

Histograms track latency distributions; counters track event volumes.
All metrics are prefixed with ``ziri_`` for namespace isolation.
"""
from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram, make_asgi_app

    REQUEST_DURATION = Histogram(
        "ziri_request_duration_seconds",
        "HTTP request latency",
        labelnames=["method", "endpoint", "status"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )

    INTENT_ROUTING_DURATION = Histogram(
        "ziri_intent_routing_seconds",
        "Time spent in the brain / LLM intent routing stage",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    )

    TTS_TTFB = Histogram(
        "ziri_tts_ttfb_seconds",
        "TTS time-to-first-byte",
        buckets=(0.05, 0.1, 0.2, 0.4, 0.6, 0.9, 1.5, 3.0),
    )

    TOOL_EXECUTION_DURATION = Histogram(
        "ziri_tool_execution_seconds",
        "Per-tool execution latency",
        labelnames=["tool_name"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    )

    DETERMINISTIC_ROUTE_TOTAL = Counter(
        "ziri_deterministic_route_total",
        "Deterministic fast-path routing outcomes",
        labelnames=["outcome"],
    )

    LLM_CALLS_TOTAL = Counter(
        "ziri_llm_calls_total",
        "LLM invocations",
        labelnames=["model", "purpose"],
    )

    PROMETHEUS_AVAILABLE = True

except ImportError:
    PROMETHEUS_AVAILABLE = False

    class _Noop:
        """Transparent no-op stand-in when prometheus_client is absent."""
        def labels(self, *a, **kw):
            return self
        def observe(self, *a, **kw):
            pass
        def inc(self, *a, **kw):
            pass
        def time(self):
            import contextlib
            return contextlib.nullcontext()

    REQUEST_DURATION = _Noop()           # type: ignore[assignment]
    INTENT_ROUTING_DURATION = _Noop()    # type: ignore[assignment]
    TTS_TTFB = _Noop()                  # type: ignore[assignment]
    TOOL_EXECUTION_DURATION = _Noop()    # type: ignore[assignment]
    DETERMINISTIC_ROUTE_TOTAL = _Noop()  # type: ignore[assignment]
    LLM_CALLS_TOTAL = _Noop()           # type: ignore[assignment]
    make_asgi_app = None                 # type: ignore[assignment]
