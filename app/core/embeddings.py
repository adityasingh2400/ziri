from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def embed_text(bedrock_client: Any, model_id: str, text: str) -> list[float] | None:
    """Generate an embedding vector using Amazon Titan Embeddings via Bedrock.

    Returns a list of floats (1536-dim for Titan v1) or None on failure.
    """
    if bedrock_client is None:
        return None

    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({"inputText": text}),
        )
        body = json.loads(response["body"].read())
        return body.get("embedding")
    except Exception as exc:
        logger.warning("Embedding generation failed (%s): %s", model_id, exc)
        return None


def build_turn_text(
    user_text: str,
    intent_type: str,
    tool_name: str,
    assistant_speak: str,
) -> str:
    """Build a combined string suitable for embedding a conversation turn."""
    parts = [f"User asked: {user_text}"]
    if intent_type:
        parts.append(f"Intent: {intent_type}")
    if tool_name:
        parts.append(f"Tool: {tool_name}")
    if assistant_speak:
        parts.append(f"Response: {assistant_speak}")
    return ". ".join(parts)
