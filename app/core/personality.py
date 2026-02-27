from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

PERSONALITY_PROMPT = """\
You are Ziri — a concise, cool voice assistant. Rewrite this music response to sound natural.

Rules:
- MAX 10 words. Shorter is always better.
- Be direct first, personality second.
- Don't add information that isn't in the raw response.
- Don't say "sure", "certainly", "of course", "absolutely".
- Return ONLY the rewritten text. No quotes, no explanation."""

# Pre-built quick reply pools for zero-latency simple commands.
# These bypass the Bedrock rewriter entirely.
QUICK_REPLIES: dict[str, list[str]] = {
    "MUSIC_PAUSE": [
        "Paused.",
        "Got it.",
        "On hold.",
        "Taking a break.",
        "Muted.",
    ],
    "MUSIC_RESUME": [
        "Back on.",
        "Let's go.",
        "Resuming.",
        "Here we go.",
        "And we're back.",
    ],
    "MUSIC_SKIP": [
        "Skipped.",
        "Next one.",
        "Moving on.",
        "Gone.",
        "Next up.",
    ],
    "MUSIC_PREVIOUS": [
        "Going back.",
        "Rewinding.",
        "Back one.",
        "Previous track.",
    ],
    "MUSIC_VOLUME": [
        "Done.",
        "Adjusted.",
        "Got it.",
    ],
    "MUSIC_SHUFFLE": [
        "Done.",
        "Got it.",
        "Set.",
    ],
    "MUSIC_REPEAT": [
        "Done.",
        "Set.",
        "Got it.",
    ],
    "MUSIC_LIKE": [
        "Liked.",
        "Saved.",
        "Added to your library.",
        "Good taste.",
    ],
    "MUSIC_QUEUE": [],  # Uses rewriter — response includes track name
}

# Action codes that skip the rewriter entirely — raw response goes straight through.
# This includes simple commands (quick reply pool) AND info responses where
# directness matters more than personality.
SKIP_REWRITER_ACTIONS = {
    *QUICK_REPLIES.keys(),
    "WEATHER_CURRENT", "WEATHER_SUN", "WEATHER_ERROR",
    "NBA_SCORES", "NBA_ERROR",
    "NEWS_HEADLINES", "NEWS_TOPIC", "NEWS_ERROR",
    "CALENDAR_READ",
    "INFO_REPLY",
    "ERROR",
}

# Subset that uses quick reply pools (simple commands only)
QUICK_REPLY_ACTIONS = set(QUICK_REPLIES.keys()) - {"MUSIC_QUEUE"}


def get_quick_reply(action_code: str) -> str | None:
    pool = QUICK_REPLIES.get(action_code)
    if not pool:
        return None
    return random.choice(pool)


def rewrite_response(
    bedrock_client: Any,
    model_id: str,
    raw_text: str,
    action_code: str,
    user_text: str,
) -> str:
    """Use Bedrock to rewrite a raw response with personality.

    Returns the original text if rewrite fails or times out.
    """
    if not bedrock_client or not raw_text:
        return raw_text

    # Quick replies for simple commands — zero latency
    if action_code in QUICK_REPLY_ACTIONS:
        quick = get_quick_reply(action_code)
        if quick:
            return quick

    # Skip rewriter for info responses — direct answers are better
    if action_code in SKIP_REWRITER_ACTIONS:
        return raw_text

    try:
        user_prompt = f"User said: {user_text}\nRaw response: {raw_text}\nAction: {action_code}"

        response = bedrock_client.converse(
            modelId=model_id,
            system=[{"text": PERSONALITY_PROMPT}],
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            inferenceConfig={"maxTokens": 50, "temperature": 0.75},
        )

        content = response.get("output", {}).get("message", {}).get("content", [])
        rewritten = "".join(part.get("text", "") for part in content if part.get("text")).strip()

        if rewritten and len(rewritten) < 200:
            return rewritten

        return raw_text
    except Exception as exc:
        logger.debug("Personality rewrite skipped: %s", exc)
        return raw_text
