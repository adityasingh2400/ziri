from __future__ import annotations

import random
from typing import Any

QUICK_REPLIES: dict[str, list[str]] = {
    "MUSIC_PAUSE": ["Paused.", "Got it.", "On hold.", "Taking a break."],
    "MUSIC_RESUME": ["Back on.", "Resuming.", "Here we go.", "And we're back."],
    "MUSIC_SKIP": ["Skipped.", "Next one.", "Moving on.", "Next up."],
    "MUSIC_PREVIOUS": ["Going back.", "Rewinding.", "Back one."],
    "MUSIC_VOLUME": ["Done.", "Adjusted.", "Got it."],
    "MUSIC_SHUFFLE": ["Done.", "Got it.", "Set."],
    "MUSIC_REPEAT": ["Done.", "Set.", "Got it."],
    "MUSIC_LIKE": ["Liked.", "Saved.", "Good taste."],
}

QUICK_REPLY_ACTIONS = set(QUICK_REPLIES.keys())


def rewrite_response(
    bedrock_client: Any,
    model_id: str,
    raw_text: str,
    action_code: str,
    user_text: str,
) -> str:
    """Fast response processor. No Bedrock calls — everything is instant.

    Simple commands get a quick reply from the pool.
    Everything else passes through the raw text unchanged.
    """
    if not raw_text:
        return raw_text

    if action_code in QUICK_REPLY_ACTIONS:
        quick = QUICK_REPLIES.get(action_code)
        if quick:
            return random.choice(quick)

    return raw_text
