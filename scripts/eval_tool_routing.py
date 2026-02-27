#!/usr/bin/env python3
"""Offline evaluation script for Ziri's intent routing accuracy.

Reads test cases from tests/fixtures/routing_eval.jsonl, runs each through
Brain.route_intent(), and scores correctness.  Optionally logs results to
Langfuse when credentials are configured.

Usage:
    python scripts/eval_tool_routing.py [--fixture PATH] [--verbose]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.settings import get_settings  # noqa: E402
from app.core.brain import Brain  # noqa: E402
from app.core.device_registry import DeviceContext  # noqa: E402
from app.core.memory import InMemoryStore  # noqa: E402
from app.core.tracing import create_trace, get_langfuse  # noqa: E402
from app.schemas import IntentRequest  # noqa: E402

DEFAULT_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "routing_eval.jsonl"


def load_cases(path: Path) -> list[dict]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def score_case(actual_tool: str, actual_args: dict, expected: dict) -> dict:
    tool_match = actual_tool == expected["expected_tool"]
    expected_args = expected.get("expected_args", {})

    if not expected_args:
        arg_score = 1.0
    else:
        matched_keys = sum(
            1 for k, v in expected_args.items()
            if str(actual_args.get(k, "")).lower() == str(v).lower()
        )
        arg_score = matched_keys / len(expected_args)

    return {
        "tool_match": tool_match,
        "arg_score": arg_score,
        "overall": 1.0 if tool_match and arg_score >= 0.5 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ziri tool routing")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    cases = load_cases(args.fixture)
    if not cases:
        print("No test cases found.")
        return

    settings = get_settings()
    memory = InMemoryStore()
    brain = Brain(settings=settings, memory=memory)
    device_context = DeviceContext(
        device_id="eval_device",
        room_name="Office",
        default_speaker="Mac_Speaker",
        spotify_device_id=settings.spotify_default_device_id or "",
    )

    lf = get_langfuse(settings)
    results = []
    correct = 0

    print(f"\nRunning {len(cases)} evaluation cases...\n")
    print(f"{'#':>3}  {'Input':<45} {'Expected':<25} {'Actual':<25} {'OK'}")
    print("-" * 125)

    for i, case in enumerate(cases):
        req = IntentRequest(
            user_id="eval_user",
            device_id="eval_device",
            room="Office",
            raw_text=case["input"],
            timestamp=datetime.now(timezone.utc),
        )

        trace = create_trace(
            name="eval_route_intent",
            user_id="eval_user",
            metadata={"eval_case": i, "input": case["input"]},
            tags=["evaluation"],
            settings=settings,
        )

        decision = brain.route_intent(req, device_context, trace=trace)
        scores = score_case(decision.tool_name, decision.tool_args, case)

        if scores["overall"] >= 1.0:
            correct += 1

        ok_str = "PASS" if scores["overall"] >= 1.0 else "FAIL"
        print(f"{i+1:>3}  {case['input']:<45} {case['expected_tool']:<25} {decision.tool_name:<25} {ok_str}")

        if args.verbose and ok_str == "FAIL":
            print(f"     expected_args={case.get('expected_args', {})}  actual_args={decision.tool_args}")

        if trace is not None:
            try:
                trace.score(name="tool_match", value=1.0 if scores["tool_match"] else 0.0)
                trace.score(name="arg_score", value=scores["arg_score"])
                trace.score(name="overall", value=scores["overall"])
            except Exception:
                pass

        results.append({"case": case, "actual_tool": decision.tool_name, "scores": scores})

    accuracy = correct / len(cases) * 100
    print(f"\n{'='*125}")
    print(f"Accuracy: {correct}/{len(cases)} ({accuracy:.1f}%)")

    if lf is not None:
        try:
            lf.flush()
            print("Results logged to Langfuse.")
        except Exception:
            pass


if __name__ == "__main__":
    main()
