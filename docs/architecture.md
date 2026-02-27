# Aura Architecture

## Request Lifecycle

1. iOS Shortcut sends payload to `POST /intent`.
2. Hub logs incoming request in `sessions`.
3. Brain routes intent using Bedrock Claude (or heuristic fallback).
4. Tool runner executes selected integration call.
5. Orchestrator builds response and stores conversational memory.
6. Optional Polly synthesis returns `audio_url`.

## Intent Classes

- `MUSIC_COMMAND`
- `INFO_QUERY`
- `PERSONAL_REMINDER`
- `HOME_SCENE`

## Stateful Follow-Ups

Memory is persisted per user so follow-ups like `play it again` and `play it louder` can use prior context.

## Privacy Rule

Sensitive requests must avoid loudspeaker output:

- `speak_text` is empty
- `private_note` contains details for phone display
