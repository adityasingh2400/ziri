# iOS Shortcut Logic Flow (Aura)

This Shortcut gives you a Siri-style trigger without shipping an App Store app.

## Flow

1. Action: `Dictate Text`
- Capture the user utterance (e.g. "Play Uzi in the living room").

2. Action: `Get Device Details`
- Extract device name (map this to `device_id`).

3. Action: `Current Date`
- Format as ISO8601 for `timestamp`.

4. Action: `Get Current Location` (optional)
- Convert to room name, or use a static room per device.

5. Action: `Text` (Build JSON payload)

```json
{
  "user_id": "Aditya",
  "device_id": "iPhone_Kitchen",
  "room": "Kitchen",
  "raw_text": "${Dictated Text}",
  "timestamp": "${ISO Date}"
}
```

6. Action: `Get Contents of URL`
- Method: `POST`
- URL: `https://<your-aura-host>/intent`
- Headers: `Content-Type: application/json`
- Body: JSON from step 5

7. Action: Branch on Response
- If `speak_text` is not empty: run `Speak Text` with `speak_text`.
- If `audio_url` exists: optionally `Get Contents of URL` then `Play Sound` for House Voice MP3.
- If `private_note` is not empty: run `Show Result` with `private_note`.

8. Optional Pro Action
- If response metadata contains `spotify_url`, run `Open URLs` to force Spotify handoff.
- If response payload indicates `shortcut_action` for reminders/scenes, branch into corresponding Shortcut actions.

## Trigger Options

- Voice trigger phrase: "Hey Siri, Aura"
- Back Tap trigger
- Action Button trigger
- Home Screen widget

## Privacy Rule

For private commands like "Read my texts", Aura should return:

- `speak_text = ""`
- `private_note = "<private output for on-screen display>"`

This keeps sensitive info off loudspeakers.
