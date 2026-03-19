# macOS helpers

## LaunchAgent (`com.ziri.listener.plist`)

Template for running `run_listener.py` at login via `launchd`.

1. Edit **ProgramArguments**, **WorkingDirectory**, and log paths for your machine.
2. Copy to `~/Library/LaunchAgents/` and load:

```bash
cp scripts/macos/com.ziri.listener.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.ziri.listener.plist
```
