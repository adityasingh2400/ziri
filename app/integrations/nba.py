from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from app.schemas import ToolResult

logger = logging.getLogger(__name__)

_ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

_TEAM_ALIASES: dict[str, str] = {
    "lakers": "LAL", "celtics": "BOS", "warriors": "GSW", "nets": "BKN",
    "knicks": "NYK", "heat": "MIA", "bulls": "CHI", "sixers": "PHI",
    "76ers": "PHI", "bucks": "MIL", "suns": "PHX", "nuggets": "DEN",
    "clippers": "LAC", "mavs": "DAL", "mavericks": "DAL", "rockets": "HOU",
    "spurs": "SAS", "raptors": "TOR", "hawks": "ATL", "cavs": "CLE",
    "cavaliers": "CLE", "pacers": "IND", "magic": "ORL", "wizards": "WAS",
    "hornets": "CHA", "pistons": "DET", "timberwolves": "MIN", "wolves": "MIN",
    "blazers": "POR", "trail blazers": "POR", "thunder": "OKC", "jazz": "UTA",
    "pelicans": "NOP", "kings": "SAC", "grizzlies": "MEM",
}


class NBAController:
    def get_todays_games(self) -> ToolResult:
        try:
            resp = httpx.get(_ESPN_SCOREBOARD, timeout=5)
            data = resp.json()
            events = data.get("events", [])

            if not events:
                return ToolResult(ok=True, action_code="NBA_SCORES", speak_text="No NBA games today.", private_note="")

            lines = []
            for event in events:
                name = event.get("name", "")
                status_obj = event.get("status", {})
                status_type = status_obj.get("type", {}).get("name", "")
                status_detail = status_obj.get("type", {}).get("shortDetail", status_obj.get("type", {}).get("detail", ""))

                competitors = event.get("competitions", [{}])[0].get("competitors", [])
                if len(competitors) == 2:
                    home = competitors[0]
                    away = competitors[1]
                    home_name = home.get("team", {}).get("shortDisplayName", "?")
                    away_name = away.get("team", {}).get("shortDisplayName", "?")
                    home_score = home.get("score", "0")
                    away_score = away.get("score", "0")

                    if status_type == "STATUS_FINAL":
                        lines.append(f"{away_name} {away_score}, {home_name} {home_score} — Final")
                    elif status_type == "STATUS_IN_PROGRESS":
                        detail = status_obj.get("type", {}).get("shortDetail", "Live")
                        lines.append(f"{away_name} {away_score}, {home_name} {home_score} — {detail}")
                    else:
                        game_time = status_obj.get("type", {}).get("shortDetail", "Scheduled")
                        lines.append(f"{away_name} vs {home_name} — {game_time}")
                else:
                    lines.append(name)

            summary = ". ".join(lines[:5])
            return ToolResult(
                ok=True,
                action_code="NBA_SCORES",
                speak_text=summary,
                private_note="",
                payload={"games": lines},
            )
        except Exception as exc:
            logger.exception("NBA scores fetch failed")
            return ToolResult(ok=False, action_code="NBA_ERROR", speak_text="I could not get NBA scores right now.", private_note=str(exc), error="nba_exception")

    def get_team_score(self, team_query: str) -> ToolResult:
        abbrev = self._resolve_team(team_query)
        result = self.get_todays_games()
        if not result.ok:
            return result

        games = result.payload.get("games", [])
        if not games:
            return ToolResult(ok=True, action_code="NBA_SCORES", speak_text=f"No games found for {team_query} today.", private_note="")

        if abbrev:
            for game in games:
                if abbrev.lower() in game.lower() or team_query.lower() in game.lower():
                    return ToolResult(ok=True, action_code="NBA_SCORES", speak_text=game, private_note="", payload={"game": game})

        for game in games:
            if team_query.lower() in game.lower():
                return ToolResult(ok=True, action_code="NBA_SCORES", speak_text=game, private_note="", payload={"game": game})

        return ToolResult(ok=True, action_code="NBA_SCORES", speak_text=f"I didn't find a game for {team_query} today.", private_note="")

    @staticmethod
    def _resolve_team(query: str) -> str | None:
        q = query.lower().strip()
        return _TEAM_ALIASES.get(q)
