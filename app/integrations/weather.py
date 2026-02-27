from __future__ import annotations

import logging
from typing import Any

import httpx

from app.schemas import ToolResult
from app.settings import Settings

logger = logging.getLogger(__name__)

_WMO_CODES: dict[int, str] = {
    0: "clear skies", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 56: "freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm",
}


class WeatherController:
    def __init__(self, settings: Settings) -> None:
        self.lat = settings.user_latitude
        self.lon = settings.user_longitude

    def get_current_weather(self) -> ToolResult:
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={self.lat}&longitude={self.lon}"
                f"&current=temperature_2m,apparent_temperature,weather_code,relative_humidity_2m,wind_speed_10m"
                f"&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto"
            )
            resp = httpx.get(url, timeout=5)
            data = resp.json()
            current = data.get("current", {})

            temp = current.get("temperature_2m", "?")
            feels_like = current.get("apparent_temperature", "?")
            code = current.get("weather_code", 0)
            humidity = current.get("relative_humidity_2m", "?")
            wind = current.get("wind_speed_10m", "?")
            condition = _WMO_CODES.get(code, "unknown conditions")

            return ToolResult(
                ok=True,
                action_code="WEATHER_CURRENT",
                speak_text=f"Currently {temp} degrees and {condition}, feels like {feels_like}.",
                private_note="",
                payload={"temp": temp, "feels_like": feels_like, "condition": condition, "humidity": humidity, "wind": wind},
            )
        except Exception as exc:
            logger.exception("Weather fetch failed")
            return ToolResult(ok=False, action_code="WEATHER_ERROR", speak_text="I could not get the weather right now.", private_note=str(exc), error="weather_exception")

    def get_sunrise_sunset(self) -> ToolResult:
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={self.lat}&longitude={self.lon}"
                f"&daily=sunrise,sunset&timezone=auto&forecast_days=1"
            )
            resp = httpx.get(url, timeout=5)
            data = resp.json()
            daily = data.get("daily", {})

            sunrise_raw = (daily.get("sunrise") or [""])[0]
            sunset_raw = (daily.get("sunset") or [""])[0]

            sunrise = self._format_time(sunrise_raw)
            sunset = self._format_time(sunset_raw)

            return ToolResult(
                ok=True,
                action_code="WEATHER_SUN",
                speak_text=f"Sunrise is at {sunrise} and sunset is at {sunset}.",
                private_note="",
                payload={"sunrise": sunrise, "sunset": sunset},
            )
        except Exception as exc:
            logger.exception("Sunrise/sunset fetch failed")
            return ToolResult(ok=False, action_code="WEATHER_ERROR", speak_text="I could not get sunrise and sunset times.", private_note=str(exc), error="sun_exception")

    @staticmethod
    def _format_time(iso_str: str) -> str:
        if not iso_str or "T" not in iso_str:
            return "unknown"
        time_part = iso_str.split("T")[1]
        try:
            h, m = time_part.split(":")[:2]
            hour = int(h)
            ampm = "AM" if hour < 12 else "PM"
            display_hour = hour % 12 or 12
            return f"{display_hour}:{m} {ampm}"
        except (ValueError, IndexError):
            return time_part
