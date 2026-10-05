"""Ingestion echter Marktdaten und Wetterdaten, beides ohne Registrierung.

  Day-Ahead-Preise  Energy-Charts (Fraunhofer ISE), Gebotszone DE-LU, CC BY 4.0
                    https://api.energy-charts.info/price?bzn=DE-LU&start=...&end=...
  Wetter            Open-Meteo, stündliche Temperatur und Globalstrahlung
                    https://api.open-meteo.com/v1/forecast

Die Parser sind von den HTTP-Aufrufen getrennt, damit sie gegen aufgezeichnete Antworten
getestet werden können. Seit Oktober 2025 liefert der Day-Ahead-Markt Viertelstundenwerte;
die Parser kommen mit jeder Schrittweite zurecht.
"""
from datetime import date, timedelta

import httpx
import pandas as pd

ENERGY_CHARTS = "https://api.energy-charts.info/price"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
ATTRIBUTION = "Preise: Energy-Charts (Fraunhofer ISE), CC BY 4.0 · Wetter: Open-Meteo"
TZ = "Europe/Berlin"


def parse_energy_charts(payload: dict) -> pd.Series:
    """Preis in €/MWh je Zeitstempel (lokale Zeit)."""
    if payload.get("unit", "EUR/MWh") != "EUR/MWh":
        raise ValueError(f"Unerwartete Einheit: {payload.get('unit')}")
    idx = pd.to_datetime(payload["unix_seconds"], unit="s", utc=True).tz_convert(TZ)
    s = pd.Series(payload["price"], index=idx, name="eur_mwh", dtype="float64")
    return s[~s.index.duplicated()].sort_index()


def parse_open_meteo(payload: dict) -> pd.DataFrame:
    """Außentemperatur [°C] und Globalstrahlung [W/m²] je Stunde (lokale Zeit)."""
    h = payload["hourly"]
    idx = pd.to_datetime(h["time"]).tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward")
    df = pd.DataFrame({"t_amb_c": h["temperature_2m"], "ghi_w_m2": h["shortwave_radiation"]}, index=idx)
    return df[~df.index.isna()]


def fetch_prices(start: date, end: date, bzn: str = "DE-LU", timeout: float = 20.0) -> pd.Series:
    r = httpx.get(ENERGY_CHARTS, params={"bzn": bzn, "start": start.isoformat(), "end": end.isoformat()},
                  timeout=timeout)
    r.raise_for_status()
    return parse_energy_charts(r.json())


def fetch_weather(lat: float, lon: float, days: int = 3, timeout: float = 20.0) -> pd.DataFrame:
    r = httpx.get(OPEN_METEO, params={"latitude": lat, "longitude": lon, "forecast_days": days,
                                      "hourly": "temperature_2m,shortwave_radiation", "timezone": TZ},
                  timeout=timeout)
    r.raise_for_status()
    return parse_open_meteo(r.json())


def fetch_all(lat: float, lon: float, today: date | None = None):
    today = today or date.today()
    return fetch_prices(today - timedelta(days=7), today + timedelta(days=1)), fetch_weather(lat, lon)
