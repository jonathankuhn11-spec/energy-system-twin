"""Täglicher Zyklus: um 13:30 Uhr, wenn die Day-Ahead-Preise für den Folgetag vorliegen.

Läuft als Thread im API-Prozess (TWIN_SCHEDULER=1), weil DuckDB nur einen Schreiber zulässt.
"""
import logging
import threading
from datetime import date, datetime, timedelta

from twin.site import Site
from . import ingest
from .service import run_cycle
from .store import Store

log = logging.getLogger("twin.scheduler")


def seconds_until(hour: int, minute: int, now: datetime | None = None) -> float:
    now = now or datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


def cycle(store: Store, site: Site, lat: float, lon: float) -> dict:
    prices, weather = ingest.fetch_all(lat, lon, date.today())
    store.upsert_prices(prices)
    store.upsert_weather(weather)
    r = run_cycle(store, site, note="scheduler")
    log.info("Lauf %s: Spitze %.0f kW, Arbeitskosten %.0f €", r.run_id, r.peak_kw, r.energy_cost_eur)
    return {"run_id": r.run_id, "peak_kw": r.peak_kw}


def start_background(store: Store, site: Site, lat: float, lon: float, hour: int = 13, minute: int = 30):
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            if stop.wait(seconds_until(hour, minute)):
                break
            try:
                cycle(store, site, lat, lon)
            except Exception as e:                           # ein fehlgeschlagener Tag bricht den Dienst nicht ab
                log.error("Zyklus fehlgeschlagen: %s", e)
    thread = threading.Thread(target=loop, name="twin-scheduler", daemon=True)
    thread.start()
    return stop
