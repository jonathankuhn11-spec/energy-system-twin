"""Plattform: Parser gegen aufgezeichnete API-Strukturen, Speicher, Zyklus, REST-API."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import api, ingest, scheduler
from backend.service import build_window, run_cycle
from backend.store import Store
from twin.site import Site

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def prices():
    return ingest.parse_energy_charts(json.load(open(FIX / "energy_charts_price.json")))


@pytest.fixture
def weather():
    return ingest.parse_open_meteo(json.load(open(FIX / "open_meteo_forecast.json")))


@pytest.fixture
def store(prices, weather):
    st = Store()
    st.upsert_prices(prices)
    st.upsert_weather(weather)
    yield st
    st.close()


def test_price_parser_handles_quarter_hours_and_timezone(prices):
    assert len(prices) == 192 and prices.index.tz is not None
    assert str(prices.index[0]) == "2026-10-05 00:00:00+02:00"
    assert (prices.index[1] - prices.index[0]) == pd.Timedelta(minutes=15)
    with pytest.raises(ValueError):
        ingest.parse_energy_charts({"unit": "ct/kWh", "unix_seconds": [0], "price": [1.0]})


def test_weather_parser(weather):
    assert list(weather.columns) == ["t_amb_c", "ghi_w_m2"] and len(weather) == 72
    assert weather["ghi_w_m2"].max() > 300 and weather["ghi_w_m2"].min() == 0


def test_store_upserts_are_idempotent(store, prices, weather):
    assert store.upsert_prices(prices) == 192 and store.coverage()["prices"]["rows"] == 192
    assert store.upsert_weather(weather) == 72 and store.coverage()["weather"]["rows"] == 72
    assert store.prices(pd.Timestamp("2026-10-06", tz="Europe/Berlin")).index.min().day == 6


def test_window_is_built_on_the_hour_and_prices_are_filled_by_persistence(store):
    start = pd.Timestamp("2026-10-06 14:00", tz="Europe/Berlin")
    series, idx, filled = build_window(store, Site(), start, hours=48)
    assert series.n == 48 and idx[0] == start
    assert filled > 0                                   # Preise reichen nur bis 06.10. 23:45
    assert series.pv.max() > 0 and series.pv.max() <= Site().pv_kwp
    assert series.price.min() > 0 and not pd.isna(series.price).any()


def test_cycle_stores_a_schedule_and_chains_the_state(store):
    site = Site(battery_kwh=500)
    r1 = run_cycle(store, site, now=datetime(2026, 10, 5, 13, 30), note="test")
    assert r1.steps == 48 and r1.peak_kw > 0 and store.last_run()["run_id"] == r1.run_id
    sched = store.schedule(r1.run_id)
    assert len(sched) == 48 and sched["temperature_c"].between(site.t_min - 1e-6, site.t_max + 1e-6).all()
    r2 = run_cycle(store, site, now=datetime(2026, 10, 6, 13, 30))
    last = store.last_run()
    assert last["run_id"] == r2.run_id == r1.run_id + 1
    assert site.t_min <= last["t_end_c"] <= site.t_max and 25 <= last["e_end_kwh"] <= 475


def test_cycle_without_data_fails_cleanly():
    st = Store()
    with pytest.raises(ValueError):
        run_cycle(st, Site(), now=datetime(2026, 10, 5, 13, 30))
    st.close()


def test_api_end_to_end(store, monkeypatch):
    client = TestClient(api.create_app(store))
    assert client.get("/schedule").status_code == 404
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["coverage"]["prices"]["rows"] == 192
    monkeypatch.setattr(api, "datetime", _FrozenDatetime)
    monkeypatch.setattr("backend.service.datetime", _FrozenDatetime)
    assert len(client.get("/prices", params={"days": 1}).json()["values"]) > 0
    run = client.post("/run").json()
    assert run["steps"] == 48 and run["peak_kw"] > 0
    sched = client.get("/schedule").json()
    assert sched["run_id"] == run["run_id"] and len(sched["rows"]) == 48
    assert {"ts", "cooler_el_kw", "temperature_c", "grid_kw"} <= set(sched["rows"][0])


def test_ingest_endpoint_reports_network_failures(store, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(ingest, "fetch_all", boom)
    client = TestClient(api.create_app(store))
    assert client.post("/ingest").status_code == 502


def test_scheduler_timing():
    now = datetime(2026, 10, 5, 13, 0)
    assert scheduler.seconds_until(13, 30, now) == 1800
    assert scheduler.seconds_until(13, 30, datetime(2026, 10, 5, 14, 0)) == 23.5 * 3600


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        base = cls(2026, 10, 5, 13, 30)
        return base.astimezone(tz) if tz else base
