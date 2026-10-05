"""REST-API des Zwillings (FastAPI).

  GET  /health          Status, Datenabdeckung, letzter Lauf
  GET  /prices?days=2   Spotpreise aus dem Speicher
  GET  /weather?days=2  Wetter aus dem Speicher
  POST /ingest          Preise und Wetter live abrufen und speichern
  POST /run             Optimierungszyklus (48 h ab jetzt) ausführen
  GET  /schedule        Fahrplan des letzten Laufs

Start: uvicorn backend.api:app --host 0.0.0.0 --port 8000
"""
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta

from fastapi import FastAPI, HTTPException

from twin.site import Site
from . import ingest
from .service import run_cycle
from .store import Store

LAT, LON = 51.96, 7.63          # Münster


def create_app(store: Store, site: Site | None = None, scheduler: bool = False) -> FastAPI:
    site = site or Site()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if scheduler:
            from .scheduler import start_background
            app.state.scheduler = start_background(store, site, LAT, LON)
        yield
        store.close()

    app = FastAPI(title="Energiesystem-Zwilling", version="1.0", lifespan=lifespan)
    app.state.store, app.state.site = store, site

    @app.get("/health")
    def health():
        return {"status": "ok", "site": site.name, "coverage": store.coverage(), "last_run": store.last_run(),
                "attribution": ingest.ATTRIBUTION}

    @app.get("/prices")
    def prices(days: int = 2):
        s = store.prices(datetime.now().astimezone() - timedelta(days=days))
        return {"unit": "EUR/MWh", "attribution": ingest.ATTRIBUTION,
                "values": [{"ts": ts.isoformat(), "eur_mwh": float(v)} for ts, v in s.items()]}

    @app.get("/weather")
    def weather(days: int = 2):
        df = store.weather(datetime.now().astimezone() - timedelta(days=days))
        return {"values": [{"ts": ts.isoformat(), "t_amb_c": float(r.t_amb_c), "ghi_w_m2": float(r.ghi_w_m2)}
                           for ts, r in df.iterrows()]}

    @app.post("/ingest")
    def do_ingest():
        try:
            p, w = ingest.fetch_all(LAT, LON, date.today())
        except Exception as e:                              # Netzfehler sauber melden
            raise HTTPException(502, f"Ingestion fehlgeschlagen: {e}")
        return {"prices": store.upsert_prices(p), "weather": store.upsert_weather(w), "coverage": store.coverage()}

    @app.post("/run")
    def do_run(hours: float = 48.0):
        try:
            r = run_cycle(store, site, hours=hours, note="api")
        except ValueError as e:
            raise HTTPException(409, str(e))
        return {"run_id": r.run_id, "window_start": r.window_start.isoformat(), "steps": r.steps,
                "peak_kw": round(r.peak_kw, 1), "energy_cost_eur": round(r.energy_cost_eur, 2),
                "filled_price_steps": r.filled_price_steps, "fill_method": r.fill_method}

    @app.get("/schedule")
    def schedule():
        last = store.last_run()
        if last is None:
            raise HTTPException(404, "Noch kein Lauf")
        df = store.schedule(last["run_id"])
        return {"run_id": last["run_id"], "window_start": str(last["window_start"]), "peak_kw": last["peak_kw"],
                "rows": [{"ts": ts.isoformat(), **{k: round(float(v), 2) for k, v in row.items()}}
                         for ts, row in df.iterrows()]}

    return app


app = create_app(Store(os.environ.get("TWIN_DB", "data/twin.duckdb")), scheduler=os.environ.get("TWIN_SCHEDULER") == "1")
