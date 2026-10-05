"""Ein Optimierungszyklus: Daten aus dem Speicher, 48-h-Fenster bauen, rollierend optimieren, Fahrplan speichern.

Prozesslast und Kältebedarf kommen aus dem Standortprofil (twin.site), bis Messdaten angebunden
sind. PV wird aus der Globalstrahlung abgeleitet: P = P_peak · GHI/1000 · Performance-Ratio.
Fehlen Preise für den hinteren Teil des Fensters (Day-Ahead reicht bis Ende des Folgetags),
wird mit Persistenz (Vortagspreis) aufgefüllt.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from twin.optimize import optimize
from twin.site import Series, Site, ambient_profile, cooling_demand, process_profile
from .store import Store

PERFORMANCE_RATIO = 0.85
TZ = "Europe/Berlin"


@dataclass
class CycleResult:
    run_id: int
    window_start: pd.Timestamp
    steps: int
    peak_kw: float
    energy_cost_eur: float
    filled_price_steps: int
    schedule: pd.DataFrame


def build_window(store: Store, site: Site, start: pd.Timestamp, hours: float = 48.0, dt_h: float = 1.0):
    """Zeitreihen des Fensters aus dem Speicher; Rückgabe: (Series, Zeitindex, aufgefüllte Preisschritte)."""
    idx = pd.date_range(start, periods=int(round(hours / dt_h)), freq=f"{int(dt_h * 60)}min", tz=TZ)
    prices = store.prices(start - timedelta(days=1), idx[-1] + timedelta(hours=1))
    if prices.empty:
        raise ValueError("Keine Preise im Speicher: erst Ingestion ausführen")
    on_grid = lambda target: prices.reindex(prices.index.union(target)).interpolate(limit_area="inside").reindex(target)
    price = on_grid(idx)
    missing = price.isna()
    if missing.any():                                         # Persistenz: gleicher Zeitpunkt am Vortag
        prev = on_grid(idx - timedelta(days=1)).to_numpy()
        price = pd.Series(np.where(missing, prev, price.to_numpy()), index=idx).ffill().bfill()
    weather = store.weather(start - timedelta(hours=2), idx[-1] + timedelta(hours=2))
    if weather.empty:
        raise ValueError("Kein Wetter im Speicher: erst Ingestion ausführen")
    weather = weather.reindex(weather.index.union(idx)).interpolate(limit_direction="both").reindex(idx)
    t_amb = weather["t_amb_c"].to_numpy()
    if np.isnan(t_amb).any():
        t_amb = np.where(np.isnan(t_amb), ambient_profile(idx.hour + idx.minute / 60), t_amb)
    pv = np.clip(site.pv_kwp * weather["ghi_w_m2"].fillna(0).to_numpy() / 1000 * PERFORMANCE_RATIO, 0, site.pv_kwp)
    hod, dow = (idx.hour + idx.minute / 60).to_numpy(), idx.dayofweek.to_numpy()
    series = Series(price.to_numpy(dtype=float), pv, process_profile(site, hod, dow), cooling_demand(site, t_amb, hod, dow),
                    t_amb, dt_h)
    return series, idx, int(missing.sum())


def run_cycle(store: Store, site: Site, now: datetime | None = None, hours: float = 48.0, dt_h: float = 1.0,
              note: str = "") -> CycleResult:
    """Fenster ab der nächsten vollen Stunde optimieren, Startzustand aus dem letzten Lauf übernehmen."""
    now = pd.Timestamp(now or datetime.now()).tz_localize(TZ) if (now is None or pd.Timestamp(now).tzinfo is None) \
        else pd.Timestamp(now).tz_convert(TZ)
    start = now.ceil("h")
    series, idx, filled = build_window(store, site, start, hours, dt_h)
    last = store.last_run()
    t0 = site.t_ref
    e0 = 0.5 * site.battery_kwh if site.battery_kwh else 0.0
    if last and last["t_end_c"] is not None:
        t0 = float(np.clip(last["t_end_c"], site.t_min, site.t_max))
        if site.battery_kwh:
            e0 = float(np.clip(last["e_end_kwh"], 0.05 * site.battery_kwh, 0.95 * site.battery_kwh))
    r = optimize(site, series, start=(t0, e0))
    schedule = pd.DataFrame({"price_eur_mwh": series.price, "pv_kw": series.pv, "process_kw": series.process,
                             "cooler_el_kw": r.cooler_el, "battery_kw": r.battery_power,
                             "temperature_c": r.temperature, "grid_kw": r.grid_import - r.grid_export}, index=idx)
    applied = int(round(24 / dt_h))                           # Zustand nach dem ersten Tag, der umgesetzt wird
    run_id = store.save_run(start, schedule, dt_h, r.peak_kw, r.energy_cost_eur,
                            float(r.temperature_path[min(applied, len(idx))]),
                            float(r.soc_path[min(applied, len(idx))]), note=note)
    return CycleResult(run_id, start, len(idx), r.peak_kw, r.energy_cost_eur, filled, schedule)
