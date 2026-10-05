"""Rollierende Optimierung (Model Predictive Control).

Statt einmal die ganze Woche mit bekannten Preisen zu lösen, wird jeden Tag ein Fenster
von 48 Stunden gegen die Preisprognose optimiert, nur der erste Tag umgesetzt und der
erreichte Zustand (Temperatur, Ladezustand) in das nächste Fenster übergeben. Bewertet
wird der umgesetzte Fahrplan anschließend mit den wahren Preisen.
"""
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

from .optimize import Result, evaluate_schedule, optimize
from .site import Series, Site


@dataclass
class RollingResult:
    realized: Result           # umgesetzter Fahrplan, bewertet mit wahren Preisen
    windows: int
    forecast_rmse_eur_mwh: float


def rolling_horizon(site: Site, s: Series, price_forecast: Optional[np.ndarray] = None,
                    window_h: float = 48.0, step_h: float = 24.0,
                    on_window: Optional[Callable[[int, Result], None]] = None) -> RollingResult:
    """Fahrplan Fenster für Fenster bestimmen und gegen die wahren Preise bewerten."""
    n, dt = s.n, s.dt_h
    w, step = int(round(window_h / dt)), int(round(step_h / dt))
    forecast = s.price if price_forecast is None else np.asarray(price_forecast)
    t_state = site.t_ref
    e_state = 0.5 * site.battery_kwh if site.battery_kwh else 0.0
    cooling_th, battery = np.zeros(n), np.zeros(n)
    windows = 0
    for k0 in range(0, n, step):
        idx = [(k0 + j) % n for j in range(w)]              # Horizontende zyklisch auffüllen
        window = Series(forecast[idx], s.pv[idx], s.process[idx], s.cooling_th[idx], s.t_amb[idx], dt)
        r = optimize(site, window, start=(t_state, e_state))
        apply = min(step, n - k0)
        cooling_th[k0:k0 + apply] = r.cooler_el[:apply] * site.cop
        battery[k0:k0 + apply] = r.battery_power[:apply]
        t_state, e_state = float(r.temperature_path[apply]), float(r.soc_path[apply])
        windows += 1
        if on_window:
            on_window(k0, r)
    realized = evaluate_schedule(site, s, cooling_th, battery)
    rmse = float(np.sqrt(np.mean((forecast - s.price) ** 2)))
    return RollingResult(realized, windows, rmse)
