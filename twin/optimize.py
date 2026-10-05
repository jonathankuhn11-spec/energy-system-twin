"""Einsatzoptimierung als lineares Programm (HiGHS via SciPy).

Entscheidungsvariablen je Zeitschritt t (Schrittweite dt):
  Qc  Kälteleistung thermisch [kW_th]     T   Kühlhaustemperatur [°C]  (n+1 Zustände)
  ch  Batterie laden [kW]                  dis Batterie entladen [kW]
  E   Batterie-Energieinhalt [kWh]         gi  Netzbezug [kW]   ge Einspeisung [kW]
sowie P = Lastspitze [kW].

Ziel: min  Σ gi·dt·(Spot+Netzentgelt) − Σ ge·dt·Spot·Anteil + P·Leistungspreis
Nebenbedingungen: Energiebilanz, Kühlhaus-Dynamik (Einknotenmodell),
Batterie-Dynamik, Temperaturband, Leistungsgrenzen.

Zwei Betriebsarten für die Zustände:
  zyklisch (Standard): Ende des Horizonts = Anfang, damit keine Energie "geliehen" wird
  Fenster (start=...): Anfangszustand vorgegeben, am Ende darf das Kühlhaus nicht wärmer
  als t_ref und die Batterie nicht leerer als am Anfang sein (rollierende Optimierung)
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from .site import Site, Series


@dataclass
class Result:
    grid_import: np.ndarray
    grid_export: np.ndarray
    cooler_el: np.ndarray
    temperature: np.ndarray        # T[0..n-1]
    battery_soc: np.ndarray        # E[0..n-1]
    battery_power: np.ndarray      # + laden, − entladen
    peak_kw: float
    energy_cost_eur: float         # Arbeitskosten über den Horizont
    demand_cost_eur_a: float       # Leistungspreis p. a.
    temperature_path: Optional[np.ndarray] = None   # T[0..n], nur Optimierung
    soc_path: Optional[np.ndarray] = None           # E[0..n], nur Optimierung


def _costs(site: Site, s: Series, gi, ge):
    dt = s.dt_h
    energy = float(np.sum(gi * dt * (s.price + site.grid_fee_eur_mwh) / 1000)
                   - np.sum(ge * dt * s.price * site.feed_in_share / 1000))
    peak = float(gi.max())
    return energy, peak, peak * site.demand_charge_eur_kw_a


def annual_factor(s: Series) -> float:
    """Hochrechnung der Arbeitskosten auf das Jahr (Woche ×52, Monat ×12,14 ...)."""
    return 52 * 168 / s.hours


def baseline(site: Site, s: Series) -> Result:
    """Status quo: Thermostat hält t_ref, kein Speicher, keine Preisführung."""
    cooler_el = s.cooling_th / site.cop
    net = s.process + cooler_el - s.pv
    gi, ge = np.clip(net, 0, None), np.clip(-net, 0, None)
    energy, peak, demand = _costs(site, s, gi, ge)
    n = s.n
    return Result(gi, ge, cooler_el, np.full(n, site.t_ref), np.zeros(n), np.zeros(n), peak, energy, demand)


def evaluate_schedule(site: Site, s: Series, cooling_th, battery_power, t0: Optional[float] = None,
                      e0: Optional[float] = None) -> Result:
    """Rechnet einen vorgegebenen Fahrplan (Kälte, Batterie) gegen die echten Zeitreihen durch."""
    n, dt = s.n, s.dt_h
    C, a, eta = site.thermal_cap_kwh_per_k, site.extra_loss_kw_per_k, site.battery_eta
    T = np.empty(n + 1); E = np.empty(n + 1)
    T[0] = site.t_ref if t0 is None else t0
    E[0] = (0.5 * site.battery_kwh if site.battery_kwh else 0.0) if e0 is None else e0
    ch, dis = np.clip(battery_power, 0, None), np.clip(-battery_power, 0, None)
    for t in range(n):
        T[t + 1] = T[t] + dt * (s.cooling_th[t] + a * (site.t_ref - T[t]) - cooling_th[t]) / C
        E[t + 1] = E[t] + dt * (eta * ch[t] - dis[t] / eta)
    cooler_el = cooling_th / site.cop
    net = s.process + cooler_el + ch - dis - s.pv
    gi, ge = np.clip(net, 0, None), np.clip(-net, 0, None)
    energy, peak, demand = _costs(site, s, gi, ge)
    return Result(gi, ge, cooler_el, T[:n], E[:n], battery_power, peak, energy, demand, T, E)


def optimize(site: Site, s: Series, use_precooling: bool = True, start: Optional[tuple] = None) -> Result:
    """LP über den ganzen Horizont. start=(T0, E0) schaltet vom zyklischen Modus in den Fenstermodus."""
    n, dt = s.n, s.dt_h
    sizes = {"Qc": n, "T": n + 1, "ch": n, "dis": n, "E": n + 1, "gi": n, "ge": n}
    idx, off = {}, 0
    for k, size in sizes.items():
        idx[k] = off
        off += size
    nP = off                      # Index der Spitzenlast-Variable
    nv = nP + 1
    v = lambda k, t: idx[k] + t

    c = np.zeros(nv)
    c[idx["gi"]: idx["gi"] + n] = dt * (s.price + site.grid_fee_eur_mwh) / 1000
    c[idx["ge"]: idx["ge"] + n] = -dt * s.price * site.feed_in_share / 1000
    c[nP] = site.demand_charge_eur_kw_a

    cyclic = start is None
    n_eq = 3 * n + (2 if cyclic else 0)
    Aeq, beq = lil_matrix((n_eq, nv)), np.zeros(n_eq)
    C, a, eta = site.thermal_cap_kwh_per_k, site.extra_loss_kw_per_k, site.battery_eta
    for t in range(n):
        r = t  # Kühlhaus: C·(T[t+1]−T[t]) = dt·(Bedarf + a·(t_ref−T[t]) − Qc)
        Aeq[r, v("T", t + 1)] = C
        Aeq[r, v("T", t)] = -C + dt * a
        Aeq[r, v("Qc", t)] = dt
        beq[r] = dt * (s.cooling_th[t] + a * site.t_ref)
        r = n + t  # Batterie: E[t+1] = E[t] + dt·(η·ch − dis/η)
        Aeq[r, v("E", t + 1)] = 1
        Aeq[r, v("E", t)] = -1
        Aeq[r, v("ch", t)] = -dt * eta
        Aeq[r, v("dis", t)] = dt / eta
        r = 2 * n + t  # Bilanz: gi − ge − Qc/COP − ch + dis = Prozess − PV
        Aeq[r, v("gi", t)] = 1
        Aeq[r, v("ge", t)] = -1
        Aeq[r, v("Qc", t)] = -1 / site.cop
        Aeq[r, v("ch", t)] = -1
        Aeq[r, v("dis", t)] = 1
        beq[r] = s.process[t] - s.pv[t]
    if cyclic:
        Aeq[3 * n, v("T", n)] = 1
        Aeq[3 * n, v("T", 0)] = -1
        Aeq[3 * n + 1, v("E", n)] = 1
        Aeq[3 * n + 1, v("E", 0)] = -1

    Aub = lil_matrix((n, nv))
    for t in range(n):  # gi[t] ≤ P
        Aub[t, v("gi", t)] = 1
        Aub[t, nP] = -1
    bub = np.zeros(n)

    cap = site.battery_kwh
    p_bat = cap * site.battery_c_rate
    tlo, thi = (site.t_min, site.t_max) if use_precooling else (site.t_ref, site.t_ref)
    t_bounds = [(tlo, thi)] * (n + 1)
    e_bounds = [(0.05 * cap, 0.95 * cap)] * (n + 1)
    if not cyclic:
        t0, e0 = start
        t_bounds[0], e_bounds[0] = (t0, t0), (e0, e0)
        t_bounds[n] = (tlo, min(site.t_ref, thi))                 # Ende nicht wärmer als Sollwert
        e_bounds[n] = (max(e0, 0.05 * cap), 0.95 * cap)            # Ende nicht leerer als Anfang
    bounds = ([(0, site.cooler_el_max_kw * site.cop)] * n + t_bounds
              + [(0, p_bat)] * n * 2 + e_bounds
              + [(0, None)] * n + [(0, float(s.pv.max()))] * n + [(0, None)])

    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Optimierung fehlgeschlagen: {res.message}")
    x = res.x
    get = lambda k: x[idx[k]: idx[k] + sizes[k]]
    gi, ge, T, E = get("gi"), get("ge"), get("T"), get("E")
    energy, peak, demand = _costs(site, s, gi, ge)
    return Result(gi, ge, get("Qc") / site.cop, T[:n], E[:n], get("ch") - get("dis"),
                  peak, energy, demand, T, E)


def annual_savings(site: Site, base: Result, opt: Result, s: Optional[Series] = None) -> float:
    """Arbeitskosten hochgerechnet (Standard: Woche ×52), Leistungspreis direkt."""
    factor = annual_factor(s) if s is not None else 52
    return factor * (base.energy_cost_eur - opt.energy_cost_eur) + (base.demand_cost_eur_a - opt.demand_cost_eur_a)
