"""Einsatzoptimierung als lineares Programm (HiGHS via SciPy).

Entscheidungsvariablen je Stunde t:
  Qc  Kälteleistung thermisch [kW_th]     T   Kühlhaustemperatur [°C]
  ch  Batterie laden [kW]                  dis Batterie entladen [kW]
  E   Batterie-Energieinhalt [kWh]         gi  Netzbezug [kW]   ge Einspeisung [kW]
sowie P = Lastspitze [kW].

Ziel: min  Σ gi·(Spot+Netzentgelt) − Σ ge·Spot·Anteil + P·Leistungspreis
Nebenbedingungen: Energiebilanz, Kühlhaus-Dynamik (Einknotenmodell),
Batterie-Dynamik, Temperaturband, Leistungsgrenzen. Alle Zustände zyklisch
(Ende der Woche = Anfang), damit keine Energie "geliehen" wird.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from .site import Site, Series, H


@dataclass
class Result:
    grid_import: np.ndarray
    grid_export: np.ndarray
    cooler_el: np.ndarray
    temperature: np.ndarray
    battery_soc: np.ndarray
    battery_power: np.ndarray  # + laden, − entladen
    peak_kw: float
    energy_cost_eur: float
    demand_cost_eur_a: float


def _costs(site, s, gi, ge):
    energy = float(np.sum(gi * (s.price + site.grid_fee_eur_mwh) / 1000)
                   - np.sum(ge * s.price * site.feed_in_share / 1000))
    peak = float(gi.max())
    return energy, peak, peak * site.demand_charge_eur_kw_a


def baseline(site: Site, s: Series) -> Result:
    """Status quo: Thermostat hält t_ref, kein Speicher, keine Preisführung."""
    cooler_el = s.cooling_th / site.cop
    net = s.process + cooler_el - s.pv
    gi, ge = np.clip(net, 0, None), np.clip(-net, 0, None)
    energy, peak, demand = _costs(site, s, gi, ge)
    return Result(gi, ge, cooler_el, np.full(H, site.t_ref), np.zeros(H),
                  np.zeros(H), peak, energy, demand)


def optimize(site: Site, s: Series, use_precooling: bool = True) -> Result:
    n_blocks = 7
    idx = {k: i * H for i, k in enumerate(["Qc", "T", "ch", "dis", "E", "gi", "ge"])}
    nP = n_blocks * H
    nv = nP + 1
    v = lambda k, t: idx[k] + (t % H)

    c = np.zeros(nv)
    c[idx["gi"]: idx["gi"] + H] = (s.price + site.grid_fee_eur_mwh) / 1000
    c[idx["ge"]: idx["ge"] + H] = -s.price * site.feed_in_share / 1000
    c[nP] = site.demand_charge_eur_kw_a

    Aeq, beq = lil_matrix((3 * H, nv)), np.zeros(3 * H)
    C, a, eta = site.thermal_cap_kwh_per_k, site.extra_loss_kw_per_k, site.battery_eta
    for t in range(H):
        r = t  # Kühlhaus: C·(T[t+1]−T[t]) = Bedarf + a·(t_ref−T[t]) − Qc
        Aeq[r, v("T", t + 1)] += C
        Aeq[r, v("T", t)] += -C + a
        Aeq[r, v("Qc", t)] = 1
        beq[r] = s.cooling_th[t] + a * site.t_ref
        r = H + t  # Batterie: E[t+1] = E[t] + η·ch − dis/η
        Aeq[r, v("E", t + 1)] += 1
        Aeq[r, v("E", t)] += -1
        Aeq[r, v("ch", t)] = -eta
        Aeq[r, v("dis", t)] = 1 / eta
        r = 2 * H + t  # Bilanz: gi − ge − Qc/COP − ch + dis = Prozess − PV
        Aeq[r, v("gi", t)] = 1
        Aeq[r, v("ge", t)] = -1
        Aeq[r, v("Qc", t)] = -1 / site.cop
        Aeq[r, v("ch", t)] = -1
        Aeq[r, v("dis", t)] = 1
        beq[r] = s.process[t] - s.pv[t]

    Aub = lil_matrix((H, nv))
    for t in range(H):  # gi[t] ≤ P
        Aub[t, v("gi", t)] = 1
        Aub[t, nP] = -1
    bub = np.zeros(H)

    p_bat = site.battery_kwh * site.battery_c_rate
    tlo, thi = (site.t_min, site.t_max) if use_precooling else (site.t_ref, site.t_ref)
    bounds = ([(0, site.cooler_el_max_kw * site.cop)] * H + [(tlo, thi)] * H
              + [(0, p_bat)] * H * 2
              + [(0.05 * site.battery_kwh, 0.95 * site.battery_kwh)] * H
              + [(0, None)] * H + [(0, s.pv.max())] * H + [(0, None)])

    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=Aeq.tocsr(), b_eq=beq,
                  bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Optimierung fehlgeschlagen: {res.message}")
    x = res.x
    get = lambda k: x[idx[k]: idx[k] + H]
    gi, ge = get("gi"), get("ge")
    energy, peak, demand = _costs(site, s, gi, ge)
    return Result(gi, ge, get("Qc") / site.cop, get("T"), get("E"),
                  get("ch") - get("dis"), peak, energy, demand)


def annual_savings(site: Site, base: Result, opt: Result) -> float:
    """Repräsentative Woche ×52 für Arbeitskosten, Leistungspreis direkt."""
    return 52 * (base.energy_cost_eur - opt.energy_cost_eur) + (base.demand_cost_eur_a - opt.demand_cost_eur_a)
