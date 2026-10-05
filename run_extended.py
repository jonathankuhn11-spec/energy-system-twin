"""Erweiterungen durchrechnen: Auflösung und Horizont, Kalibrierung, Prognoseunsicherheit, rollierende Optimierung.

    python run_extended.py            -> results/erweiterung.json
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

from twin import forecast
from twin.calibrate import fit, simulate_measurements
from twin.optimize import annual_savings, baseline, optimize
from twin.rolling import rolling_horizon
from twin.site import Site, make_series


def total_eur_a(r, s):
    return 52 * 168 / s.hours * r.energy_cost_eur + r.demand_cost_eur_a


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    site = Site()
    report = {}

    # 1. Auflösung und Horizont
    rows = []
    for label, kw in (("Woche, stündlich", {}), ("Woche, viertelstündlich", {"dt_h": 0.25}),
                      ("Monat (30 Tage), stündlich", {"days": 30})):
        s = make_series(site, **kw)
        base, opt = baseline(site, s), optimize(site, s)
        rows.append({"horizont": label, "schritte": s.n, "lastspitze_status_quo_kw": round(base.peak_kw),
                     "lastspitze_optimiert_kw": round(opt.peak_kw),
                     "einsparung_eur_a": round(annual_savings(site, base, opt, s))})
    report["aufloesung_horizont"] = rows

    # 2. Kalibrierung aus simulierten Messdaten
    s = make_series(site)
    cal = fit(simulate_measurements(site, s, optimize(site, s).cooler_el * site.cop), site)
    truth = {"thermal_cap_kwh_per_k": site.thermal_cap_kwh_per_k, "extra_loss_kw_per_k": site.extra_loss_kw_per_k,
             "cooling_base_kw": site.cooling_base_kw, "cooling_per_k_amb": site.cooling_per_k_amb,
             "cooling_shift_kw": site.cooling_shift_kw}
    report["kalibrierung"] = {"wahr": truth, "geschaetzt": {k: round(v, 2) for k, v in asdict(cal).items()},
                              "abweichung_pct": {k: round(100 * (getattr(cal, k) - v) / v, 1) for k, v in truth.items()}}

    # 3. Unsicherheit: rollierende 48-h-Optimierung gegen Prognosen, bewertet mit wahren Preisen
    base, week = baseline(site, s), optimize(site, s)
    variants = [("Status quo", None, base), ("Woche, perfekte Voraussicht", None, week)]
    rows = []
    for label, r, rmse in [(l, r, 0.0) for l, _, r in variants] + [
            (l, rr.realized, rr.forecast_rmse_eur_mwh) for l, rr in (
                ("MPC 48 h, perfekte Preisprognose", rolling_horizon(site, s)),
                ("MPC 48 h, Persistenz (Vortagspreis)", rolling_horizon(site, s, forecast.persistence(s.price, s.dt_h))),
                ("MPC 48 h, Prognosefehler σ = 15 €/MWh", rolling_horizon(site, s, forecast.noisy(s.price, 15.0))),
                ("MPC 48 h, Prognosefehler σ = 30 €/MWh", rolling_horizon(site, s, forecast.noisy(s.price, 30.0))))]:
        rows.append({"variante": label, "prognose_rmse_eur_mwh": round(rmse, 1), "lastspitze_kw": round(r.peak_kw),
                     "kosten_eur_a": round(total_eur_a(r, s)), "einsparung_eur_a": round(annual_savings(site, base, r))})
    report["unsicherheit"] = rows

    (out / "erweiterung.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for section, content in report.items():
        print(section, json.dumps(content, ensure_ascii=False)[:600])
    return report


if __name__ == "__main__":
    main()
