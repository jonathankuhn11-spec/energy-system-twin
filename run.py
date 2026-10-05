"""Startet alle Szenarien und schreibt Ergebnisse nach results/.

    python run.py                          # Normalwoche
    python run.py --scenario dunkelflaute  # Preisspitzen, wenig PV
    python run.py --battery 1500           # andere Batteriegröße für Szenario C
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from twin.site import Site, make_series, H
from twin.optimize import baseline, optimize, annual_savings
from twin.data_quality import make_dirty_load, clean_load


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="normal", choices=["normal", "dunkelflaute"])
    ap.add_argument("--battery", type=float, default=1000, help="Batteriegröße kWh für Szenario C")
    ap.add_argument("--out", default="results", help="Zielordner")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    site = Site()
    s = make_series(site, args.scenario)
    base = baseline(site, s)
    pre = optimize(site, s, use_precooling=True)
    bat_site = replace(site, battery_kwh=args.battery)
    bat = optimize(bat_site, s, use_precooling=True)

    def kpi(name, r, ref):
        return {"szenario": name, "lastspitze_kw": round(r.peak_kw),
                "arbeitskosten_woche_eur": round(r.energy_cost_eur),
                "leistungspreis_eur_a": round(r.demand_cost_eur_a),
                "einsparung_eur_a": round(annual_savings(site, ref, r)),
                "t_min_c": round(float(r.temperature.min()), 2),
                "t_max_c": round(float(r.temperature.max()), 2)}

    kpis = [kpi("A Status quo", base, base), kpi("B Vorkühlung", pre, base),
            kpi(f"C Vorkühlung + Batterie {args.battery:.0f} kWh", bat, base)]

    # Batterie-Dimensionierung: Amortisation über Größe, bewertet als Mehrwert gegenüber B
    sizing = []
    pre_sav = annual_savings(site, base, pre)
    for kwh in [250, 500, 750, 1000, 1500, 2000, 3000]:
        r = optimize(replace(site, battery_kwh=kwh), s)
        extra = annual_savings(site, base, r) - pre_sav
        capex = kwh * site.battery_capex_eur_kwh
        sizing.append({"kwh": kwh, "mehrwert_eur_a": round(extra), "capex_eur": round(capex),
                       "amortisation_a": round(capex / extra, 1) if extra > 0 else None})

    clean, dirty = make_dirty_load()
    fixed, log = clean_load(dirty)
    mape = float((abs(fixed - clean) / clean).mean() * 100)

    report = {"standort": site.name, "szenario": args.scenario, "kpis": kpis,
              "batterie_dimensionierung": sizing,
              "datenqualitaet": {"protokoll": log, "abweichung_zur_wahrheit_mape_pct": round(mape, 2)}}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    t = pd.date_range("2026-10-05", periods=H, freq="h")
    pd.DataFrame({"spot_eur_mwh": s.price, "pv_kw": s.pv, "prozess_kw": s.process,
                  "netz_status_quo_kw": base.grid_import, "netz_optimiert_kw": bat.grid_import,
                  "kaelte_el_kw": bat.cooler_el, "temperatur_c": bat.temperature,
                  "batterie_soc_kwh": bat.battery_soc}, index=t).round(1).to_csv(out / "fahrplan.csv")

    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(t, s.price, color="#B4436C"); ax[0].set_ylabel("Spot €/MWh")
    ax[1].plot(t, base.grid_import, label="Status quo", color="#999")
    ax[1].plot(t, bat.grid_import, label="Optimiert", color="#1F6F8B")
    ax[1].fill_between(t, s.pv, alpha=.25, color="#E8A317", label="PV")
    ax[1].set_ylabel("kW"); ax[1].legend(loc="upper right")
    ax[2].plot(t, bat.temperature, color="#2B8CA8")
    ax[2].axhspan(site.t_min, site.t_max, alpha=.08, color="#2B8CA8")
    ax[2].set_ylabel("Kühlhaus °C")
    fig.suptitle(f"{site.name}: Netzbezug und Kühlhaus ({args.scenario})")
    fig.tight_layout(); fig.savefig(out / "fahrplan.png", dpi=130); plt.close(fig)

    for k in kpis:
        print(k)
    print("Batterie:", sizing)
    print("Datenqualität:", log, f"MAPE {mape:.2f} %")
    return report


if __name__ == "__main__":
    main()
