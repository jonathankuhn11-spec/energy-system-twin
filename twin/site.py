"""Standortparameter und synthetische Zeitreihen (stündlich, 1 Woche).

Alle Werte sind fiktiv, aber in plausiblen Größenordnungen für eine
mittelgroße Molkerei mit angeschlossenem Kühllager (ca. 11,8 GWh/a).
"""
from dataclasses import dataclass
import numpy as np

H = 168  # Stunden pro Woche


@dataclass
class Site:
    name: str = "Frischwerk Münsterland GmbH"
    pv_kwp: float = 1500.0
    cooler_el_max_kw: float = 900.0       # el. Nennleistung Kälteanlage (NH3)
    cop: float = 3.0                      # Leistungszahl Kälte
    t_min: float = 2.0                    # Kühlhaus-Sollband [°C]
    t_max: float = 6.0
    t_ref: float = 4.0                    # Baseline-Sollwert (Thermostat)
    thermal_cap_kwh_per_k: float = 1500.0  # thermische Kapazität Kühlgut + Gebäude
    extra_loss_kw_per_k: float = 35.0     # Mehr-Wärmeeintrag je K unter t_ref
    battery_kwh: float = 0.0
    battery_c_rate: float = 0.5
    battery_eta: float = 0.95             # Wirkungsgrad je Richtung
    grid_fee_eur_mwh: float = 110.0       # Netzentgelt Arbeit, Umlagen, Steuern
    demand_charge_eur_kw_a: float = 120.0  # Leistungspreis
    feed_in_share: float = 0.9            # Anteil Spotpreis bei Einspeisung (DV)
    battery_capex_eur_kwh: float = 350.0


@dataclass
class Series:
    price: np.ndarray        # €/MWh Day-Ahead
    pv: np.ndarray           # kW
    process: np.ndarray      # kW el.
    cooling_th: np.ndarray   # kW thermischer Kältebedarf
    t_amb: np.ndarray        # °C


def make_series(site: Site, scenario: str = "normal", seed: int = 7) -> Series:
    rng = np.random.default_rng(seed)
    t = np.arange(H)
    hod, dow = t % 24, t // 24  # 0 = Montag

    # Day-Ahead-Preis: Morgen-/Abendspitze, Solartal mittags
    shape = (18 * np.exp(-((hod - 8) ** 2) / 6) + 30 * np.exp(-((hod - 19) ** 2) / 5)
             - 35 * np.exp(-((hod - 13) ** 2) / 8))
    weekend = np.where(dow >= 5, -15, 0)
    price = 95 + shape + weekend + rng.normal(0, 6, H)
    if scenario == "dunkelflaute":
        price = price * 1.6 + 40
        for d in (2, 3):  # Mi/Do Abendspitzen > 300 €/MWh
            price[d * 24 + 17: d * 24 + 21] += 190

    # PV: Glockenkurve 6–20 Uhr, Tages-Bewölkungsfaktor
    cloud = rng.uniform(0.35, 1.0, 7)
    if scenario == "dunkelflaute":
        cloud *= 0.25
    bell = np.clip(np.sin((hod - 6) / 14 * np.pi), 0, None) ** 1.5
    pv = site.pv_kwp * 0.72 * bell * np.repeat(cloud, 24)

    # Prozesslast: Zweischichtbetrieb Mo–Sa 06–22 Uhr, CIP-Reinigung 22–23 Uhr
    shift = (hod >= 6) & (hod < 22) & (dow < 6)
    process = np.where(shift, 950, 420) + rng.normal(0, 40, H)
    process[(hod == 6) & (dow < 6)] += 250   # Anfahren Abfüllung
    process[(hod == 22) & (dow < 6)] += 180  # CIP
    process[dow == 6] = 380

    # Kältebedarf: Grundlast + Außentemperatur + Türöffnungen im Schichtbetrieb
    t_amb = 14 + 6 * np.sin((hod - 9) / 24 * 2 * np.pi)
    cooling_th = 1450 + 45 * (t_amb - 10) + np.where(shift, 420, 0)

    return Series(price, pv, process, cooling_th, t_amb)
