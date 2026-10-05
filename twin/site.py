"""Standortparameter und synthetische Zeitreihen.

Alle Werte sind fiktiv, aber in plausiblen Größenordnungen für eine
mittelgroße Molkerei mit angeschlossenem Kühllager (ca. 11,8 GWh/a).
Die Zeitreihen lassen sich in beliebiger Auflösung (Stunde, Viertelstunde)
und Länge (Woche, Monat) erzeugen; Standard ist eine Woche stündlich.
"""
from dataclasses import dataclass
import numpy as np

H = 168  # Stunden pro Woche (Standardhorizont)


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
    # Kältebedarf-Modell: Grundlast + Außentemperatur-Anteil + Schichtbetrieb (Türöffnungen)
    cooling_base_kw: float = 1450.0
    cooling_per_k_amb: float = 45.0
    cooling_shift_kw: float = 420.0


@dataclass
class Series:
    price: np.ndarray        # €/MWh Day-Ahead
    pv: np.ndarray           # kW
    process: np.ndarray      # kW el.
    cooling_th: np.ndarray   # kW thermischer Kältebedarf
    t_amb: np.ndarray        # °C
    dt_h: float = 1.0        # Schrittweite in Stunden

    @property
    def n(self) -> int:
        return len(self.price)

    @property
    def hours(self) -> float:
        return self.n * self.dt_h


def time_grid(days: int, dt_h: float):
    """Stunde des Tages und Wochentag (0 = Montag) je Zeitschritt."""
    n = int(round(days * 24 / dt_h))
    k = np.arange(n)
    hod = (k * dt_h) % 24
    dow = ((k * dt_h) // 24).astype(int) % 7
    return n, hod, dow


def shift_mask(hod, dow):
    """Zweischichtbetrieb Mo–Sa 06–22 Uhr."""
    return (hod >= 6) & (hod < 22) & (dow < 6)


def process_profile(site: Site, hod, dow, rng=None):
    """Prozesslast: Abfüllung, CIP-Reinigung, Druckluft; Sonntag nur Grundlast."""
    shift = shift_mask(hod, dow)
    process = np.where(shift, 950.0, 420.0)
    if rng is not None:
        process = process + rng.normal(0, 40, len(hod))
    process[(hod >= 6) & (hod < 7) & (dow < 6)] += 250   # Anfahren Abfüllung
    process[(hod >= 22) & (hod < 23) & (dow < 6)] += 180  # CIP
    process[dow == 6] = 380
    return process


def cooling_demand(site: Site, t_amb, hod, dow):
    """Kältebedarf: Grundlast + Außentemperatur + Türöffnungen im Schichtbetrieb."""
    return (site.cooling_base_kw + site.cooling_per_k_amb * (t_amb - 10)
            + np.where(shift_mask(hod, dow), site.cooling_shift_kw, 0))


def ambient_profile(hod):
    return 14 + 6 * np.sin((hod - 9) / 24 * 2 * np.pi)


def make_series(site: Site, scenario: str = "normal", seed: int = 7, days: int = 7, dt_h: float = 1.0) -> Series:
    rng = np.random.default_rng(seed)
    n, hod, dow = time_grid(days, dt_h)

    # Day-Ahead-Preis: Morgen-/Abendspitze, Solartal mittags
    shape = (18 * np.exp(-((hod - 8) ** 2) / 6) + 30 * np.exp(-((hod - 19) ** 2) / 5)
             - 35 * np.exp(-((hod - 13) ** 2) / 8))
    weekend = np.where(dow >= 5, -15, 0)
    price = 95 + shape + weekend + rng.normal(0, 6, n)
    if scenario == "dunkelflaute":
        price = price * 1.6 + 40
        price[np.isin(dow, (2, 3)) & (hod >= 17) & (hod < 21)] += 190   # Mi/Do Abendspitzen > 300 €/MWh

    # PV: Glockenkurve 6–20 Uhr, Tages-Bewölkungsfaktor
    cloud = rng.uniform(0.35, 1.0, days)
    if scenario == "dunkelflaute":
        cloud *= 0.25
    bell = np.clip(np.sin((hod - 6) / 14 * np.pi), 0, None) ** 1.5
    day_index = ((np.arange(n) * dt_h) // 24).astype(int)
    pv = site.pv_kwp * 0.72 * bell * cloud[day_index]

    process = process_profile(site, hod, dow, rng)
    t_amb = ambient_profile(hod)
    cooling_th = cooling_demand(site, t_amb, hod, dow)
    return Series(price, pv, process, cooling_th, t_amb, dt_h)
