"""Kalibrierung des Kühlhaus-Modells aus Messdaten per linearer Regression.

Gemessen werden im Betrieb: Kühlhaustemperatur, elektrische Leistung der Kälteanlage,
Außentemperatur und Schichtstatus. Unbekannt sind die Modellparameter
  C   thermische Kapazität [kWh/K]        a   Mehr-Wärmeeintrag je K unter t_ref [kW/K]
  b0  Grundlast Kältebedarf [kW]          b1  Außentemperatur-Anteil [kW/K]      b2  Schichtzuschlag [kW]

Die Dynamik C·dT/dt = b0 + b1·(T_amb−10) + b2·Schicht + a·(T_ref−T) − Qc wird in
integrierter Form geschätzt (Integralmethode):
  T[t] − T[0] = Σ dt·[ b0/C + (b1/C)·(T_amb−10) + (b2/C)·Schicht + (a/C)·(T_ref−T) − (1/C)·Qc ]
Das ist linear in den fünf Koeffizienten b0/C, b1/C, b2/C, a/C und 1/C. Gegenüber der
Regression auf Temperaturdifferenzen (ΔT/dt) ist die Integralmethode robust gegen
Sensorrauschen, weil das Rauschen nicht durch dt geteilt wird, sondern sich in den Summen
herausmittelt. Voraussetzung: Die Temperatur muss sich im Datensatz tatsächlich bewegen
(Vorkühlung), sonst ist die Kapazität nicht identifizierbar.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .site import Site, Series, shift_mask, time_grid


@dataclass
class Calibration:
    thermal_cap_kwh_per_k: float
    extra_loss_kw_per_k: float
    cooling_base_kw: float
    cooling_per_k_amb: float
    cooling_shift_kw: float
    rmse_fit_k: float           # Residuum der Regression (Temperatur)
    rmse_sim_k: float           # Temperaturfehler einer Simulation mit den geschätzten Parametern
    n_samples: int

    def as_site(self, site: Site) -> Site:
        from dataclasses import replace
        return replace(site, thermal_cap_kwh_per_k=self.thermal_cap_kwh_per_k,
                       extra_loss_kw_per_k=self.extra_loss_kw_per_k, cooling_base_kw=self.cooling_base_kw,
                       cooling_per_k_amb=self.cooling_per_k_amb, cooling_shift_kw=self.cooling_shift_kw)


def simulate_measurements(site: Site, s: Series, cooling_th, seed: int = 11, t0: float = None,
                          noise_t_k: float = 0.1, noise_q_rel: float = 0.02) -> pd.DataFrame:
    """Erzeugt einen Messdatensatz: wahre Dynamik plus Sensorrauschen."""
    rng = np.random.default_rng(seed)
    n, dt = s.n, s.dt_h
    C, a = site.thermal_cap_kwh_per_k, site.extra_loss_kw_per_k
    T = np.empty(n + 1)
    T[0] = site.t_ref if t0 is None else t0
    for t in range(n):
        T[t + 1] = T[t] + dt * (s.cooling_th[t] + a * (site.t_ref - T[t]) - cooling_th[t]) / C
    _, hod, dow = time_grid(int(round(s.hours / 24)), dt)
    return pd.DataFrame({
        "t_room_c": T[:n] + rng.normal(0, noise_t_k, n),
        "t_room_next_c": T[1:] + rng.normal(0, noise_t_k, n),
        "cooler_el_kw": cooling_th / site.cop * (1 + rng.normal(0, noise_q_rel, n)),
        "t_amb_c": s.t_amb,
        "shift": shift_mask(hod, dow).astype(float),
        "dt_h": dt,
    })


def fit(df: pd.DataFrame, site: Site) -> Calibration:
    """Schätzt die fünf Parameter per Integralmethode und prüft sie per Simulation."""
    dt = df["dt_h"].to_numpy()
    t_room = df["t_room_c"].to_numpy()
    q = df["cooler_el_kw"].to_numpy() * site.cop
    amb, shift = df["t_amb_c"].to_numpy() - 10, df["shift"].to_numpy()
    # Zustand am Ende jedes Schritts; Summen laufen bis einschließlich Schritt t
    y = df["t_room_next_c"].to_numpy() - t_room[0]
    X = np.column_stack([np.cumsum(dt), np.cumsum(dt * amb), np.cumsum(dt * shift),
                         np.cumsum(dt * (site.t_ref - t_room)), -np.cumsum(dt * q)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    inv_c = float(beta[4])
    if inv_c <= 0:
        raise ValueError("Kapazität nicht identifizierbar: Temperatur bewegt sich nicht genug")
    C = 1 / inv_c
    b0, b1, b2, a = (float(beta[i]) * C for i in range(4))
    rmse_fit = float(np.sqrt(np.mean((X @ beta - y) ** 2)))

    # Validierung: Temperatur mit den geschätzten Parametern nachsimulieren
    n = len(df)
    T = np.empty(n + 1)
    T[0] = t_room[0]
    demand = b0 + b1 * amb + b2 * shift
    for t in range(n):
        T[t + 1] = T[t] + dt[t] * (demand[t] + a * (site.t_ref - T[t]) - q[t]) / C
    rmse_sim = float(np.sqrt(np.mean((T[1:] - df["t_room_next_c"].to_numpy()) ** 2)))
    return Calibration(C, a, b0, b1, b2, rmse_fit, rmse_sim, n)
