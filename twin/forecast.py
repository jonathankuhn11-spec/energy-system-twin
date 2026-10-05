"""Preisprognosen für die Optimierung unter Unsicherheit.

Im Betrieb kennt der Optimierer die Preise nicht: Der Day-Ahead-Markt veröffentlicht sie
gegen 13 Uhr für den Folgetag, alles danach ist Prognose. Zwei einfache Prognosen:
  persistence  Preis von vor 24 Stunden (Standard-Benchmark der Preisprognose)
  noisy        wahrer Preis plus normalverteilter Fehler (Prognosegüte als Parameter)
"""
import numpy as np


def persistence(price: np.ndarray, dt_h: float, lag_h: float = 24.0) -> np.ndarray:
    """Prognose = Preis vor lag_h Stunden (zyklisch über den Horizont)."""
    return np.roll(price, int(round(lag_h / dt_h)))


def noisy(price: np.ndarray, sigma_eur_mwh: float, seed: int = 21) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return price + rng.normal(0, sigma_eur_mwh, len(price))


def rmse(forecast: np.ndarray, actual: np.ndarray) -> float:
    return float(np.sqrt(np.mean((forecast - actual) ** 2)))
