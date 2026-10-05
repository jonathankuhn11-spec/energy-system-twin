"""Preisprognose aus Daten statt Persistenz: Ridge-Regression auf Kalender-, Lag- und Wetterfeatures.

Features je Zielstunde τ (Prognosehorizont 48 h ab Ursprung O):
  Kalender    Stunde des Tages (24 Indikatoren), Wochentag (7 Indikatoren)
  Lags        Preis vor 24 h, sofern zum Ursprung bekannt, sonst vor 48 h; Preis vor 168 h
  Niveau      Mittelwert der 24 Stunden vor dem Ursprung
  Wetter      Außentemperatur und Globalstrahlung (Open-Meteo-Prognose; mehr PV drückt den Mittagspreis)
Das Modell wird auf der gespeicherten Preishistorie trainiert und per Walk-Forward-Backtest gegen die
Persistenz (Vortagspreis) gemessen: Nur wenn es im Backtest besser ist, nutzt der Zyklus es.
"""
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

MIN_HISTORY_DAYS = 14
LAMBDA = 1.0


@dataclass
class Backtest:
    origins: int
    rmse_model: float
    rmse_persistence: float

    @property
    def model_is_better(self) -> bool:
        return self.origins > 0 and self.rmse_model < self.rmse_persistence


def _hourly(prices: pd.Series) -> pd.Series:
    """Viertelstundenprodukte auf Stundenmittel bringen."""
    return prices.resample("h").mean().interpolate(limit_area="inside")


def _features(targets: pd.DatetimeIndex, hist: pd.Series, origin: pd.Timestamp, weather: pd.DataFrame) -> np.ndarray:
    lag24 = hist.reindex(targets - timedelta(hours=24)).to_numpy()
    lag48 = hist.reindex(targets - timedelta(hours=48)).to_numpy()
    known = (targets - timedelta(hours=24)) <= origin
    lag = np.where(known, lag24, lag48)
    lag168 = hist.reindex(targets - timedelta(hours=168)).to_numpy()
    level = float(hist[(hist.index > origin - timedelta(hours=24)) & (hist.index <= origin)].mean())
    w = weather.reindex(targets).interpolate(limit_direction="both")
    hour = np.eye(24)[targets.hour]
    wday = np.eye(7)[targets.dayofweek]
    return np.column_stack([np.ones(len(targets)), hour, wday, lag, lag168, np.full(len(targets), level),
                            w["t_amb_c"].to_numpy(), w["ghi_w_m2"].to_numpy() / 100])


class PriceModel:
    def __init__(self):
        self.beta = None
        self.mean = None
        self.scale = None

    def fit(self, hist: pd.Series, weather: pd.DataFrame, until: pd.Timestamp) -> "PriceModel":
        """Trainingszeilen: jede Stunde der Historie, Ursprung jeweils 24 h vorher (Day-Ahead-Logik)."""
        hist = _hourly(hist[hist.index <= until])
        targets = hist.index[hist.index >= hist.index[0] + timedelta(hours=192)]
        rows, ys = [], []
        for day, group in hist.reindex(targets).groupby(targets.floor("D")):
            origin = day - timedelta(hours=11)                    # 13:00 des Vortags
            X = _features(group.index, hist, origin, weather)
            rows.append(X); ys.append(group.to_numpy())
        X, y = np.vstack(rows), np.concatenate(ys)
        ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[ok], y[ok]
        self.mean, self.scale = X.mean(axis=0), X.std(axis=0) + 1e-9
        self.mean[0], self.scale[0] = 0.0, 1.0                     # Achsenabschnitt nicht skalieren
        Z = (X - self.mean) / self.scale
        reg = LAMBDA * np.eye(Z.shape[1]); reg[0, 0] = 0
        self.beta = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
        return self

    def predict(self, targets: pd.DatetimeIndex, hist: pd.Series, origin: pd.Timestamp, weather: pd.DataFrame) -> np.ndarray:
        X = _features(targets, _hourly(hist[hist.index <= origin]), origin, weather)
        X = np.where(np.isnan(X), self.mean, X)                   # fehlende Lags auf Trainingsmittel
        return (X - self.mean) / self.scale @ self.beta


def persistence(targets: pd.DatetimeIndex, hist: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    h = _hourly(hist[hist.index <= origin])
    lag24 = h.reindex(targets - timedelta(hours=24)).to_numpy()
    lag48 = h.reindex(targets - timedelta(hours=48)).to_numpy()
    return np.where((targets - timedelta(hours=24)) <= origin, lag24, lag48)


def backtest(prices: pd.Series, weather: pd.DataFrame, horizon_h: int = 48, max_origins: int = 30) -> Backtest:
    """Walk-forward: an jedem Tag um 13:00 trainieren und die nächsten 48 h prognostizieren."""
    hist = _hourly(prices)
    first = hist.index[0] + timedelta(days=MIN_HISTORY_DAYS)
    origins = pd.date_range(first.ceil("D") + timedelta(hours=13), hist.index[-1] - timedelta(hours=horizon_h + 11),
                            freq="D")[-max_origins:]
    err_m, err_p = [], []
    for origin in origins:
        targets = pd.date_range(origin + timedelta(hours=11), periods=horizon_h, freq="h")   # ab 0:00 Folgetag
        actual = hist.reindex(targets).to_numpy()
        model = PriceModel().fit(hist, weather, origin)
        err_m.append(model.predict(targets, hist, origin, weather) - actual)
        err_p.append(persistence(targets, hist, origin) - actual)
    if not len(err_m):
        return Backtest(0, float("nan"), float("nan"))
    rm = lambda e: float(np.sqrt(np.nanmean(np.concatenate(e) ** 2)))
    return Backtest(len(origins), rm(err_m), rm(err_p))
