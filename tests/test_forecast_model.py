"""Preisprognosemodell: schlägt die Persistenz im Walk-Forward-Backtest und füllt das Fenster im Zyklus."""

import numpy as np
import pandas as pd

from backend import forecast_model
from backend.service import build_window, run_cycle
from backend.store import Store
from twin.site import Site, make_series


def synthetic_history(days: int = 40, seed: int = 3):
    """60 Tage Preise und Wetter mit Wochenstruktur, Wetterabhängigkeit und Rauschen."""
    site = Site()
    s = make_series(site, seed=seed, days=days)
    idx = pd.date_range("2026-08-01", periods=s.n, freq="h", tz="Europe/Berlin")
    ghi = s.pv / site.pv_kwp * 1000 / 0.72
    price = s.price - 0.02 * ghi + 0.8 * (s.t_amb - 14) + np.random.default_rng(seed).normal(0, 3, s.n)
    return (pd.Series(price, index=idx, name="eur_mwh"),
            pd.DataFrame({"t_amb_c": s.t_amb, "ghi_w_m2": ghi}, index=idx))


def test_model_beats_persistence_in_backtest():
    prices, weather = synthetic_history()
    bt = forecast_model.backtest(prices, weather, max_origins=10)
    assert bt.origins == 10
    assert bt.rmse_model < bt.rmse_persistence
    assert bt.rmse_model < 10


def test_prediction_has_the_right_shape_and_uses_lags():
    prices, weather = synthetic_history()
    origin = prices.index[-60]
    targets = pd.date_range(origin.ceil("h"), periods=48, freq="h")
    model = forecast_model.PriceModel().fit(prices, weather, origin)
    pred = model.predict(targets, prices, origin, weather)
    assert pred.shape == (48,) and not np.isnan(pred).any()
    assert abs(pred.mean() - prices[prices.index > origin - pd.Timedelta(days=7)].mean()) < 15


def test_cycle_switches_to_the_model_with_enough_history():
    prices, weather = synthetic_history()
    st = Store()
    st.upsert_prices(prices[:-30])                      # Historie endet 30 h vor dem Fensterende
    st.upsert_weather(weather)
    start = prices.index[-30].ceil("h")
    series, idx, filled, method = build_window(st, Site(), start, hours=48)
    assert filled > 0 and method.startswith("modell")
    r = run_cycle(st, Site(), now=start - pd.Timedelta(minutes=30))
    assert r.fill_method.startswith("modell") and r.steps == 48
    st.close()
