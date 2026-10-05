"""Auflösung und Horizont, Kalibrierung, Prognoseunsicherheit, rollierende Optimierung."""
from dataclasses import replace

import numpy as np
import pytest

from twin import forecast
from twin.calibrate import fit, simulate_measurements
from twin.optimize import annual_savings, baseline, evaluate_schedule, optimize
from twin.rolling import rolling_horizon
from twin.site import Site, make_series

SITE = Site()


def test_quarter_hourly_week_matches_hourly_economics():
    s_h, s_q = make_series(SITE), make_series(SITE, dt_h=0.25)
    assert s_q.n == 4 * s_h.n and s_q.hours == s_h.hours == 168
    base_h, base_q = baseline(SITE, s_h), baseline(SITE, s_q)
    assert base_q.energy_cost_eur == pytest.approx(base_h.energy_cost_eur, rel=0.03)
    opt_q = optimize(SITE, s_q)
    assert opt_q.temperature.min() >= SITE.t_min - 1e-6 and opt_q.temperature.max() <= SITE.t_max + 1e-6
    assert opt_q.peak_kw < 0.8 * base_q.peak_kw
    lhs = opt_q.grid_import - opt_q.grid_export - opt_q.cooler_el - opt_q.battery_power
    assert np.allclose(lhs, s_q.process - s_q.pv, atol=1e-6)


def test_monthly_horizon_prices_the_monthly_peak():
    s_m = make_series(SITE, days=30)
    assert s_m.n == 720 and s_m.hours == 720
    base, opt = baseline(SITE, s_m), optimize(SITE, s_m)
    assert opt.peak_kw == pytest.approx(opt.grid_import.max())
    assert opt.peak_kw < 0.8 * base.peak_kw
    savings = annual_savings(SITE, base, opt, s_m)
    assert 80_000 < savings < 140_000                       # gleiche Größenordnung wie die Wochenrechnung


def test_window_mode_respects_start_and_terminal_state():
    s = make_series(replace(SITE, battery_kwh=1000))
    site = replace(SITE, battery_kwh=1000)
    r = optimize(site, s, start=(5.0, 300.0))
    assert r.temperature_path[0] == pytest.approx(5.0) and r.soc_path[0] == pytest.approx(300.0)
    assert r.temperature_path[-1] <= site.t_ref + 1e-6
    assert r.soc_path[-1] >= 300.0 - 1e-6


def test_evaluate_schedule_reproduces_the_optimizer():
    s = make_series(SITE)
    opt = optimize(SITE, s)
    re = evaluate_schedule(SITE, s, opt.cooler_el * SITE.cop, opt.battery_power,
                           t0=opt.temperature_path[0], e0=opt.soc_path[0])
    assert re.energy_cost_eur == pytest.approx(opt.energy_cost_eur, rel=1e-6)
    assert np.allclose(re.temperature, opt.temperature, atol=1e-6)


def test_calibration_recovers_the_true_parameters():
    s = make_series(SITE)
    schedule = optimize(SITE, s).cooler_el * SITE.cop          # Vorkühlung bewegt die Temperatur
    df = simulate_measurements(SITE, s, schedule)
    cal = fit(df, SITE)
    assert cal.thermal_cap_kwh_per_k == pytest.approx(SITE.thermal_cap_kwh_per_k, rel=0.05)
    assert cal.extra_loss_kw_per_k == pytest.approx(SITE.extra_loss_kw_per_k, rel=0.20)
    assert cal.cooling_base_kw == pytest.approx(SITE.cooling_base_kw, rel=0.03)
    assert cal.cooling_shift_kw == pytest.approx(SITE.cooling_shift_kw, rel=0.10)
    assert cal.rmse_sim_k < 0.15 and cal.n_samples == s.n
    calibrated = cal.as_site(SITE)
    assert abs(optimize(calibrated, s).peak_kw - optimize(SITE, s).peak_kw) < 30


def test_calibration_fails_without_temperature_movement():
    s = make_series(SITE)
    df = simulate_measurements(SITE, s, baseline(SITE, s).cooler_el * SITE.cop)   # Thermostat: T konstant
    try:
        cal = fit(df, SITE)
    except ValueError:
        return                                                   # sauber erkannt
    assert abs(cal.thermal_cap_kwh_per_k - SITE.thermal_cap_kwh_per_k) > 0.2 * SITE.thermal_cap_kwh_per_k


def test_forecasts_and_rolling_horizon():
    s = make_series(SITE)
    base, week = baseline(SITE, s), optimize(SITE, s)
    total = lambda r: 52 * r.energy_cost_eur + r.demand_cost_eur_a
    pers = forecast.persistence(s.price, s.dt_h)
    assert 0 < forecast.rmse(pers, s.price) < 20
    runs = {"perfect": rolling_horizon(SITE, s), "persistence": rolling_horizon(SITE, s, pers),
            "noisy15": rolling_horizon(SITE, s, forecast.noisy(s.price, 15.0)),
            "noisy30": rolling_horizon(SITE, s, forecast.noisy(s.price, 30.0))}
    for r in runs.values():
        assert r.windows == 7
        assert r.realized.temperature.min() >= SITE.t_min - 1e-6 and r.realized.temperature.max() <= SITE.t_max + 1e-6
        assert total(r.realized) >= total(week) - 1e-6                 # die Wochenlösung ist das Optimum
        assert annual_savings(SITE, base, r.realized) > 0.7 * annual_savings(SITE, base, week)
    assert total(runs["perfect"].realized) < 1.01 * total(week)        # 48-h-Fenster kosten unter 1 %
    assert total(runs["noisy30"].realized) > total(runs["noisy15"].realized) > total(runs["perfect"].realized)
