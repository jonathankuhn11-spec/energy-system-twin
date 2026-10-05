"""Physik und Ökonomie des LP-Modells: Bilanzen, Bänder, Zyklizität, Plausibilität der Ergebnisse."""
from dataclasses import replace

import numpy as np
import pytest

from twin.optimize import annual_savings, baseline, optimize
from twin.site import H, Site, make_series

SITE = Site()
SERIES = make_series(SITE)


@pytest.fixture(scope="module")
def runs():
    return {
        "base": baseline(SITE, SERIES),
        "pre": optimize(SITE, SERIES, use_precooling=True),
        "bat": optimize(replace(SITE, battery_kwh=1000), SERIES, use_precooling=True),
        "fixed": optimize(SITE, SERIES, use_precooling=False),
    }


def test_series_are_reproducible_and_plausible():
    s2 = make_series(SITE)
    assert np.array_equal(SERIES.price, s2.price) and np.array_equal(SERIES.process, s2.process)
    assert len(SERIES.price) == H and SERIES.pv.min() >= 0 and SERIES.pv.max() <= SITE.pv_kwp
    assert 60 < SERIES.price.mean() < 130
    annual_gwh = (SERIES.process + SERIES.cooling_th / SITE.cop).sum() * 52 / 1e6
    assert 10 < annual_gwh < 13


@pytest.mark.parametrize("key", ["pre", "bat", "fixed"])
def test_energy_balance_holds_every_hour(runs, key):
    r = runs[key]
    lhs = r.grid_import - r.grid_export - r.cooler_el - r.battery_power
    rhs = SERIES.process - SERIES.pv
    assert np.allclose(lhs, rhs, atol=1e-6)


def test_temperature_stays_inside_the_band(runs):
    r = runs["pre"]
    assert r.temperature.min() >= SITE.t_min - 1e-6 and r.temperature.max() <= SITE.t_max + 1e-6
    assert np.allclose(runs["fixed"].temperature, SITE.t_ref, atol=1e-6)


def test_battery_is_cyclic_and_within_limits(runs):
    r = runs["bat"]
    charge, discharge = np.clip(r.battery_power, 0, None), np.clip(-r.battery_power, 0, None)
    assert abs(np.sum(SITE.battery_eta * charge - discharge / SITE.battery_eta)) < 1e-6   # Ende = Anfang
    assert r.battery_soc.min() >= 0.05 * 1000 - 1e-6 and r.battery_soc.max() <= 0.95 * 1000 + 1e-6
    assert np.abs(r.battery_power).max() <= 1000 * SITE.battery_c_rate + 1e-6
    assert np.allclose(runs["pre"].battery_power, 0)


def test_cooler_respects_its_rating(runs):
    for key in ("pre", "bat"):
        assert runs[key].cooler_el.max() <= SITE.cooler_el_max_kw + 1e-6
        assert runs[key].cooler_el.min() >= -1e-9


def test_peak_is_the_maximum_grid_import(runs):
    for key in ("base", "pre", "bat"):
        assert runs[key].peak_kw == pytest.approx(runs[key].grid_import.max())
        assert runs[key].demand_cost_eur_a == pytest.approx(runs[key].peak_kw * SITE.demand_charge_eur_kw_a)


def test_fixed_temperature_reproduces_the_baseline(runs):
    assert runs["fixed"].energy_cost_eur == pytest.approx(runs["base"].energy_cost_eur, rel=1e-3)
    assert runs["fixed"].peak_kw == pytest.approx(runs["base"].peak_kw, rel=1e-3)


def test_precooling_cuts_peak_and_cost_and_battery_adds_little(runs):
    base, pre, bat = runs["base"], runs["pre"], runs["bat"]
    assert pre.peak_kw < 0.8 * base.peak_kw
    sav_pre, sav_bat = annual_savings(SITE, base, pre), annual_savings(SITE, base, bat)
    assert sav_pre > 90_000
    assert sav_bat > sav_pre
    assert sav_bat - sav_pre < 0.25 * sav_pre          # Kühlhaus ist der günstigere Speicher


def test_dunkelflaute_scenario_has_price_spikes_and_less_pv():
    df = make_series(SITE, "dunkelflaute")
    assert df.price.max() > 300 and df.price.mean() > SERIES.price.mean()
    assert df.pv.sum() < 0.5 * SERIES.pv.sum()
    r = optimize(SITE, df)
    assert r.temperature.min() >= SITE.t_min - 1e-6 and r.temperature.max() <= SITE.t_max + 1e-6
    assert annual_savings(SITE, baseline(SITE, df), r) > 0
