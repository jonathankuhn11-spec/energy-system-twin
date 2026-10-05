"""Lastgang-Bereinigung: Sprung finden und korrigieren, Lücken füllen, nichts verschweigen."""
import numpy as np
import pandas as pd

from twin.data_quality import clean_load, make_dirty_load


def test_dirty_load_contains_the_injected_faults():
    clean, dirty = make_dirty_load()
    assert len(clean) == 4 * 24 * 21 and dirty.isna().sum() == 2 + 40 + 3
    jump_at = dirty.index[len(dirty) // 2]
    assert (dirty[jump_at:].dropna() / clean[jump_at:][dirty[jump_at:].notna()]).round(3).eq(1.25).all()


def test_jump_is_located_and_corrected():
    clean, dirty = make_dirty_load()
    fixed, log = clean_load(dirty)
    jump_at = dirty.index[len(dirty) // 2]
    entry = next(line for line in log if line.startswith("Niveausprung"))
    assert f"{jump_at:%d.%m. %H:%M}" in entry
    factor = float(entry.split("Faktor ")[1].split(" ")[0])
    assert abs(factor - 1.25) < 0.01
    assert not fixed.isna().any()
    assert np.abs(fixed[jump_at:] - clean[jump_at:]).mean() < 10          # ohne Korrektur wären es ~300 kW


def test_gaps_are_filled_by_length_and_logged():
    clean, dirty = make_dirty_load()
    fixed, log = clean_load(dirty)
    assert sum("linear" in line for line in log) == 2
    assert sum("Wochenprofil" in line for line in log) == 1
    long_gap = dirty.index[900:940]
    assert (abs(fixed[long_gap] - clean[long_gap]) / clean[long_gap]).mean() < 0.06


def test_cleaned_series_matches_the_truth_within_half_a_percent():
    clean, dirty = make_dirty_load()
    fixed, _ = clean_load(dirty)
    mape = (abs(fixed - clean) / clean).mean() * 100
    assert mape < 0.5


def test_clean_series_is_left_alone():
    clean, _ = make_dirty_load()
    fixed, log = clean_load(clean)
    assert log == [] and np.allclose(fixed, clean)
