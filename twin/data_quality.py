"""Datenqualität eines RLM-Lastgangs (Lücken, Wandler-Sprung).

Vorgehen:
  1. Sprungerkennung: saisonbereinigte log-Residuen, binäre Segmentierung.
  2. Niveaukorrektur ab dem Sprung (Wandlerfaktor).
  3. Lücken ≤ 1 h linear, längere Lücken über Wochentag-Stunden-Profil.
Jeder Eingriff wird protokolliert.
"""
import numpy as np
import pandas as pd


def make_dirty_load(seed: int = 3) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-09-07", periods=4 * 24 * 21, freq="15min")
    hod = idx.hour + idx.minute / 60
    shift = (hod >= 6) & (hod < 22) & (idx.dayofweek < 6)
    clean = pd.Series(np.where(shift, 1550, 1000) + rng.normal(0, 50, len(idx)), idx)
    dirty = clean.copy()
    jump_at = idx[len(idx) // 2]
    dirty[jump_at:] *= 1.25                           # falscher Wandlerfaktor
    dirty.iloc[300:302] = np.nan                       # kurze Lücke (30 min)
    dirty.iloc[900:940] = np.nan                       # lange Lücke (10 h)
    dirty.iloc[1700:1703] = np.nan
    return clean, dirty


def clean_load(s: pd.Series, jump_tol: float = 0.08):
    log, s = [], s.copy()

    # Saisonbereinigung (Wochentag × Uhrzeit), dann binäre Segmentierung auf log-Residuen
    key = [s.index.dayofweek, s.index.hour, s.index.minute]
    resid = np.log(s) - np.log(s).groupby(key).transform("median")
    r = resid.dropna()
    n, cs = len(r), r.cumsum().to_numpy()
    k = np.arange(1, n)
    stat = np.abs(cs[:-1] / k - (cs[-1] - cs[:-1]) / (n - k)) * np.sqrt(k * (n - k) / n)
    kb = int(stat.argmax()) + 1
    t_jump = r.index[kb]
    ls = np.log(s.dropna())  # Faktor: Median der Differenz je Wochentag-Uhrzeit-Slot
    k_of = lambda x: [x.index.dayofweek, x.index.hour, x.index.minute]
    pre, post = ls[:t_jump].iloc[:-1], ls[t_jump:]
    diff = post.groupby(k_of(post)).median() - pre.groupby(k_of(pre)).median()
    factor = float(np.exp(diff.dropna().median()))
    if abs(factor - 1) > jump_tol:
        s[t_jump:] /= factor
        log.append(f"Niveausprung bei {t_jump:%d.%m. %H:%M}, Faktor {factor:.3f} korrigiert")

    gaps = s.isna()
    groups = (gaps != gaps.shift()).cumsum()[gaps]
    profile = s.groupby([s.index.dayofweek, s.index.hour, s.index.minute]).median()
    for _, g in s[gaps].groupby(groups):
        start, end, n = g.index[0], g.index[-1], len(g)
        if n <= 4:
            s[start:end] = np.nan
            s = s.interpolate(limit_area="inside")
            log.append(f"Lücke {start:%d.%m. %H:%M} ({n * 15} min) linear gefüllt")
        else:
            for ts in g.index:
                s[ts] = profile[(ts.dayofweek, ts.hour, ts.minute)]
            log.append(f"Lücke {start:%d.%m. %H:%M} ({n * 15} min) per Wochenprofil gefüllt")
    return s, log
