"""Zeitreihenspeicher auf DuckDB: eine Datei, kein Server, SQL mit Zeitreihenfunktionen.

Tabellen: prices (Spot), weather (Temperatur, Strahlung), schedules (jeder Optimierungslauf
mit seinem Fahrplan). Schreibzugriffe sind Upserts, damit Ingestion und Läufe wiederholbar sind.
"""
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices   (ts TIMESTAMPTZ PRIMARY KEY, eur_mwh DOUBLE, source VARCHAR);
CREATE TABLE IF NOT EXISTS weather  (ts TIMESTAMPTZ PRIMARY KEY, t_amb_c DOUBLE, ghi_w_m2 DOUBLE);
CREATE TABLE IF NOT EXISTS runs     (run_id INTEGER PRIMARY KEY, created TIMESTAMPTZ, window_start TIMESTAMPTZ,
                                     steps INTEGER, dt_h DOUBLE, peak_kw DOUBLE, energy_cost_eur DOUBLE,
                                     t_end_c DOUBLE, e_end_kwh DOUBLE, note VARCHAR);
CREATE TABLE IF NOT EXISTS schedules (run_id INTEGER, ts TIMESTAMPTZ, price_eur_mwh DOUBLE, pv_kw DOUBLE,
                                     process_kw DOUBLE, cooler_el_kw DOUBLE, battery_kw DOUBLE,
                                     temperature_c DOUBLE, grid_kw DOUBLE, PRIMARY KEY (run_id, ts));
"""


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path))
        self.con.execute("SET TimeZone = 'Europe/Berlin'")
        for stmt in SCHEMA.strip().split(";"):
            if stmt.strip():
                self.con.execute(stmt)

    def close(self):
        self.con.close()

    # --- Ingestion -------------------------------------------------------------
    def upsert_prices(self, s: pd.Series, source: str = "energy-charts") -> int:
        df = pd.DataFrame({"ts": s.index, "eur_mwh": s.to_numpy(), "source": source})
        self.con.register("df_p", df)
        self.con.execute("INSERT OR REPLACE INTO prices SELECT ts, eur_mwh, source FROM df_p")
        self.con.unregister("df_p")
        return len(df)

    def upsert_weather(self, df: pd.DataFrame) -> int:
        w = pd.DataFrame({"ts": df.index, "t_amb_c": df["t_amb_c"].to_numpy(), "ghi_w_m2": df["ghi_w_m2"].to_numpy()})
        self.con.register("df_w", w)
        self.con.execute("INSERT OR REPLACE INTO weather SELECT ts, t_amb_c, ghi_w_m2 FROM df_w")
        self.con.unregister("df_w")
        return len(w)

    # --- Lesen -----------------------------------------------------------------
    def prices(self, start=None, end=None) -> pd.Series:
        q, args = "SELECT ts, eur_mwh FROM prices", []
        if start is not None:
            q += " WHERE ts >= ?"; args.append(start)
            if end is not None:
                q += " AND ts < ?"; args.append(end)
        elif end is not None:
            q += " WHERE ts < ?"; args.append(end)
        df = self.con.execute(q + " ORDER BY ts", args).df()
        return pd.Series(df["eur_mwh"].to_numpy(), index=pd.DatetimeIndex(df["ts"]), name="eur_mwh")

    def weather(self, start=None, end=None) -> pd.DataFrame:
        q, args = "SELECT ts, t_amb_c, ghi_w_m2 FROM weather", []
        if start is not None:
            q += " WHERE ts >= ?"; args.append(start)
            if end is not None:
                q += " AND ts < ?"; args.append(end)
        df = self.con.execute(q + " ORDER BY ts", args).df()
        return df.set_index(pd.DatetimeIndex(df["ts"])).drop(columns="ts")

    def coverage(self) -> dict:
        p = self.con.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM prices").fetchone()
        w = self.con.execute("SELECT MIN(ts), MAX(ts), COUNT(*) FROM weather").fetchone()
        return {"prices": {"from": p[0], "to": p[1], "rows": p[2]}, "weather": {"from": w[0], "to": w[1], "rows": w[2]}}

    # --- Läufe -----------------------------------------------------------------
    def save_run(self, window_start, schedule: pd.DataFrame, dt_h: float, peak_kw: float, energy_cost_eur: float,
                 t_end_c: float, e_end_kwh: float, note: str = "", created: datetime | None = None) -> int:
        run_id = self.con.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM runs").fetchone()[0]
        self.con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                         [run_id, created or datetime.now().astimezone(), window_start, len(schedule), dt_h,
                          peak_kw, energy_cost_eur, t_end_c, e_end_kwh, note])
        df = schedule.reset_index().rename(columns={"index": "ts"})
        df.insert(0, "run_id", run_id)
        self.con.register("df_s", df[["run_id", "ts", "price_eur_mwh", "pv_kw", "process_kw", "cooler_el_kw",
                                      "battery_kw", "temperature_c", "grid_kw"]])
        self.con.execute("INSERT INTO schedules SELECT * FROM df_s")
        self.con.unregister("df_s")
        return run_id

    def last_run(self) -> dict | None:
        row = self.con.execute("SELECT run_id, created, window_start, steps, dt_h, peak_kw, energy_cost_eur, "
                               "t_end_c, e_end_kwh, note FROM runs ORDER BY run_id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        keys = ["run_id", "created", "window_start", "steps", "dt_h", "peak_kw", "energy_cost_eur", "t_end_c",
                "e_end_kwh", "note"]
        return dict(zip(keys, row))

    def schedule(self, run_id: int) -> pd.DataFrame:
        df = self.con.execute("SELECT ts, price_eur_mwh, pv_kw, process_kw, cooler_el_kw, battery_kw, temperature_c, "
                              "grid_kw FROM schedules WHERE run_id = ? ORDER BY ts", [run_id]).df()
        return df.set_index(pd.DatetimeIndex(df["ts"])).drop(columns="ts")
