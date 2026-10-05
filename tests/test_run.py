"""Ende-zu-Ende: der Runner schreibt Report, Fahrplan und Abbildung."""
import json

import run


def test_runner_writes_all_artifacts(tmp_path):
    report = run.main(["--out", str(tmp_path), "--battery", "500"])
    assert (tmp_path / "report.json").exists() and (tmp_path / "fahrplan.csv").exists()
    assert (tmp_path / "fahrplan.png").stat().st_size > 10_000
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["kpis"][1]["einsparung_eur_a"] == report["kpis"][1]["einsparung_eur_a"] > 0
    assert saved["kpis"][2]["szenario"].endswith("500 kWh")
    assert len(saved["batterie_dimensionierung"]) == 7
    assert saved["datenqualitaet"]["abweichung_zur_wahrheit_mape_pct"] < 0.5
