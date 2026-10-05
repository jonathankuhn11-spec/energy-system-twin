# Energiesystem-Zwilling: Industriestandort mit Kühlhaus

![Tests](https://github.com/jonathankuhn11-spec/energy-system-twin/actions/workflows/tests.yml/badge.svg)
**Live-Demo:** <https://jonathankuhn11-spec.github.io/energy-system-twin/>

Digitaler Zwilling eines fiktiven Industriestandorts (Molkerei mit Kühllager, rund 11,8 GWh Strom pro Jahr).
Das Modell bildet Prozesslast, Kälteanlage mit dem Kühlhaus als thermischem Speicher, PV-Dachanlage und
optionalen Batteriespeicher ab und optimiert den Einsatz als lineares Programm. Dazu kommt ein
Datenqualitätsmodul, das einen RLM-Lastgang vor der Kalibrierung bereinigt: Lücken füllen, Wandlersprung
per Change-Point-Erkennung korrigieren.

**Stack:** Python, SciPy (HiGHS), NumPy, pandas, Matplotlib · Web-Demo in HTML/JavaScript mit Chart.js ·
**Status:** Modell, Optimierung, Datenqualität und Demo stehen; Plattform-Variante (Ingestion, Datenbank, API, Scheduler) folgt

## Ergebnis

Eine repräsentative Woche, hochgerechnet auf das Jahr (Arbeitskosten × 52, Leistungspreis 120 €/kW·a auf die Wochenspitze):

| Szenario | Lastspitze | Einsparung p. a. |
| --- | --- | --- |
| A Status quo: Thermostat 4 °C, kein Speicher | 1.879 kW | – |
| B Vorkühlung im Band 2–6 °C | 1.355 kW | 106 k€ |
| C Vorkühlung + Batterie 1.000 kWh | 1.293 kW | 125 k€ |

**Kernbefund: Das Kühlhaus ist der günstigere Speicher.** Vorkühlung allein kappt die Lastspitze um 28 %
und bringt den Großteil der Einsparung. Eine Batterie liefert darüber hinaus wenig:

| Batterie | Mehrwert gegenüber B | Capex (350 €/kWh) | Amortisation |
| --- | --- | --- | --- |
| 250 kWh | 5,6 k€/a | 88 k€ | 15,7 a |
| 1.000 kWh | 18,3 k€/a | 350 k€ | 19,1 a |
| 3.000 kWh | 28,2 k€/a | 1.050 k€ | 37,2 a |

Der häufige Rechenfehler wäre, den Batteriewert gegen den Status quo statt gegen die Vorkühlung zu messen.
Dann sähe die Batterie nach 125 k€/a aus, obwohl 106 k€ davon das Kühlhaus bringt. Im Szenario Dunkelflaute
(Preisspitzen über 300 €/MWh, wenig PV) bleibt das Bild gleich: 101 k€/a durch Vorkühlung, 121 k€/a mit Batterie.

![Fahrplan](results/fahrplan.png)

Alle Zahlen erzeugt `run.py` in rund 20 Sekunden; Rohwerte in [`results/report.json`](results/report.json),
Fahrplan in [`results/fahrplan.csv`](results/fahrplan.csv).

## Live-Demo

Die [Web-Demo](https://jonathankuhn11-spec.github.io/energy-system-twin/) zeigt denselben Standort interaktiv:
animiertes Anlagenschema mit Energieflüssen je Stunde, Regler für Vorkühlung, Temperaturband, Batterie, PV,
Leistungspreis und Marktszenario. Im Reiter „Arbeitstag" liegen vier Kundenanfragen (Werkleiter, Energiemanagerin,
CFO, Schichtleitung), die sich live gegen das Modell prüfen, etwa: Lastspitze unter 1.500 kW ohne Batterie,
oder die CFO-Frage, ob sich 1.000 kWh für 350 k€ lohnen.

Die Demo optimiert im Browser per dynamischer Programmierung über diskretisierte Speicherzustände (Kühlhaus
und Batterie nacheinander), damit sie ohne Server läuft. Das Python-Projekt löst dasselbe Problem exakt als
lineares Programm; die Ergebnisse liegen nah beieinander, die LP-Zahlen sind die verbindlichen.

## Modell

Stündliche Auflösung, Horizont eine Woche (168 Stunden). Entscheidungsvariablen je Stunde: Kälteleistung
`Qc`, Kühlhaustemperatur `T`, Batterie laden `ch` und entladen `dis`, Energieinhalt `E`, Netzbezug `gi`,
Einspeisung `ge`; dazu die Lastspitze `P` als eine Variable für die ganze Woche.

Zielfunktion:

```
min  Σ gi·(Spot + Netzentgelt) − Σ ge·Spot·0,9 + P·Leistungspreis
```

Nebenbedingungen:

- **Energiebilanz je Stunde:** `gi − ge − Qc/COP − ch + dis = Prozesslast − PV`
- **Kühlhaus als thermisches Einknotenmodell:** `C·(T[t+1] − T[t]) = Q_Bedarf[t] + a·(T_ref − T[t]) − Qc[t]`.
  `C` ist die thermische Kapazität von Kühlgut und Gebäude, der Term `a·(T_ref − T)` der zusätzliche
  Wärmeeintrag, wenn tiefer als der Sollwert gekühlt wird. Vorkühlen kostet also messbar Energie.
- **Batterie:** `E[t+1] = E[t] + η·ch − dis/η`, Ladezustand zwischen 5 % und 95 %, Leistung bis C-Rate 0,5
- **Grenzen:** Temperaturband 2 bis 6 °C, Kälteanlage bis 900 kW elektrisch, `gi[t] ≤ P`
- **Zyklisch:** Temperatur und Ladezustand am Wochenende gleich dem Wochenanfang. Ohne diese Bedingung
  würde der Optimierer am Horizontende Energie „leihen" und die Einsparung überschätzen.

Das Problem hat 1.177 Variablen, 504 Gleichungen und 168 Ungleichungen; HiGHS löst es in unter einer Sekunde. Der Status quo wird
nicht optimiert, sondern simuliert: Thermostat auf 4 °C, kein Speicher, keine Preisführung.

## Datenqualität

Ein RLM-Lastgang (15-Minuten-Werte, drei Wochen) wird mit drei typischen Fehlern präpariert: zwei kurze Lücken,
eine Lücke von zehn Stunden und ein Wandlertausch zur Wochenmitte, ab dem der Zähler 25 % zu viel misst.
`twin/data_quality.py` bereinigt in drei Schritten:

1. **Sprung finden:** Saisonbereinigung über das Wochentag-Uhrzeit-Profil, dann binäre Segmentierung der
   logarithmierten Residuen (CUSUM-Statistik). Ohne Saisonbereinigung würde jeder Schichtwechsel wie ein Sprung aussehen.
2. **Sprung korrigieren:** Faktor als Median der Differenz je Wochentag-Uhrzeit-Slot vor und nach dem Sprung.
   Gefunden: Faktor 1,248 exakt an der richtigen Viertelstunde (wahr: 1,25).
3. **Lücken füllen:** bis eine Stunde linear, längere über das Wochenprofil.

Jeder Eingriff landet im Protokoll. Die bereinigte Reihe weicht im Mittel 0,15 % von der Wahrheit ab.

## Reproduzieren

Voraussetzung: Python 3.11 oder neuer.

Windows (PowerShell):

```powershell
git clone https://github.com/jonathankuhn11-spec/energy-system-twin.git
cd energy-system-twin
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

macOS und Linux: statt der vierten Zeile `source .venv/bin/activate`.

| Was | Befehl |
| --- | --- |
| Normalwoche | `python run.py` |
| Dunkelflaute | `python run.py --scenario dunkelflaute --out results/dunkelflaute` |
| Andere Batteriegröße für Szenario C | `python run.py --battery 1500` |
| Tests | `python -m pytest -q` |
| Web-Demo lokal | `docs/index.html` im Browser öffnen |

Die 17 Tests prüfen die Physik des Modells (Energiebilanz in jeder Stunde, Temperaturband, zyklische Batterie,
Anlagengrenzen), die Ökonomie (Vorkühlung senkt Spitze und Kosten, Batterie bringt wenig dazu) und die
Datenbereinigung (Sprung an der richtigen Stelle, Faktor auf 1 % genau, Lücken gefüllt, Abweichung unter 0,5 %).

## Projektstruktur

```
twin/
  site.py            Standortparameter, synthetische Zeitreihen (Spot, PV, Prozesslast, Kältebedarf)
  optimize.py        Status quo und LP-Einsatzoptimierung
  data_quality.py    Lastgang-Bereinigung: Change-Point-Erkennung, Niveaukorrektur, Lücken
run.py               Szenarien, Batterie-Dimensionierung, Report, Fahrplan, Abbildung
docs/index.html      Web-Demo (GitHub Pages)
tests/               17 Tests
results/             Report, Fahrplan und Abbildung der Normalwoche; Dunkelflaute in results/dunkelflaute/
```

## Grenzen

- **Stündlich statt viertelstündlich.** Der Leistungspreis fällt real auf die höchste Viertelstunde des Jahres;
  hier steht die Wochenspitze stellvertretend.
- **Perfekte Voraussicht.** Preise, PV und Lasten der Woche sind bekannt. Ein Betrieb bräuchte Prognosen und
  eine rollierende Optimierung.
- **Konstanter COP.** Real hängt die Leistungszahl von Außen- und Verdampfungstemperatur ab.
- **Einknotenmodell.** Das Kühlhaus hat eine Temperatur; Schichtung, Türöffnungen und Kühlgutwechsel fehlen.
- **Bis an den Bandrand.** Der Optimierer nutzt 2 und 6 °C voll aus; im Betrieb bliebe eine Sicherheitsmarge.
- **Synthetische Daten.** Standort, Lastgang und Fehler sind erfunden, in plausiblen Größenordnungen.

## Nächste Schritte

1. Plattform-Variante: Ingestion echter Day-Ahead-Preise und Wetterdaten, Zeitreihen-Datenbank, REST-API,
   Scheduler mit rollierender Optimierung über 48 Stunden
2. Viertelstündliche Auflösung und Monats-Peak statt Wochen-Peak
3. Kalibrierung des Kühlhaus-Modells aus Messdaten per Regression
4. Unsicherheit: Optimierung gegen Preisprognosen statt gegen bekannte Preise

## Lizenz

MIT, siehe [LICENSE](LICENSE). Frischwerk Münsterland ist ein erfundener Kunde.
