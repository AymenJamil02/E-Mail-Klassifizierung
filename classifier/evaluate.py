"""Auswertung der E-Mail-Klassifizierung.

Aufruf im Container:
    docker compose exec classifier python evaluate.py            # nur ML (schnell)
    docker compose exec classifier python evaluate.py --hybrid   # ML + LLM über die API

Misst drei Dinge:
1. Kreuzvalidierung auf den Trainingsdaten (5-fach, nur ML)
2. Ergebnis auf NEUEN Testdaten (testdaten.json, nie trainiert)
3. Optional: das komplette Hybrid-System (ML + LLM) auf den Testdaten
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from classifier import KATEGORIEN, LokalerEmailKlassifikator, daten_laden

ORDNER = Path(__file__).resolve().parent
API_URL = "http://localhost:8000/classify"


def tabelle(wahr: list[str], vorhergesagt: list[str], titel: str) -> float:
    labels = list(KATEGORIEN)
    richtig = sum(w == v for w, v in zip(wahr, vorhergesagt))
    genauigkeit = richtig / len(wahr)

    print(f"\n{'=' * 64}\n{titel}\n{'=' * 64}")
    print(f"Genauigkeit (Accuracy): {genauigkeit:.0%}  ({richtig} von {len(wahr)} richtig)\n")
    print(classification_report(wahr, vorhergesagt, labels=labels, zero_division=0, digits=2))

    print("Konfusionsmatrix (Zeile = richtig, Spalte = vorhergesagt):")
    matrix = confusion_matrix(wahr, vorhergesagt, labels=labels)
    kurz = [k[:6] for k in labels]
    print("            " + "".join(f"{k:>8}" for k in kurz))
    for name, zeile in zip(kurz, matrix):
        print(f"{name:>10}  " + "".join(f"{z:>8}" for z in zeile))
    return genauigkeit


def kreuzvalidierung(daten: list[dict]) -> None:
    texte_labels = [d["erwartete_kategorie"] for d in daten]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    wahr, vorher = [], []
    for train_idx, test_idx in folds.split(daten, texte_labels):
        modell = LokalerEmailKlassifikator().trainieren([daten[i] for i in train_idx])
        for i in test_idx:
            kat, _, _ = modell.roh_vorhersagen(daten[i]["betreff"], daten[i]["inhalt"])
            wahr.append(daten[i]["erwartete_kategorie"])
            vorher.append(kat)
    tabelle(wahr, vorher, "1) Kreuzvalidierung auf Trainingsdaten (5-fach, nur ML)")


def ml_auf_testdaten(trainingsdaten: list[dict], testdaten: list[dict]) -> None:
    modell = LokalerEmailKlassifikator().trainieren(trainingsdaten)
    wahr = [t["erwartete_kategorie"] for t in testdaten]
    roh, final, pruefung = [], [], 0
    for t in testdaten:
        e = modell.klassifizieren(t["betreff"], t["inhalt"])
        roh.append(e.modell_kategorie)
        final.append(e.kategorie)
        pruefung += e.menschliche_pruefung

    tabelle(wahr, roh, "2a) NEUE Testdaten – reines ML-Modell")

    sicher = [(w, f) for w, f in zip(wahr, final) if f != "Unklar"]
    print(f"\n2b) Mit Sicherheitslogik: {len(sicher)} von {len(wahr)} Mails sicher entschieden, "
          f"{len(wahr) - len(sicher)} als 'Unklar' markiert.")
    if sicher:
        r = sum(w == f for w, f in sicher)
        print(f"    Davon richtig: {r} von {len(sicher)} ({r / len(sicher):.0%})")
    print(f"    Zur menschlichen Prüfung markiert: {pruefung} von {len(wahr)}")


def hybrid_auf_testdaten(testdaten: list[dict]) -> None:
    print(f"\n3) Hybrid-System (ML + LLM) über {API_URL} – dauert einige Minuten …")
    wahr, vorher, quellen, fehler = [], [], Counter(), []
    start = time.time()
    for n, t in enumerate(testdaten, 1):
        body = json.dumps({
            "betreff": t["betreff"], "inhalt": t["inhalt"],
            "mit_llm_entwurf": False, "mit_llm_klassifikation": True,
        }).encode()
        req = urllib.request.Request(API_URL, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            e = json.loads(r.read())
        wahr.append(t["erwartete_kategorie"])
        vorher.append(e["kategorie"])
        quellen[e["klassifikation_quelle"]] += 1
        if e["kategorie"] != t["erwartete_kategorie"]:
            fehler.append((t["id"], t["betreff"], t["erwartete_kategorie"], e["kategorie"], e["klassifikation_quelle"]))
        print(f"   {n:>2}/{len(testdaten)} {t['betreff'][:35]:<35} -> {e['kategorie']:<10} ({e['klassifikation_quelle']})")

    tabelle(wahr, vorher, "3) NEUE Testdaten – Hybrid-System (ML + LLM)")
    print(f"\nEntschieden durch: {dict(quellen)}  |  Dauer: {time.time() - start:.0f} s")
    if fehler:
        print("\nFalsch klassifiziert:")
        for f in fehler:
            print(f"   #{f[0]} '{f[1]}': erwartet {f[2]}, erkannt {f[3]} (Quelle: {f[4]})")


def main() -> None:
    trainingsdaten = daten_laden(ORDNER / "trainingsdaten.json")
    testdaten = daten_laden(ORDNER / "testdaten.json")
    print(f"Trainingsdaten: {len(trainingsdaten)} | Testdaten: {len(testdaten)} | "
          f"Verteilung Test: {dict(Counter(t['erwartete_kategorie'] for t in testdaten))}")

    kreuzvalidierung(trainingsdaten)
    ml_auf_testdaten(trainingsdaten, testdaten)
    if "--hybrid" in sys.argv:
        hybrid_auf_testdaten(testdaten)
    else:
        print("\nTipp: Mit --hybrid wird zusätzlich das komplette System inkl. LLM gemessen.")


if __name__ == "__main__":
    main()
