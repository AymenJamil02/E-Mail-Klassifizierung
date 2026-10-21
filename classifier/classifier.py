
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


KATEGORIEN = ("Anfrage", "Beschwerde", "Rechnung", "Unklar")

ANTWORTVORLAGEN = {
    "Anfrage": (
        "Vielen Dank für Ihre Anfrage. Wir prüfen Ihr Anliegen und melden "
        "uns nach der fachlichen Prüfung bei Ihnen."
    ),
    "Beschwerde": (
        "Vielen Dank für Ihren Hinweis. Wir bedauern Ihre Unzufriedenheit "
        "und prüfen den geschilderten Vorgang."
    ),
    "Rechnung": (
        "Vielen Dank für Ihre Nachricht zur Abrechnung. Die Unterlagen "
        "werden geprüft; bei Rückfragen melden wir uns bei Ihnen."
    ),
    "Unklar": (
        "Vielen Dank für Ihre Nachricht. Wir prüfen zunächst die "
        "Zuständigkeit und den weiteren Bearbeitungsweg."
    ),
}

SICHERHEITSSIGNALE = {
    "Anfrage": (
        "wann",
        "warum",
        "welche",
        "wie",
        "wo",
        "wer",
        "kann",
        "?",
    ),
    "Beschwerde": (
        "beschwerde",
        "unzufrieden",
        "falsch",
        "fehler",
        "warte",
        "keine antwort",
        "abgesagt",
        "storniert",
        "respektlos",
    ),
    "Rechnung": (
        "rechnung",
        "betrag",
        "zahlung",
        "überweisen",
        "kosten",
        "euro",
        "abrechnung",
        "mahnung",
    ),
}


@dataclass(frozen=True)
class Klassifizierung:
    """Strukturiertes Ergebnis für Oberfläche, Tests und spätere Integration."""

    kategorie: str
    modell_kategorie: str
    modellscore: float
    abstand_zum_zweiten: float
    signal_kategorien: tuple[str, ...]
    menschliche_pruefung: bool
    begruendung: str
    antwortentwurf: str

    def als_dict(self) -> dict[str, Any]:
        return asdict(self)


def email_text(betreff: str, inhalt: str) -> str:
    """Betreff und Inhalt normalisieren und zu einem Text verbinden."""

    betreff = " ".join(betreff.split())
    inhalt = " ".join(inhalt.split())

    if not betreff and not inhalt:
        raise ValueError("Betreff und Inhalt dürfen nicht beide leer sein.")

    return f"{betreff} {inhalt}".strip()


def signal_kategorien(text: str) -> tuple[str, ...]:
    """Einfache, transparente Signale für Mehrdeutigkeit erkennen."""

    text = text.lower()
    gefunden = []

    for kategorie, signale in SICHERHEITSSIGNALE.items():
        for signal in signale:
            if signal == "?" or " " in signal:
                passt = signal in text
            else:
                passt = re.search(
                    rf"\b{re.escape(signal)}\b",
                    text,
                ) is not None

            if passt:
                gefunden.append(kategorie)
                break

    return tuple(gefunden)


def daten_laden(dateiname: str | Path) -> list[dict[str, Any]]:
    """Annotierte JSON-Beispiele laden und verständlich validieren."""

    pfad = Path(dateiname)

    try:
        with pfad.open(encoding="utf-8") as datei:
            daten = json.load(datei)
    except FileNotFoundError as fehler:
        raise ValueError(f"Datendatei nicht gefunden: {pfad}") from fehler
    except json.JSONDecodeError as fehler:
        raise ValueError(
            f"Ungültiges JSON in {pfad}, Zeile {fehler.lineno}."
        ) from fehler

    if not isinstance(daten, list) or not daten:
        raise ValueError(f"{pfad} muss eine nicht leere JSON-Liste enthalten.")

    erforderliche_felder = {
        "id",
        "betreff",
        "inhalt",
        "erwartete_kategorie",
    }
    bekannte_ids: set[Any] = set()

    for position, beispiel in enumerate(daten, start=1):
        if not isinstance(beispiel, dict):
            raise ValueError(f"Eintrag {position} in {pfad} ist kein Objekt.")

        fehlende_felder = erforderliche_felder - beispiel.keys()
        if fehlende_felder:
            felder = ", ".join(sorted(fehlende_felder))
            raise ValueError(f"Eintrag {position}: Felder fehlen: {felder}.")

        if beispiel["id"] in bekannte_ids:
            raise ValueError(f"Doppelte ID in {pfad}: {beispiel['id']}.")
        bekannte_ids.add(beispiel["id"])

        kategorie = beispiel["erwartete_kategorie"]
        if kategorie not in KATEGORIEN:
            raise ValueError(
                f"Eintrag {position}: unbekannte Kategorie {kategorie!r}."
            )

        email_text(beispiel["betreff"], beispiel["inhalt"])

    return daten


class LokalerEmailKlassifikator:
    """TF-IDF plus logistische Regression mit Sicherheitsrückweisung."""

    def __init__(
        self,
        pruefschwelle: float = 0.35,
        mindestabstand: float = 0.08,
    ) -> None:
        if not 0.0 <= pruefschwelle <= 1.0:
            raise ValueError("Die Prüfschwelle muss zwischen 0 und 1 liegen.")
        if not 0.0 <= mindestabstand <= 1.0:
            raise ValueError("Der Mindestabstand muss zwischen 0 und 1 liegen.")

        self.pruefschwelle = pruefschwelle
        self.mindestabstand = mindestabstand
        self.modell = Pipeline(
            [
                (
                    "merkmale",
                    FeatureUnion(
                        [
                            (
                                "wort_tfidf",
                                TfidfVectorizer(
                                    ngram_range=(1, 2),
                                    sublinear_tf=True,
                                ),
                            ),
                            (
                                "zeichen_tfidf",
                                TfidfVectorizer(
                                    analyzer="char_wb",
                                    ngram_range=(3, 5),
                                    sublinear_tf=True,
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "klassifikator",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )
        self._trainiert = False

    def trainieren(
        self,
        beispiele: list[dict[str, Any]],
    ) -> "LokalerEmailKlassifikator":
        vorhandene_kategorien = {
            beispiel["erwartete_kategorie"] for beispiel in beispiele
        }
        fehlende_kategorien = set(KATEGORIEN) - vorhandene_kategorien

        if fehlende_kategorien:
            fehlend = ", ".join(sorted(fehlende_kategorien))
            raise ValueError(f"Trainingsbeispiele fehlen für: {fehlend}.")

        texte = [
            email_text(beispiel["betreff"], beispiel["inhalt"])
            for beispiel in beispiele
        ]
        labels = [beispiel["erwartete_kategorie"] for beispiel in beispiele]

        self.modell.fit(texte, labels)
        self._trainiert = True
        return self

    def roh_vorhersagen(self, betreff: str, inhalt: str) -> tuple[str, float, float]:
        """Modellkategorie, höchsten Score und Abstand zum zweiten liefern."""

        if not self._trainiert:
            raise RuntimeError("Das Modell muss vor der Nutzung trainiert werden.")

        text = email_text(betreff, inhalt)
        wahrscheinlichkeiten = self.modell.predict_proba([text])[0]
        klassen = self.modell.named_steps["klassifikator"].classes_
        rangliste = sorted(
            zip(klassen, wahrscheinlichkeiten),
            key=lambda eintrag: eintrag[1],
            reverse=True,
        )

        beste_kategorie, bester_score = rangliste[0]
        zweiter_score = rangliste[1][1]
        abstand = bester_score - zweiter_score

        return str(beste_kategorie), float(bester_score), float(abstand)

    def klassifizieren(self, betreff: str, inhalt: str) -> Klassifizierung:
        text = email_text(betreff, inhalt)
        modell_kategorie, score, abstand = self.roh_vorhersagen(
            betreff,
            inhalt,
        )
        signale = signal_kategorien(text)
        mehrere_signale = len(signale) > 1

        unsicher = (
            score < self.pruefschwelle
            or abstand < self.mindestabstand
            or mehrere_signale
        )
        menschliche_pruefung = unsicher or modell_kategorie == "Unklar"
        kategorie = "Unklar" if unsicher else modell_kategorie

        if mehrere_signale:
            begruendung = (
                "Die Nachricht enthält gleichzeitig Signale für mehrere "
                f"Kategorien: {', '.join(signale)}."
            )
        elif score < self.pruefschwelle:
            begruendung = (
                "Der höchste Modellscore liegt unter der festgelegten "
                "Prüfschwelle."
            )
        elif abstand < self.mindestabstand:
            begruendung = (
                "Die beiden höchsten Kategoriescores liegen zu nah "
                "beieinander."
            )
        elif modell_kategorie == "Unklar":
            begruendung = (
                "Das Textmodell hat die Nachricht der Sammelkategorie "
                "Unklar zugeordnet."
            )
        else:
            begruendung = (
                f"Das lokale Textmodell erkennt am ehesten die Kategorie "
                f"{modell_kategorie}."
            )

        return Klassifizierung(
            kategorie=kategorie,
            modell_kategorie=modell_kategorie,
            modellscore=score,
            abstand_zum_zweiten=abstand,
            signal_kategorien=signale,
            menschliche_pruefung=menschliche_pruefung,
            begruendung=begruendung,
            antwortentwurf=ANTWORTVORLAGEN[kategorie],
        )


def standardmodell(projektordner: str | Path) -> LokalerEmailKlassifikator:
    """Das lokale Standardmodell mit den Projekt-Trainingsdaten erstellen."""

    projektordner = Path(projektordner)
    trainingsdaten = daten_laden(projektordner / "trainingsdaten.json")
    return LokalerEmailKlassifikator().trainieren(trainingsdaten)
