# Lokale E-Mail-Klassifizierung mit KI (Python + n8n + Ollama)

Eingehende Mails an ein Gmail-Postfach werden automatisch klassifiziert,
der zuständigen Abteilung zugeordnet und beantwortet – vollständig lokal in Docker.

## Ablauf

```
Gmail-Posteingang ──IMAP──► n8n ──► Classifier-API (Python) ──► Ollama (llama3.2:3b)
                                 │
                                 ├─► Gmail-ENTWURF mit inhaltsbezogener Antwort (Mensch prüft & sendet)
                                 ├─► Eingangsbestätigung an den Absender (automatisch)
                                 └─► Info-Mail an INFO_EMAIL (Kategorie, Abteilung, Score, Entwurf)
```

| Kategorie  | Abteilung            |
|------------|----------------------|
| Anfrage    | Kundenservice        |
| Beschwerde | Beschwerdemanagement |
| Rechnung   | Buchhaltung          |
| Unklar     | Zentrale Poststelle  |

**Hybrid-Klassifikation:** Zuerst ein schnelles, erklärbares ML-Modell
(TF-IDF + logistische Regression). Ist es unsicher (niedriger Score, knapper Abstand,
Signale für mehrere Kategorien), entscheidet das lokale LLM als zweite Meinung.

**Human-in-the-Loop:** Die eigentliche, vom LLM formulierte Antwort wird **nie
automatisch versendet**, sondern als Entwurf abgelegt. Das schützt vor
Halluzinationen und Prompt-Injection.

## Voraussetzungen

- Docker Desktop (bzw. Docker Engine + Compose v2)
- ca. 4 GB freier RAM und ~3 GB Speicherplatz für das Modell `llama3.2:3b`
- Ein **eigenes** Gmail-Konto nur für dieses Projekt (empfohlen, nicht das private Postfach)

## Einrichtung

1. Im Gmail-Konto: 2-Schritt-Verifizierung aktivieren, **App-Passwort** erstellen,
   in den Gmail-Einstellungen **IMAP aktivieren**.
2. `cp .env.example .env` und ausfüllen:
   - `GMAIL_USER` – das Gmail-Konto aus Schritt 1
   - `GMAIL_APP_PASSWORD` – das App-Passwort
   - `INFO_EMAIL` – deine eigene Adresse für die Info-Mails
3. `docker compose up -d --build` (beim ersten Start lädt `ollama-init` das Modell, ~2 GB).
4. n8n öffnen: http://localhost:5678 → `workflow-gmail-v3.json` importieren.
   Die Adressen liest der Workflow automatisch aus der `.env` (`$env.GMAIL_USER`, `$env.INFO_EMAIL`).
5. Credentials in n8n zuweisen:
   - **IMAP:** imap.gmail.com, Port 993, SSL, User = `GMAIL_USER`, App-Passwort
   - **SMTP:** smtp.gmail.com, Port 465, SSL, gleiche Zugangsdaten
6. Workflow **aktivieren** (Schalter oben rechts).

Damit es dauerhaft läuft: Docker Desktop → Settings → „Start Docker Desktop when you sign in“.
Alle Container haben `restart: unless-stopped`. (Wenn der Rechner schläft, läuft nichts.)

> **Hinweis:** Jede eingehende Mail löst automatisch eine Eingangsbestätigung aus.
> Die Adresse des Projekt-Postfachs daher nicht öffentlich posten, sonst kann es
> für Spam bzw. Rückstreuung (Backscatter) missbraucht werden.

## Testen

- Von einer **anderen** Adresse (nicht `GMAIL_USER` – die wird als
  Schleifenschutz ignoriert) eine Mail an das Projekt-Postfach senden.
- API direkt: `curl -X POST localhost:8000/classify -H 'Content-Type: application/json' -d '{"betreff":"Rechnung falsch","inhalt":"Der Betrag auf Rechnung 4711 ist zu hoch."}'`
- Status: `curl localhost:8000/health` · Logs: `docker compose logs -f classifier n8n`
- Ohne Gmail: `docker compose --profile test up -d` und `workflow.json` (Webhook + MailHog,
  Mails ansehen unter http://localhost:8025)

## Dateien

| Datei | Zweck |
|---|---|
| `classifier/classifier.py` | ML-Modell, Sicherheitsregeln, Vorlagen |
| `classifier/api.py` | REST-API: `/classify`, `/entwurf`, `/health` |
| `classifier/evaluate.py` | Auswertung des Modells mit `testdaten.json` |
| `classifier/trainingsdaten.json`, `testdaten.json` | Synthetische Beispiel-Mails (keine echten Daten) |
| `workflow-gmail-v3.json` | **aktueller** n8n-Workflow |
| `workflow.json` | Test-Workflow per Webhook + MailHog |
| `.env.example` | Vorlage für die Zugangsdaten (`.env` wird nie eingecheckt) |

## Lizenz

[MIT](LICENSE)
