# Formica

**Ein KI-Arbeitsplatz zum Selbstbetreiben: Chat, Benutzer und Wissensdatenbank, eingerichtet über eine Weboberfläche.**

Formica (lateinisch „Ameise“) ist das Gegenstück zu [MyrmidLM](https://github.com/Miki84/MyrmidLM): MyrmidLM bündelt Rechner zu einem Modell-Backend, Formica macht daraus einen Arbeitsplatz für die Benutzer. Formica funktioniert aber mit jedem OpenAI-kompatiblen Backend (vLLM, Ollama, LiteLLM, OpenRouter).

## Was drin ist

| Dienst | Aufgabe |
|---|---|
| **Open WebUI** | Chat, Benutzer, Gruppen, Wissensdatenbanken |
| **PostgreSQL** | Daten von Open WebUI |
| **Qdrant** | Vektordatenbank |
| **Docling** | Liest PDFs, Office-Dateien, Tabellen, Scans |
| **Embeddings** | bge-m3 über llama.cpp (dieselbe Engine wie MyrmidLM) |
| **Caddy** | Reverse Proxy und HTTPS |
| SearXNG (optional) | Websuche in Open WebUI |
| Whisper (optional) | Spracheingabe im Chat |
| n8n (optional) | Workflows und Automatisierung |
| Dateien erzeugen (optional) | Excel, Word, PDF und CSV aus dem Chat |
| Bilder erzeugen (optional) | Stable Diffusion über stable-diffusion.cpp oder einen externen Dienst |
| Eigene Lösungen | beliebige Container, über Port oder Subdomain erreichbar |
| **Formica** | Verwaltungsoberfläche für alles oben |

## Installation

Auf einer frischen Linux-VM (Ubuntu 24.04 empfohlen, mindestens 4 Kerne, 16 GB RAM, 100 GB Platte):

```bash
curl -fsSL https://raw.githubusercontent.com/Miki84/Formica/main/install.sh | bash
```

Das Skript installiert Docker, baut die Verwaltungsoberfläche und zeigt am Ende Adresse und Einrichtungscode:

```
Verwaltungsoberfläche:  http://192.168.2.50:8095
Einrichtungscode:       x8Fk2…
```

Im Browser öffnen, Code und Admin-Passwort eingeben, dann:

1. **Einstellungen:** Hostname, Betriebsart (HTTP, HTTPS mit eigener CA, Let's Encrypt, hinter Reverse Proxy) und Modell-Backend eintragen, „Verbindung testen“, speichern.
2. **Übernehmen:** Formica lädt die Images und startet alle Dienste. Beim ersten Mal dauert das einige Minuten.
3. **Open WebUI öffnen:** Das erste Konto, das dort angelegt wird, ist Administrator.

## Bedienung auf dem Server

```bash
sudo formica token            # Einrichtungscode anzeigen
sudo formica status           # Dienste
sudo formica logs             # Log der Verwaltungsoberfläche
sudo formica update           # neue Formica-Version holen
sudo formica reset-password   # Admin-Passwort vergessen
```

## Wie es funktioniert

Formica speichert alle Einstellungen in `/opt/formica/config.json` (nur für root lesbar). Beim Übernehmen erzeugt es daraus `stack/compose.yaml` und `stack/caddy/Caddyfile` und startet `docker compose`. Beide Dateien sind im Reiter „Dateien“ einsehbar: Was dort steht, läuft.

**Wichtige Regel:** Was Formica verwaltet (Backend, Dokumentenverarbeitung, Embeddings, Registrierung), wird bei jedem Start von Formica gesetzt. Änderungen daran im Admin-Bereich von Open WebUI gehen beim nächsten Neustart verloren. In Open WebUI selbst pflegen Sie Modelle, Wissensdatenbanken, Benutzer und Gruppen.

## Fahrplan

| Version | Inhalt |
|---|---|
| 0.1 | Installation, Verwaltungsoberfläche, Kernmodule, Anbindung an MyrmidLM |
| 0.2 | Anmeldung mit Entra ID (mit und ohne P1), Zertifikat hochladen |
| 0.3 | Backup und Wiederherstellung, Updates mit Rollback |
| 0.4 | GPU-Monitoring, weitere Module (SearXNG, n8n, Whisper) |
| 1.0 | Releases über CI, fertiges Image auf ghcr.io |

## Lizenz

MIT. Open WebUI steht unter eigener Lizenz: Ab 50 Benutzern darf das Open-WebUI-Branding nur mit einer Enterprise-Lizenz entfernt werden. Formica lässt es unverändert.
