# Änderungen

## 0.4.2
- Bilder über MyrmidLM (ab 2.16.0) mit beiden Engines: stable-diffusion.cpp oder ComfyUI auf dem MyrmidLM-Host, Modelle wie Z-Image Turbo, FLUX.1 schnell und Qwen-Image
- Neue Wahl „ComfyUI mit Workflow“: Open WebUI nutzt seine ComfyUI-Anbindung über den Master (`/comfy`), mit dem Standard-Workflow des Modells von MyrmidLM oder einem eigenen Workflow samt Knoten-Zuordnung
- Beim Speichern fragt Formica MyrmidLM nach Modell, Engine und Standardgröße und zeigt sie an
- Neue Installationen nutzen für Bilder standardmäßig das Modell-Backend

## 0.4.1
- Bilder erzeugen „über das Modell-Backend (MyrmidLM)“: das Bildmodell läuft auf einem MyrmidLM-Host (CPU oder eigene GPU, ab MyrmidLM 2.15.0), Open WebUI nutzt `/v1/images/generations` des Masters mit dessen Key. Formica startet dann keinen eigenen Bild-Container.

## 0.4.0
- Neue Lösung „Dateien erzeugen“: Open WebUI erstellt Excel (.xlsx), Word (.docx), PDF und CSV direkt aus dem Chat und gibt einen Download-Link zurück. Zahlen werden als Zahlen übernommen, auch im deutschen Format (11,25); Dateien werden nach einstellbarer Zeit gelöscht. Wird automatisch als Werkzeug in Open WebUI eingetragen.
- Neue Lösung „Bilder erzeugen“: Stable Diffusion über stable-diffusion.cpp (sd-server) auf der CPU oder einer NVIDIA-GPU, Modell wählbar (SD 1.5, SD Turbo, SDXL Turbo, wird einmalig geladen), oder ein externer OpenAI-kompatibler Bilddienst
- Docker-Images aus dem Formica-Repo werden beim Übernehmen gebaut

## 0.3.1
- „Von Formica gesetzt“ zeigt jetzt Name und Wert jeder Variable; Passwörter und Keys verdeckt mit „anzeigen“, lange Werte gekürzt mit „ganz anzeigen“
- Knopf „Anpassen“ übernimmt eine Variable mit ihrem Wert als eigenen Eintrag; überschriebene Werte sind durchgestrichen

## 0.3.0
- Neue Startseite „Cockpit“ wie im meloki AdminPACK: Lösungskarten mit Status, Updates, Monitoring (CPU, RAM, GPU, Speicher, GPU-Verlauf der letzten 5 Minuten, Ereignisse), Sicherheit, Backup, Modell-Backend, System
- Neuer Reiter „Lösungen“: Lösungen und Verbünde aktivieren und deaktivieren, starten, stoppen, neu starten
- Neue Lösungen: Websuche (SearXNG, direkt in Open WebUI eingebunden), Whisper (Spracheingabe, Modell wählbar), n8n
- Eigene Lösungen: beliebiger Container mit Image, Port, Umgebungsvariablen, Volumes und optional GPU
- Erreichbar über eigenen Port oder Subdomain, mit HTTPS je nach Betriebsart
- Ereignisprotokoll (Anmeldungen, Änderungen, Vorgänge), im Reiter Logs durchsuchbar; Suche auch in den Logs der Dienste

## 0.2.3
- Einstellungen → „Erweitert: eigene Umgebungsvariablen“ je Dienst (Open WebUI, Docling, Qdrant, Embeddings, PostgreSQL, Caddy), wie bei meloki
- Markiert, wenn ein Eintrag einen Wert von Formica überschreibt; Warnung bei Variablen, die die Dienste miteinander verbinden
- Liste der Variablen, die Formica selbst setzt

## 0.2.2
- Adressen dürfen jetzt auch mit dem vollen Pfad eingetragen werden (…/v1/embeddings, …/v1/chat/completions): Formica nimmt die Basisadresse

## 0.2.1
- Embeddings wählbar: auf diesem Server, über das Modell-Backend (MyrmidLM ab 2.14.0, GPU oder CPU im Cluster) oder eigene Adresse
- Knopf „Embeddings testen“ zeigt Dimensionen und Antwortzeit
- Läuft das Embedding-Modell nicht mehr hier, startet Formica den Container dafür nicht mehr

## 0.2.0
- Neuer Reiter „Monitor“ wie bei MyrmidLM: CPU mit Kernen und Temperatur, RAM, Datenträger- und Netzwerkdurchsatz mit Verlauf
- Auslastung jedes Dienstes (CPU, RAM, Netz, Platte, Prozesse)
- Modell-Cluster aus dem MyrmidLM-Dashboard: Tempo, GPU-Auslastung, VRAM, Temperatur, Leistung, PCIe, Kontext-Füllung, Cache und letzte Anfragen
- Die Verwaltung läuft jetzt im Host-Netz (echte Netzwerkwerte); ältere Installationen werden einmalig automatisch umgestellt

## 0.1.3
- Neue Karte „Chat und Tempo“: eigenes Modell für Nebenaufgaben (z. B. fast), Suchanfragen, Titel, Tags, Folgefragen und Autovervollständigung einzeln schaltbar
- Standard schneller: keine eigenen Suchanfragen, keine Tags, keine Autovervollständigung

## 0.1.2
- Updates direkt in der Oberfläche: prüfen und mit einem Klick installieren, mit Rückfall auf die vorige Version
- CPU-Auslastung in Prozent mit Balken statt Load-Werten
- Design wie MyrmidLM, mit Umschalter automatisch, dunkel und hell

## 0.1.1
- Texterkennung (OCR) für Docling einstellbar: automatisch, erzwingen, aus
- Tabellen werden immer genau erkannt
- Update-Hinweis von Open WebUI ausgeblendet, Updates kommen über Formica

## 0.1.0
- Erste Version: Installation, Verwaltungsoberfläche, Open WebUI, PostgreSQL, Qdrant, Docling, Embeddings, Caddy
