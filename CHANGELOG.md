# Änderungen

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
