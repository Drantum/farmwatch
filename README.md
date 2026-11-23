# LKW-Erkennung und Hofkameras

Automatisches Kennzeichen- und Videoüberwachungssystem für Hof- und Ladekameras (Tank-LKWs, Abkalbebox) gemäß SOP.

---

## 1. Übersicht & Architektur

Das System empfängt RTSP-Streams zweier Kameras, verteilt diese latenzarm (< 0,5s über WebRTC/MSE) über **go2rtc** und wertet den LKW-Stream mit **YOLO** und **fast-alpr** kontinuierlich auf der CPU / Intel iGPU aus.

```
Kamera Ladefläche ─┐                      ┌─► Browser (Live-Ansicht, WebRTC/MSE)
                   ├─► go2rtc ────────────┤
Kamera Abkalbebox ─┘   (Restream)         └─► RTSP intern (:8554/lkw)
                                               │
                                               ▼
                                   Auswertungs-App (Python)
                                   ├─ ffmpeg-Recorder (10s Segmente -> ./data/seg/)
                                   ├─ YOLO: LKW erkennen (~5 FPS, CPU-optimiert)
                                   ├─ fast-alpr: Kennzeichen lesen
                                   ├─ Optionale Vision-API: Speditionsaufschrift
                                   └─ Ereignis speichern
                                               │
                          SQLite + Dateien (/data) ◄─ FastAPI (:8000) ─► Weboberfläche
```

### Hauptfunktionen
- **Live-Streaming:** Ladefläche und Abkalbebox nebeneinander im Browser via go2rtc WebRTC (auf Mobilgeräten responsive untereinander).
- **Verlustfreier ffmpeg-Recorder:** Kontinuierliche 10-Sekunden-Segmente (`-c copy`), minimaler CPU-Verbrauch. Segmente älter als 10 Minuten werden automatisch bereinigt.
- **Präziser Clip-Zuschnitt:** Beim Verlassen des LKWs (8s Debounce-Timeout) wartet das System 12s und schneidet aus den Segmenten einen MP4-Clip mit **3s Vorlauf und 2s Nachlauf** zusammen.
- **Kennzeichen-Logik (deutsches Format):**
  - Bereinigung auf Großbuchstaben und Ziffern (inkl. Umlaute Ä, Ö, Ü).
  - Strikte Formatprüfung gegen das deutsche Kennzeichenmuster (`^[A-ZÄÖÜ]{1,3}[A-Z]{1,2}[1-9][0-9]{0,3}[EH]?$`) und offizielle Zulassungsbezirke.
  - Gewichtete Abstimmung (Voting) über alle Frames eines Ereignisses; Snapshot ist das Bild mit der höchsten Lesesicherheit.
- **Optionale Speditionserkennung:** Über jede OpenAI-kompatible Vision-API (z.B. GPT-4o-mini, vLLM, OpenRouter) konfigurierbar.
- **Weboberfläche:**
  - Modernes Dashboard mit automatischem Dark- und Light-Mode.
  - Auto-Refresh der LKW-Ereignisse alle 15 Sekunden.
  - Interaktives Video-Modal zum direkten Abspielen und Herunterladen der Clips.

---

## 2. Schnellstart mit Docker Compose (Empfohlen)

### Voraussetzungen
- Docker und Docker Compose auf dem Server installiert.
- Kameras im selben Netzwerk erreichbar.

### 1. Konfiguration vorbereiten
Kopieren Sie die Vorlage `.env.example` nach `.env`:
```bash
cp .env.example .env
```
Passen Sie die RTSP-Adressen Ihrer Kameras in der `.env` an:
```env
LKW_CAM_RTSP=rtsp://admin:passwort@192.168.1.100:554/h264Preview_01_main
KALB_CAM_RTSP=rtsp://admin:passwort@192.168.1.101:554/h264Preview_01_main

# Optional: Vision API für Speditionserkennung
VISION_ENABLED=false
VISION_API_BASE_URL=https://api.openai.com/v1
VISION_API_KEY=sk-...
VISION_MODEL=gpt-4o-mini
```

### 2. Container starten
```bash
docker compose up -d --build
```

### 3. Weboberfläche aufrufen
- **Weboberfläche:** [http://SERVER:8000](http://localhost:8000)
- **go2rtc Web-UI:** [http://SERVER:1984](http://localhost:1984)

---

## 3. Lokale Entwicklung / Betrieb ohne Docker

1. **Python 3.10+ & ffmpeg installieren:**
   - ffmpeg muss im Systempfad (`PATH`) vorhanden sein.
2. **Virtuelle Umgebung erstellen:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```
3. **Abhängigkeiten installieren:**
   ```bash
   pip install -r requirements.txt
   ```
4. **go2rtc starten:**
   go2rtc als Binary herunterladen und mit `./go2rtc -c go2rtc.yaml` starten.
5. **App starten:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 4. Konfigurationsoptionen (`config.yaml`)

| Parameter | Standard | Beschreibung |
|---|---|---|
| `recorder.segment_time` | `10` | Länge der Rohsegmente in Sekunden |
| `recorder.retention_minutes` | `10` | Aufbewahrungsdauer der Rohsegmente vor dem Löschen |
| `recorder.pre_buffer_seconds` | `3` | Vorlaufzeit für den finalen Clip |
| `recorder.post_buffer_seconds` | `2` | Nachlaufzeit für den finalen Clip |
| `detector.target_fps` | `5.0` | Auswertungsfrequenz der Frame-Analyse |
| `detector.yolo_model` | `yolov8n.pt` | Schlankes YOLO-Modell für CPU-Inferenz |
| `detector.device` | `cpu` | Inferenz-Gerät (`cpu` für CPU & Intel iGPU) |
| `event.idle_timeout_seconds` | `8.0` | Zeit ohne LKW bis zum Abschluss des Ereignisses |
| `event.segment_flush_wait_seconds` | `12.0` | Wartezeit vor dem Schneiden, um alle Segmente zu sichern |
| `vision.enabled` | `false` | Aktiviert OpenAI-kompatible Speditionserkennung |

---

## 5. Schnittstellen (API)

| Methode | Pfad | Beschreibung |
|---|---|---|
| `GET` | `/` | Web-Dashboard (Live-Player & LKW-Karten) |
| `GET` | `/api/events` | Liste der letzten 100 Ereignisse als JSON (Filterbar mit `?limit=` & `?offset=`) |
| `GET` | `/api/events/{id}` | Detaildaten zu einem einzelnen LKW-Ereignis |
| `DELETE` | `/api/events/{id}` | Löscht ein Ereignis und die zugehörigen Bild- und Videodateien |
| `GET` | `/api/status` | Diagnose-Status (Recorder, Detektor, FPS, Festplattenspeicher) |
| `GET` | `/media/{datei}` | Auslieferung von Snapshots (`.jpg`) und Videoclips (`.mp4`) |

---

## 6. Automatisierte Tests

Das Projekt verfügt über eine umfassende Test-Suite:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
Getestet werden:
- **ALPR-Logik & Kennzeichen-Filter:** Reguläre Kennzeichen, E- und H-Kennzeichen, Umlaute, Ablehnung ungültiger Formate.
- **Gewichtete Abstimmung (Voting):** Höchste Confidence und korrekte Snapshot-Auswahl bei Bildserien.
- **Clip-Zuschnitt-Formel:** Segment-Auswahl für Zeitintervalle mit 3s Vorlauf und 2s Nachlauf.
- **Event-State-Machine:** 8s Timeout-Debounce, Karenzzeit und Event-Persistierung.
- **SQLite-Datenbank:** CRUD-Operationen, Sortierung und Verbindungsverwaltung.
- **Vision-Client:** Bereinigung und Parsing von API-Antworten.
