# Toaster SYP – Benutzerhandbuch

#syp #anleitung #toaster

---

## Voraussetzungen

Vor dem ersten Start sicherstellen:

- [ ] Python 3.8+ installiert
- [ ] Abhängigkeiten installiert: `pip install opencv-python torch torchvision numpy matplotlib Pillow`
- [ ] Webcam angeschlossen (USB bevorzugt, Kamera-ID 1)
- [ ] Shelly Plug S im selben WLAN (nur für automatisches Abschalten)
- [ ] Toaster mit Shelly Plug S verbunden

---

## Schritt-für-Schritt: Manueller Modus

> [!info] Wann diesen Modus nutzen?
> Wenn du volle Kontrolle über das Labeling willst oder bereits Bilder gesammelt hast.

### 1. Kameraaufnahme starten

```bash
cd SYP_Toaster/cam_app/manual_training
python recording.py
```

**Tastenbelegung im Aufnahmefenster:**

| Taste | Aktion |
|-------|--------|
| `R` | Videoaufnahme starten / stoppen |
| `S` | Screenshot in `data/raw_images/` speichern |
| `G` | Schwarz/Weiß-Modus umschalten |
| `A` | Bräunungsanalyse (Helligkeits-Overlay) ein/aus |
| `E` | Belichtung erhöhen |
| `D` | Belichtung verringern |
| `ESC` | Programm beenden |

> [!tip]
> Kamera fixieren und Belichtung so anpassen, dass die Heizstäbe nicht überstrahlen. Mindestens **20–30 Screenshots pro Klasse** sammeln.

---

### 2. Bilder zuschneiden

```bash
python 1_crop_images.py
```

Das Skript zeigt zunächst eine **Vorschau** mit dem grünen Crop-Rahmen. Falls der Rahmen nicht den Toast trifft:

1. Skript schließen (`Q`)
2. In `1_crop_images.py` diese Werte anpassen:

```python
CROP_X = 250      # Linker Rand
CROP_Y = 150      # Oberer Rand
CROP_WIDTH = 750  # Breite
CROP_HEIGHT = 550 # Höhe
```

3. Skript erneut starten. Mit `j` bestätigen → Bilder werden nach `data/cropped_images/` gespeichert.

---

### 3. Bilder labeln

```bash
python 2_label_images.py
```

Jedes zugeschnittene Bild wird einzeln angezeigt. Klasse mit Taste vergeben:

| Taste | Klasse | Beschreibung |
|-------|--------|--------------|
| `1` | roh | Ungetoastet |
| `2` | leicht | Leicht gebräunt |
| `3` | perfekt | Goldbraun |
| `4` | dunkel | Zu dunkel |
| `5` | verbrannt | Verbrannt |
| `S` | – | Bild überspringen |
| `ESC` | – | Beenden |

Bilder werden automatisch in `data/labeled_dataset/<klasse>/` sortiert.

---

### 4. Modell trainieren

```bash
python 3_train_model.py
```

- Trainiert 30 Epochen (Adam, ReduceLROnPlateau)
- Speichert das **beste Modell** in `models/toast_model.pt`
- Erstellt einen **Training-Plot** in `models/training_history.png`
- 80/20 Train-Val-Split, mit Data Augmentation

> [!warning]
> Mindestens 10 Bilder werden benötigt. Empfohlen: 20–30 pro Klasse für gute Genauigkeit.

---

### 5. Live-Erkennung mit Shelly-Steuerung

```bash
python 4_live_prediction.py
```

Oder mit Parametern:

```bash
python 4_live_prediction.py --level 3 --min-conf 70 --stable-frames 12 --shelly-host 192.168.178.50
```

**CLI-Parameter:**

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `--level` | 3 | Ziel-Bräunungsstufe (1–5) |
| `--min-conf` | 70 | Minimale Konfidenz in % |
| `--stable-frames` | 12 | Stabile Frames bis Aktion |
| `--shelly-host` | aus Config | IP des Shelly Plug S |

**Tastenbelegung im Live-Modus:**

| Taste | Aktion |
|-------|--------|
| `1`–`5` | Ziel-Bräunungsstufe setzen |
| `A` | Auto-Off ein/aus |
| `Leertaste` | Session starten (Shelly EIN + Reset) |
| `O` | Shelly manuell einschalten |
| `F` | Shelly manuell ausschalten |
| `ESC` | Beenden |

---

## Schritt-für-Schritt: Vollautomatischer Modus (PID)

> [!info] Wann diesen Modus nutzen?
> Wenn du kein manuelles Labeling machen willst. Das System lernt selbstständig aus dem zeitlichen Verlauf des Toastvorgangs.

### 1. Trainingsdaten aufzeichnen (Lehrer-Modus)

```bash
cd SYP_Toaster/cam_app
python manual_trainer.py
```

**Ablauf:**

1. `Leertaste` → Aufnahme starten
2. Während der Toast bräunt: Taste `0`–`4` drücken, um den aktuellen Zustand zu setzen
3. `Leertaste` → Pause (z.B. zwischen zwei Toasts)
4. `S` → Speichern und Beenden

| Taste | Klasse |
|-------|--------|
| `0` | ROH |
| `1` | LEICHT |
| `2` | PERFEKT |
| `3` | DUNKEL |
| `4` | VERBRANNT |
| `Leertaste` | Start / Pause |
| `S` | Speichern + Beenden |
| `Q` | Abbrechen ohne Speichern |

Daten werden in `auto_training_data/training_data.pkl` gespeichert. Mehrere Sessions werden automatisch zusammengeführt.

---

### 2. PID-Steuerung starten

```bash
python pid_toaster_control.py
```

Das Programm:
1. Lädt alle vorhandenen `.pkl`-Dateien aus `auto_training_data/`
2. Trainiert das ResNet18-Modell (falls genug Daten vorhanden: ≥ 50 Samples)
3. Öffnet die Kamera und startet die Echtzeit-Klassifikation
4. Der PID-Regler berechnet kontinuierlich die Abweichung vom Zielzustand (Standard: `perfekt`)
5. Bei Ziel-Erreichung → Shelly Plug S ausschalten

**Wichtige Konfigurationsparameter** in `pid_toaster_control.py` → Klasse `Config`:

```python
TARGET_STATE = 2          # Zielklasse (0=roh … 4=verbrannt)
PID_KP = 1.5              # P-Anteil
PID_KI = 0.3              # I-Anteil
PID_KD = 0.5              # D-Anteil
MAX_TOAST_TIME = 180.0    # Sicherheits-Timeout in Sekunden
MIN_SAMPLES_FOR_TRAINING = 50
RETRAIN_INTERVAL = 100    # Alle N Samples neu trainieren
```

**Ergebnisse** werden automatisch gespeichert:
- `pid_results/pid_response_<timestamp>.png` – PID-Antwort-Plot
- `pid_results/session_<timestamp>.json` – Session-Daten

---

## Shelly Plug S einrichten

### 1. Konfigurationsdatei anlegen

```bash
cp shelly_config.example.json shelly_config.json
```

### 2. IP-Adresse eintragen

```json
{
  "host": "192.168.178.XX",
  "relay": 0,
  "username": "",
  "password": "",
  "timeout_s": 2.0
}
```

> [!tip]
> Die IP des Shelly findet man in der Shelly-App oder im Router-DHCP-Dashboard. Eine feste IP im Router vergeben, damit sie sich nicht ändert.

### 3. Verbindung testen

Direkt im Browser aufrufen:
```
http://192.168.178.XX/relay/0?turn=on
http://192.168.178.XX/relay/0?turn=off
```

Wenn der Toaster sich ein- und ausschaltet, ist alles korrekt konfiguriert.

---

## Häufige Probleme

| Problem | Mögliche Ursache | Lösung |
|---------|-----------------|--------|
| `Keine Kamera gefunden` | Falsche Kamera-ID | Andere ID ausprobieren (`--camera-id 0` oder `1`) |
| Modell nicht gefunden | Training noch nicht durchgeführt | Erst `3_train_model.py` ausführen |
| Shelly antwortet nicht | Falsehe IP / nicht im WLAN | IP prüfen, Shelly neu starten |
| Schlechte Genauigkeit | Zu wenige Trainingsdaten | Mehr Bilder pro Klasse labeln (mind. 30) |
| Kamera überstrahlt | Belichtung zu hoch | `E`/`D` in `recording.py` nutzen |
| PKL-Datei leer | Keine Aufnahme gestartet | `Leertaste` in `manual_trainer.py` drücken |

---

## Tipps für gutes Training

> [!tip] Best Practices
> - **Kamera fixieren** – immer aus derselben Position aufnehmen
> - **Belichtung anpassen** – Heizstäbe dürfen nicht überstrahlen
> - **Verschiedene Toast-Sorten** – macht das Modell robuster
> - **Mindestens 20–30 Bilder pro Klasse** für den manuellen Modus
> - **Mindestens 3–5 Toast-Vorgänge** für den automatischen Modus

---

## Verwandte Notizen

- [[Toaster_SYP_Projektdokumentation]]
