# Toaster SYP – Projektdokumentation

#syp #computer-vision #deep-learning #pid #toaster

---

## Projektübersicht

**Toaster SYP** ist ein automatisches Toaster-Steuerungssystem, das Computer Vision, Deep Learning und einen PID-Regler kombiniert, um Toast auf eine gewünschte Bräunungsstufe zu bringen – und ihn dann automatisch via Shelly Smart Plug abzuschalten.

Die Kamera beobachtet den Toast in Echtzeit. Ein trainiertes neuronales Netz klassifiziert den aktuellen Bräunungsgrad, und ein PID-Regler entscheidet, wann der Toaster ausgeschaltet werden soll.

---

## Systemarchitektur

```
Kamera
  │
  ▼
Bildvorverarbeitung (Grayscale, 224×224)
  │
  ▼
CNN-Klassifikation (ResNet18 / ToastCNN)
  │ → Klasse: roh / leicht / perfekt / dunkel / verbrannt
  │ → Konfidenzwert
  ▼
PID-Regler
  │ → Vergleicht Ist-Zustand mit Soll-Zustand
  │ → Berechnet Zeitkorrektur
  ▼
Shelly Plug S (HTTP API)
  │ → Schaltet Strom EIN / AUS
  ▼
Toast fertig!
```

---

## Zwei Betriebsmodi

### Modus 1 – Manuelles Training (Schritt-für-Schritt)

Klassischer ML-Workflow: Daten sammeln → labeln → trainieren → live nutzen.

| Schritt | Skript | Beschreibung |
|---------|--------|--------------|
| 1 | `recording.py` | Kameraaufnahme & Screenshots erstellen |
| 2 | `1_crop_images.py` | Rohbilder auf Toast-Bereich zuschneiden |
| 3 | `2_label_images.py` | Bilder manuell in Klassen einordnen |
| 4 | `3_train_model.py` | CNN (ToastCNN) trainieren |
| 5 | `4_live_prediction.py` | Live-Erkennung mit Shelly-Integration |

### Modus 2 – Vollautomatisch (PID-Steuerung)

Kein manuelles Labeling nötig. Das System lernt selbstständig aus Zeitstempeln.

| Schritt | Skript | Beschreibung |
|---------|--------|--------------|
| 1 | `manual_trainer.py` | Daten mit Kamera und Tasten aufzeichnen |
| 2 | `pid_toaster_control.py` | PID-Regler + Online-Training + Shelly-Steuerung |

---

## Klassifikation

Das System unterscheidet 5 Bräunungsstufen:

| Index | Klasse | Beschreibung | Farbe (Overlay) |
|-------|--------|--------------|-----------------|
| 0 | **roh** | Ungetoastet, sehr hell | Grau |
| 1 | **leicht** | Leicht gebräunt | Gelb |
| 2 | **perfekt** | Goldbraun, ideal | Grün |
| 3 | **dunkel** | Zu dunkel | Orange |
| 4 | **verbrannt** | Verbrannt, schwarz | Rot |

---

## Neuronale Netze

### ToastCNN (manueller Modus)
- Eigenentwickeltes CNN mit 5 Convolutional-Blöcken
- Input: 224×224 RGB-Bild
- Output: 5-Klassen-Softmax
- Training: 30 Epochen, Adam-Optimierer, ReduceLROnPlateau
- Data Augmentation: Horizontal Flip, Rotation ±10°, Brightness/Contrast Jitter
- Gespeichert in: `models/toast_model.pt`

### ResNet18 (PID-Modus)
- Vortrainiertes ResNet18 (Transfer Learning)
- Letzte Schicht ersetzt durch Linear(512 → 5)
- Input: 224×224 Grayscale (als 3-Kanal dupliziert)
- Gespeichert in: `models/pid_toast_model.pt`
- Online-Training: Alle 100 neuen Samples wird neu trainiert

---

## PID-Regler

Der PID-Regler steuert die Toastzeit, indem er den aktuellen Klassifikations-Index (0–4) mit dem Zielwert vergleicht.

```
Ausgabe = Kp × e(t) + Ki × ∫e(t)dt + Kd × de(t)/dt
```

**Standardparameter:**

| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| Kp | 1.5 | Proportional Gain |
| Ki | 0.3 | Integral Gain |
| Kd | 0.5 | Derivative Gain |
| Abtastzeit | 1.0 s | PID-Update-Intervall |
| Max. Toastzeit | 180 s | Sicherheits-Timeout |
| Min. Toastzeit | 30 s | Mindestlaufzeit |

**Ergebnisse** (Plots, JSON-Logs) werden automatisch in `pid_results/` gespeichert.

---

## Shelly Plug S Integration

Der Toaster wird über einen **Shelly Plug S** (WLAN-Steckdose) gesteuert.

- **Protokoll:** HTTP GET (kein MQTT, kein Cloud-Zugriff nötig)
- **Gen1-API:** `GET http://<host>/relay/0?turn=on|off`
- **Gen2-API:** `GET http://<host>/rpc/Switch.Set?id=0&on=true|false`
- Der Client probiert automatisch erst Gen1, dann Gen2.
- Optionaler HTTP Basic Auth Support.

**Konfiguration:** `shelly_config.json`

```json
{
  "host": "192.168.178.50",
  "relay": 0,
  "username": "",
  "password": "",
  "timeout_s": 2.0
}
```

---

## Ordnerstruktur

```
toaster_syp/
└── SYP_Toaster/
    └── cam_app/
        ├── manual_trainer.py          # Datenaufnahme (Lehrer-Modus)
        ├── pid_toaster_control.py     # PID-Steuerung (vollautomatisch)
        ├── shelly_config.json         # Shelly-Verbindungsdaten
        │
        ├── manual_training/           # Manueller Workflow
        │   ├── recording.py           # Kamera + Screenshots
        │   ├── 1_crop_images.py       # Bilder zuschneiden
        │   ├── 2_label_images.py      # Bilder labeln (GUI)
        │   ├── 3_train_model.py       # CNN trainieren
        │   └── 4_live_prediction.py   # Live-Erkennung + Shelly
        │
        ├── data/                      # Rohdaten
        │   ├── videos/
        │   ├── raw_images/
        │   ├── cropped_images/
        │   └── labeled_dataset/
        │       ├── roh/
        │       ├── leicht/
        │       ├── perfekt/
        │       ├── dunkel/
        │       └── verbrannt/
        │
        ├── auto_training_data/        # PKL-Dateien (automatischer Modus)
        ├── models/                    # Gespeicherte Modelle (.pt)
        ├── pid_results/               # PID-Logs und Plots
        └── utils/                     # YOLOv3 (nicht aktiv im Hauptsystem)
```

---

## Abhängigkeiten

```bash
pip install opencv-python torch torchvision numpy matplotlib Pillow
```

| Paket | Verwendung |
|-------|-----------|
| `opencv-python` | Kamera, Bildverarbeitung, GUI |
| `torch` / `torchvision` | CNN-Training und Inferenz |
| `numpy` | Array-Operationen |
| `matplotlib` | Training-Plots, PID-Response-Plots |
| `Pillow` | Bildkonvertierung |

> [!note]
> CUDA wird automatisch erkannt. Ohne GPU läuft alles auf CPU – beim Online-Training im PID-Modus ggf. langsamer.

---

## Verwandte Notizen

- [[Toaster_SYP_Benutzerhandbuch]]
