# 🍞 Toast-Bräunungs-Erkennung

Automatische Erkennung der Toast-Bräunung mittels Computer Vision und Deep Learning.

## 📁 Ordnerstruktur

```
cam_app/
├── recording.py           # Kamera-Aufnahme & Screenshots
├── 1_crop_images.py       # Bilder zuschneiden
├── 2_label_images.py      # Bilder labeln (GUI)
├── 3_train_model.py       # CNN trainieren
├── 4_live_prediction.py   # Live-Erkennung
│
├── data/                  # Alle Daten
│   ├── videos/            # Aufgenommene Videos
│   ├── raw_images/        # Rohbilder von recording.py
│   ├── cropped_images/    # Zugeschnittene Bilder
│   └── labeled_dataset/   # Gelabelte Trainingsdaten
│       ├── roh/
│       ├── leicht/
│       ├── perfekt/
│       ├── dunkel/
│       └── verbrannt/
│
├── models/                # Trainierte Modelle
│   ├── toast_model.pt
│   └── training_history.png
│
└── utils/                 # Hilfsdateien
```

## 🚀 Workflow

### 1️⃣ Daten sammeln
```bash
python recording.py
```
**Tasten:**
- `[R]` - Aufnahme starten/stoppen
- `[S]` - Screenshot (speichert in `data/raw_images/`)
- `[G]` - Schwarz/Weiß Modus
- `[A]` - Bräunungsanalyse ein/aus
- `[E/D]` - Belichtung +/-
- `[ESC]` - Beenden

### 2️⃣ Bilder zuschneiden
```bash
python 1_crop_images.py
```
- Zeigt Vorschau des Crop-Bereichs
- Passe `CROP_X`, `CROP_Y`, `CROP_WIDTH`, `CROP_HEIGHT` an

### 3️⃣ Bilder labeln
```bash
python 2_label_images.py
```
**Tasten:**
- `[1]` = roh 🍞
- `[2]` = leicht 🥪
- `[3]` = perfekt ✅
- `[4]` = dunkel 🔥
- `[5]` = verbrannt 💀
- `[S]` = überspringen
- `[ESC]` = beenden

### 4️⃣ Modell trainieren
```bash
python 3_train_model.py
```
- Trainiert CNN auf gelabelten Daten
- Speichert bestes Modell in `models/toast_model.pt`

### 5️⃣ Live-Erkennung
```bash
python 4_live_prediction.py
```
- Echtzeit-Klassifikation mit Kamera
- Zeigt Konfidenz und Warnungen

## 🔌 Shelly Plug S: Auto-Abschalten (Toast auswerfen)

In `manual_training/4_live_prediction.py` kannst du jetzt eine **Ziel-Bräunestufe (5 Stufen)** wählen und optional den **Shelly Plug S** nutzen, um bei Ziel-Erreichung den Strom abzuschalten.

- **Konfiguration**: kopiere `shelly_config.example.json` → `shelly_config.json` und trage die IP ein.
- **Live-Steuerung**:
  - `[1]-[5]`: Zielstufe setzen (roh → verbrannt)
  - `[A]`: Auto-Off an/aus
  - `[LEERTASTE]`: Session starten (schaltet Shelly **EIN** + reset; ab dann wird bei Ziel-Erreichung Shelly **AUS** gesendet)
  - `[O]` / `[F]`: Shelly manuell EIN/AUS

Alternativ per CLI:

```bash
python manual_training/4_live_prediction.py --level 3 --min-conf 70 --stable-frames 12 --shelly-host 192.168.178.50
```

## 📊 Klassen

| Klasse | Beschreibung | Farbe |
|--------|--------------|-------|
| 🍞 roh | Nicht getoastet | Grau |
| 🥪 leicht | Leicht gebräunt | Gelb |
| ✅ perfekt | Goldbraun | Grün |
| 🔥 dunkel | Zu dunkel | Orange |
| 💀 verbrannt | Verbrannt | Rot |

## ⚙️ Anforderungen

```bash
pip install opencv-python torch torchvision numpy matplotlib
```

## 💡 Tipps

- **Mindestens 20-30 Bilder pro Klasse** für gutes Training
- **Kamera fixieren** für konsistente Bilder
- **Belichtung anpassen** um Heizstäbe nicht zu überstrahlen
- **Verschiedene Toast-Sorten** für robusteres Modell
