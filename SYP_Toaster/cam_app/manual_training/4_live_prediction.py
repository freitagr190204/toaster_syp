"""
Skript 4: Live Toast-Erkennung mit trainiertem Modell
=====================================================
Nutzt das trainierte CNN für Echtzeit-Vorhersagen.

Workflow: recording.py -> 1_crop_images.py -> 2_label_images.py -> 3_train_model.py -> 4_live_prediction.py
"""

import argparse
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

# ================= PFADE =================
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "toast_model.pt"

# ================= KONSTANTEN =================
IMG_SIZE = 224
CLASSES = ['roh', 'leicht', 'perfekt', 'dunkel', 'verbrannt']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= EINSTELLUNGEN =================
BOX_WIDTH = 550
BOX_HEIGHT = 450

# Farben für Klassen (BGR)
CLASS_COLORS = {
    'roh': (200, 200, 200),      # Grau
    'leicht': (0, 255, 255),     # Gelb
    'perfekt': (0, 255, 0),      # Grün
    'dunkel': (0, 165, 255),     # Orange
    'verbrannt': (0, 0, 255)     # Rot
}

CLASS_EMOJIS = {
    'roh': '🍞 ROH',
    'leicht': '🥪 LEICHT',
    'perfekt': '✅ PERFEKT!',
    'dunkel': '🔥 DUNKEL',
    'verbrannt': '💀 VERBRANNT'
}

# ================= SHELLY (PLUG S / Gen1) =================
@dataclass(frozen=True)
class ShellyConfig:
    host: str  # z.B. "192.168.178.50"
    relay: int = 0
    username: Optional[str] = None
    password: Optional[str] = None
    timeout_s: float = 2.0


class ShellyClient:
    """
    Shelly Plug S HTTP API:
      - Gen1:  GET http://<host>/relay/0?turn=on|off
      - Gen2:  GET http://<host>/rpc/Switch.Set?id=0&on=true|false
    Optional HTTP Basic Auth (wenn im Shelly gesetzt).
    """

    def __init__(self, cfg: ShellyConfig):
        self.cfg = cfg

    def _add_auth(self, req: urllib.request.Request) -> None:
        if self.cfg.username and self.cfg.password is not None:
            import base64

            token = f"{self.cfg.username}:{self.cfg.password}".encode("utf-8")
            req.add_header("Authorization", "Basic " + base64.b64encode(token).decode("ascii"))

    def _build_request_gen1(self, turn: str) -> urllib.request.Request:
        url = f"http://{self.cfg.host}/relay/{self.cfg.relay}?turn={urllib.parse.quote(turn)}"
        req = urllib.request.Request(url, method="GET")
        self._add_auth(req)
        return req

    def _build_request_gen2(self, on: bool) -> urllib.request.Request:
        # Shelly Gen2 / Plus API
        on_str = "true" if on else "false"
        url = f"http://{self.cfg.host}/rpc/Switch.Set?id={self.cfg.relay}&on={on_str}"
        req = urllib.request.Request(url, method="GET")
        self._add_auth(req)
        return req

    def set_power(self, on: bool) -> Tuple[bool, str]:
        turn = "on" if on else "off"
        # 1) Versuche klassische Gen1-API
        req1 = self._build_request_gen1(turn)
        try:
            with urllib.request.urlopen(req1, timeout=self.cfg.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            return True, body
        except Exception as e1:
            # 2) Fallback: Gen2-/rpc-API probieren
            req2 = self._build_request_gen2(on)
            try:
                with urllib.request.urlopen(req2, timeout=self.cfg.timeout_s) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                return True, body
            except Exception as e2:
                return False, f"Gen1-Fehler: {e1}; Gen2-Fehler: {e2}"


def load_shelly_config() -> Optional[ShellyConfig]:
    """
    Lädt `shelly_config.json` aus dem Projekt-Root (cam_app/).
    Beispiel siehe `shelly_config.example.json`.
    """
    cfg_path = BASE_DIR / "shelly_config.json"
    if not cfg_path.exists():
        return None
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        host = str(data.get("host", "")).strip()
        if not host:
            return None
        return ShellyConfig(
            host=host,
            relay=int(data.get("relay", 0)),
            username=(str(data["username"]) if "username" in data and str(data["username"]).strip() else None),
            password=(str(data["password"]) if "password" in data and data["password"] is not None else None),
            timeout_s=float(data.get("timeout_s", 2.0)),
        )
    except Exception:
        return None


# ================= CNN MODELL (kopiert aus 3_train_model.py) =================
class ToastCNN(nn.Module):
    """Einfaches CNN für Toast-Klassifikation"""
    def __init__(self, num_classes=5):
        super(ToastCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ================= MODELL LADEN =================
def load_model():
    """Lädt das trainierte Modell"""
    if not MODEL_PATH.exists():
        print(f"❌ Modell nicht gefunden: {MODEL_PATH}")
        print("   Bitte zuerst Skript 3 (3_train_model.py) ausführen!")
        return None
    
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = ToastCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Modell geladen (Accuracy: {checkpoint['accuracy']:.1f}%)")
    return model

# ================= KAMERA =================
def init_camera(preferred_ids=None):
    """
    Initialisiert die Kamera.
    - Wenn preferred_ids gesetzt ist, werden genau diese IDs in dieser Reihenfolge probiert.
    - Sonst Standard-Reihenfolge: [1, 0, 2] (extern, intern, weitere).
    """
    if preferred_ids is None or len(preferred_ids) == 0:
        preferred_ids = [1, 0, 2]

    print(f"Suche Kamera (Reihenfolge: {preferred_ids})...")
    for idx in preferred_ids:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                print(f"✅ Kamera {idx} bereit!")
                return cap
        cap.release()
    print("❌ Keine Kamera gefunden!")
    return None

# ================= VORHERSAGE =================
def predict(model, roi):
    """Macht Vorhersage für ROI"""
    # Bild vorbereiten
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Vorhersage
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100
    
    probs_np = probs[0].cpu().numpy()
    expected_state = float(np.sum(np.arange(len(CLASSES), dtype=np.float32) * probs_np))
    return CLASSES[pred_idx], pred_idx, confidence, probs_np, expected_state

# ================= HAUPTPROGRAMM =================
def main():
    parser = argparse.ArgumentParser(description="Live Toast-Erkennung (+ optional Shelly Auto-Off)")
    parser.add_argument("--level", type=int, default=3, help="Ziel-Bräunestufe 1-5 (1=roh ... 5=verbrannt). Default: 3")
    parser.add_argument("--min-conf", type=float, default=70.0, help="Min. Konfidenz (%%) fürs Auslösen. Default: 70")
    parser.add_argument("--stable-frames", type=int, default=12, help="Wie viele Frames am Stück Ziel erreicht sein muss. Default: 12")
    parser.add_argument("--shelly-host", type=str, default=None, help='Shelly Host/IP, z.B. "192.168.178.50". Überschreibt shelly_config.json')
    parser.add_argument("--shelly-relay", type=int, default=None, help="Shelly Relay (meist 0). Überschreibt shelly_config.json")
    parser.add_argument("--shelly-user", type=str, default=None, help="Optional Shelly Username (Basic Auth)")
    parser.add_argument("--shelly-pass", type=str, default=None, help="Optional Shelly Passwort (Basic Auth)")
    parser.add_argument("--camera-id", type=int, default=None, help="Feste Kamera-ID (z.B. 1 für USB-Kamera). Wenn gesetzt, wird nur diese ID verwendet.")
    args = parser.parse_args()

    print("="*50)
    print("🍞 LIVE TOAST-ERKENNUNG")
    print("="*50)
    
    # Modell laden
    model = load_model()
    if model is None:
        return
    
    # Kamera starten
    if args.camera_id is not None:
        cap = init_camera([args.camera_id])
    else:
        cap = init_camera()
    if cap is None:
        return
    
    print("\nSteuerung:")
    print(" [ESC] - Beenden")
    print(" [1]-[5] - Zielstufe setzen (roh..verbrannt)")
    print(" [A] - Auto-Off (Shelly) an/aus")
    print(" [LEERTASTE] - (wenn Auto-Off an) Session starten: Shelly EIN + Trigger reset")
    print(" [O] - Shelly EIN (manuell)")
    print(" [F] - Shelly AUS (manuell)")
    print(" [R] - Trigger zurücksetzen (erneut auslösen erlauben)")
    print(" [G] - Anzeige: Farbe <-> S/W")
    print("="*50)

    # Zielstufe 1..5 -> Index 0..4
    target_idx = int(np.clip(args.level, 1, 5)) - 1

    # Shelly Config (optional)
    shelly_cfg = load_shelly_config()
    if args.shelly_host:
        shelly_cfg = ShellyConfig(
            host=args.shelly_host.strip(),
            relay=int(args.shelly_relay) if args.shelly_relay is not None else (shelly_cfg.relay if shelly_cfg else 0),
            username=args.shelly_user,
            password=args.shelly_pass,
            timeout_s=(shelly_cfg.timeout_s if shelly_cfg else 2.0),
        )
    shelly: Optional[ShellyClient] = ShellyClient(shelly_cfg) if shelly_cfg else None

    # Auto-Modus: Strom automatisch EIN beim Start (falls Shelly konfiguriert),
    # und automatisch AUS schalten, wenn der Toast zwischen Stufe 3 ("perfekt")
    # und 4 ("dunkel") liegt (kontinuierlicher Erwartungswert zwischen 2.5 und 3.5).
    auto_off_enabled = shelly is not None
    session_active = auto_off_enabled
    consecutive_reached = 0
    power_off_sent = False
    if shelly is None:
        last_shelly_status = "Shelly: nicht konfiguriert"
    else:
        ok, msg = shelly.set_power(True)
        last_shelly_status = "Shelly: EIN ✅ (Auto-Start)" if ok else f"Shelly ON Fehler: {msg}"
    session_started_at = None
    grayscale_mode = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Original für CNN behalten
        color_frame = frame
        h, w = color_frame.shape[:2]
        
        # ROI berechnen (leicht nach unten versetzt)
        center_x, center_y = w // 2, h // 2 + 50
        x1 = center_x - BOX_WIDTH // 2
        y1 = center_y - BOX_HEIGHT // 2
        x2 = center_x + BOX_WIDTH // 2
        y2 = center_y + BOX_HEIGHT // 2
        
        # ROI extrahieren und Vorhersage machen
        roi = color_frame[y1:y2, x1:x2]
        pred_class, pred_idx, confidence, all_probs, expected_state = predict(model, roi)
        
        # Frame für Anzeige vorbereiten (optional S/W)
        if grayscale_mode:
            gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            frame = color_frame.copy()

        # Farbe basierend auf Vorhersage
        color = CLASS_COLORS[pred_class]
        
        # ROI-Rahmen zeichnen
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Vorhersage anzeigen
        label = CLASS_EMOJIS[pred_class]
        cv2.putText(frame, f"{label}", (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        cv2.putText(frame, f"Konfidenz: {confidence:.1f}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Target/Auto-Off Status Overlay
        target_name = CLASSES[target_idx]
        target_color = CLASS_COLORS[target_name]
        status_y = y2 + 85
        cv2.putText(frame, f"Auto-Modus: AUS wenn 'perfekt'/'dunkel' mit >80% erkannt", (x1, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, target_color, 2)
        status_y += 25
        cv2.putText(frame, f"Auto-Off: {'AN' if auto_off_enabled else 'AUS'} | Session: {'AKTIV' if session_active else 'STOP'} | Erwartungswert: {expected_state:.2f}",
                    (x1, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        status_y += 22
        cv2.putText(frame, f"{last_shelly_status}", (x1, status_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Balkendiagramm für alle Klassen
        bar_x = x2 + 30
        bar_width = 150
        bar_height = 25
        
        cv2.putText(frame, "Vorhersage:", (bar_x, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, (cls, prob) in enumerate(zip(CLASSES, all_probs)):
            y_pos = y1 + 20 + i * (bar_height + 10)
            
            # Hintergrund
            cv2.rectangle(frame, (bar_x, y_pos), 
                         (bar_x + bar_width, y_pos + bar_height),
                         (50, 50, 50), -1)
            
            # Füllbalken
            fill_width = int(prob * bar_width)
            bar_color = CLASS_COLORS[cls]
            cv2.rectangle(frame, (bar_x, y_pos),
                          (bar_x + fill_width, y_pos + bar_height),
                          bar_color, -1)
            
            # Label
            cv2.putText(frame, f"{cls}: {prob*100:.0f}%", 
                       (bar_x + 5, y_pos + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Markiere Zielklasse
            if i == target_idx:
                cv2.rectangle(frame, (bar_x - 3, y_pos - 3), (bar_x + bar_width + 3, y_pos + bar_height + 3), (255, 255, 255), 1)
        
        # Auto-Off Logik (nur wenn Session aktiv)
        if session_active and auto_off_enabled:
            # Ausschalten, wenn Klasse "perfekt" ODER "dunkel" mit hoher Sicherheit erkannt wird.
            # Klassenindizes: 0=roh, 1=leicht, 2=perfekt, 3=dunkel, 4=verbrannt
            # → Bedingung: pred_class in {'perfekt','dunkel'} UND Konfidenz >= 80%
            conf_threshold = max(float(args.min_conf), 80.0)
            reached = (pred_class in ('perfekt', 'dunkel')) and (confidence >= conf_threshold)
            if reached:
                consecutive_reached += 1
            else:
                consecutive_reached = 0

            if consecutive_reached > 0:
                cv2.putText(frame, f"Ziel erreicht: {consecutive_reached}/{int(args.stable_frames)} Frames",
                            (w//2 - 220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, target_color, 2)

            if (not power_off_sent) and consecutive_reached >= int(args.stable_frames):
                if shelly is None:
                    last_shelly_status = "Shelly: nicht konfiguriert (kein OFF gesendet)"
                    power_off_sent = True
                else:
                    ok, msg = shelly.set_power(False)
                    last_shelly_status = "Shelly: AUS ✅" if ok else f"Shelly OFF Fehler: {msg}"
                    power_off_sent = True
                    session_active = False
                    session_started_at = None
                    consecutive_reached = 0
                    cv2.putText(frame, ">>> STROM AUS - TOAST SOLLTE AUSWERFEN <<<", (w//2 - 330, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
        
        # Warnung bei verbrannt
        if pred_class in ['dunkel', 'verbrannt'] and confidence > 60:
            cv2.putText(frame, "!!! ACHTUNG - ZU DUNKEL !!!", (w//2 - 200, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        cv2.imshow("Toast AI - Live Erkennung", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        # Zielstufe live setzen
        if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            target_idx = int(chr(key)) - 1
            consecutive_reached = 0
            power_off_sent = False

        # Auto-Off an/aus
        elif key == ord('a'):
            auto_off_enabled = not auto_off_enabled
            if not auto_off_enabled:
                session_active = False
                session_started_at = None
                consecutive_reached = 0

        # Session Start (Shelly ON + reset)
        elif key == ord(' '):
            if auto_off_enabled:
                power_off_sent = False
                consecutive_reached = 0
                session_active = True
                session_started_at = time.time()
                if shelly is None:
                    last_shelly_status = "Shelly: nicht konfiguriert (kann nicht EIN schalten)"
                else:
                    ok, msg = shelly.set_power(True)
                    last_shelly_status = "Shelly: EIN ✅" if ok else f"Shelly ON Fehler: {msg}"

        # Manuelle Shelly Steuerung
        elif key == ord('o'):
            if shelly is None:
                last_shelly_status = "Shelly: nicht konfiguriert"
            else:
                ok, msg = shelly.set_power(True)
                last_shelly_status = "Shelly: EIN ✅" if ok else f"Shelly ON Fehler: {msg}"

        elif key == ord('f'):
            if shelly is None:
                last_shelly_status = "Shelly: nicht konfiguriert"
            else:
                ok, msg = shelly.set_power(False)
                last_shelly_status = "Shelly: AUS ✅" if ok else f"Shelly OFF Fehler: {msg}"
                session_active = False
                session_started_at = None
                consecutive_reached = 0
                power_off_sent = False

        # Trigger reset
        elif key == ord('r'):
            consecutive_reached = 0
            power_off_sent = False

        # Anzeige: Farbe / S/W
        elif key == ord('g'):
            grayscale_mode = not grayscale_mode
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Beendet.")

if __name__ == "__main__":
    main()
