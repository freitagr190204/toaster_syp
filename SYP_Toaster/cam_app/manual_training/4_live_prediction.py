"""
Live Toast-Erkennung + Shelly Plug S Auto-Abschaltung
=====================================================
Kamera schaut durch Glasfenster in den Toaster.
Du wählst per Taste [1]-[5] wie braun dein Toast werden soll,
drückst den Toaster-Hebel runter, und sobald die Zielstufe
stabil erkannt wird, schaltet der Shelly Plug S den Strom ab
→ Toast wird ausgeworfen.

Steuerung:
  [1]-[5]      Zielstufe wählen (1=roh … 5=verbrannt)
  [LEERTASTE]  Neue Session starten (Reset Trigger, Shelly EIN)
  [O] / [F]    Shelly manuell EIN / AUS
  [G]          Anzeige Farbe ↔ S/W
  [R]          Trigger zurücksetzen
  [ESC]        Beenden

Workflow: recording.py → 1_crop_images.py → 2_label_images.py → 3_train_model.py → 4_live_prediction.py
"""

import argparse
import sys
import cv2
import json
import torch
import numpy as np
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

# ================= PFADE =================
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import toast_net

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "toast_model.pt"

# ================= KONSTANTEN =================
IMG_SIZE = 224
CLASSES = ['roh', 'leicht', 'perfekt', 'dunkel', 'verbrannt']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BOX_WIDTH = 550
BOX_HEIGHT = 450

CLASS_COLORS = {
    'roh':       (200, 200, 200),
    'leicht':    (0, 255, 255),
    'perfekt':   (0, 255, 0),
    'dunkel':    (0, 165, 255),
    'verbrannt': (0, 0, 255),
}

CLASS_LABELS = {
    'roh':       '1: ROH',
    'leicht':    '2: LEICHT',
    'perfekt':   '3: PERFEKT',
    'dunkel':    '4: DUNKEL',
    'verbrannt': '5: VERBRANNT',
}

# ================= SHELLY PLUG S =================
@dataclass(frozen=True)
class ShellyConfig:
    host: str
    relay: int = 0
    username: Optional[str] = None
    password: Optional[str] = None
    timeout_s: float = 2.0


class ShellyClient:
    """Shelly Plug S HTTP API (Gen1 + Gen2 fallback)."""

    def __init__(self, cfg: ShellyConfig):
        self.cfg = cfg

    def _add_auth(self, req: urllib.request.Request) -> None:
        if self.cfg.username and self.cfg.password is not None:
            import base64
            token = f"{self.cfg.username}:{self.cfg.password}".encode("utf-8")
            req.add_header("Authorization", "Basic " + base64.b64encode(token).decode("ascii"))

    def set_power(self, on: bool) -> Tuple[bool, str]:
        turn = "on" if on else "off"
        # Gen1: /relay/0?turn=on|off
        url1 = f"http://{self.cfg.host}/relay/{self.cfg.relay}?turn={turn}"
        req1 = urllib.request.Request(url1, method="GET")
        self._add_auth(req1)
        try:
            with urllib.request.urlopen(req1, timeout=self.cfg.timeout_s) as resp:
                return True, resp.read().decode("utf-8", errors="replace")
        except Exception as e1:
            pass
        # Gen2: /rpc/Switch.Set?id=0&on=true|false
        on_str = "true" if on else "false"
        url2 = f"http://{self.cfg.host}/rpc/Switch.Set?id={self.cfg.relay}&on={on_str}"
        req2 = urllib.request.Request(url2, method="GET")
        self._add_auth(req2)
        try:
            with urllib.request.urlopen(req2, timeout=self.cfg.timeout_s) as resp:
                return True, resp.read().decode("utf-8", errors="replace")
        except Exception as e2:
            return False, f"Gen1: {e1}; Gen2: {e2}"


def load_shelly_config() -> Optional[ShellyConfig]:
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
            username=(str(data["username"]) if data.get("username") else None),
            password=(str(data["password"]) if data.get("password") else None),
            timeout_s=float(data.get("timeout_s", 2.0)),
        )
    except Exception:
        return None


# ================= MODELL =================
def load_model():
    if not MODEL_PATH.exists():
        print(f"Modell nicht gefunden: {MODEL_PATH}")
        print("   Bitte zuerst 3_train_model.py ausfuehren!")
        return None
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    if checkpoint.get("arch") != "resnet18":
        print("Altes toast_model.pt (Custom-CNN) ist nicht kompatibel.")
        print("   Bitte neu trainieren: python manual_training/3_train_model.py")
        return None
    model = toast_net.build_toast_classifier(len(CLASSES), pretrained_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"ResNet18 geladen (Val-Accuracy beim Training: {checkpoint['accuracy']:.1f}%)")
    return model


# ================= KAMERA =================
def init_camera(preferred_ids=None):
    if preferred_ids is None or len(preferred_ids) == 0:
        preferred_ids = [1, 0, 2]
    print(f"Suche Kamera (Reihenfolge: {preferred_ids})...")
    for idx in preferred_ids:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                print(f"Kamera {idx} bereit!")
                return cap
        cap.release()
    print("Keine Kamera gefunden!")
    return None


# ================= VORHERSAGE =================
def predict(model, roi):
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    img = toast_net.normalize_imagenet(img)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100
    probs_np = probs[0].cpu().numpy()
    expected_state = float(np.sum(np.arange(len(CLASSES), dtype=np.float32) * probs_np))
    return pred_idx, confidence, probs_np, expected_state


def target_reached(target_idx: int, pred_idx: int, expected_state: float, confidence: float, min_conf: float) -> bool:
    """
    Returns True when the toast has reached (or passed) the chosen target level.
    - target 0 (roh):       pred >= 0 and expected >= 0.0  (basically always, but still need confidence)
    - target 1 (leicht):    expected >= 0.8
    - target 2 (perfekt):   expected >= 1.8
    - target 3 (dunkel):    expected >= 2.8
    - target 4 (verbrannt): expected >= 3.5
    """
    thresholds = [0.0, 0.8, 1.8, 2.8, 3.5]
    if confidence < min_conf:
        return False
    return expected_state >= thresholds[target_idx]


# ================= HAUPTPROGRAMM =================
def main():
    parser = argparse.ArgumentParser(description="Live Toast-Erkennung + Shelly Auto-Off")
    parser.add_argument("--level", type=int, default=3, help="Ziel-Stufe 1-5 (1=roh..5=verbrannt). Default: 3 (perfekt)")
    parser.add_argument("--min-conf", type=float, default=65.0, help="Min. Konfidenz (%%) Default: 65")
    parser.add_argument("--stable-frames", type=int, default=10, help="Stabile Frames bevor Strom aus. Default: 10")
    parser.add_argument("--camera-id", type=int, default=None, help="Kamera-ID (z.B. 1 fuer USB)")
    parser.add_argument("--shelly-host", type=str, default=None, help="Shelly IP (ueberschreibt shelly_config.json)")
    args = parser.parse_args()

    print("=" * 55)
    print("    TOAST AI - Automatische Braeunungserkennung")
    print("=" * 55)

    model = load_model()
    if model is None:
        return

    cap = init_camera([args.camera_id] if args.camera_id is not None else None)
    if cap is None:
        return

    # Shelly
    shelly_cfg = load_shelly_config()
    if args.shelly_host:
        shelly_cfg = ShellyConfig(host=args.shelly_host.strip())
    shelly: Optional[ShellyClient] = ShellyClient(shelly_cfg) if shelly_cfg else None

    # State
    target_idx = int(np.clip(args.level, 1, 5)) - 1
    min_conf = float(args.min_conf)
    stable_needed = int(args.stable_frames)
    consecutive = 0
    power_off_sent = False
    session_active = False
    grayscale_mode = False
    shelly_status = "Shelly: nicht konfiguriert" if shelly is None else "Shelly: bereit"

    print()
    print("Steuerung:")
    print("  [1]-[5]      Zielstufe waehlen")
    print("  [LEERTASTE]  Session starten (Shelly EIN + Reset)")
    print("  [O] / [F]    Shelly manuell EIN / AUS")
    print("  [G]          Farbe / S/W")
    print("  [R]          Trigger reset")
    print("  [ESC]        Beenden")
    print("=" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        color_frame = frame
        h, w = color_frame.shape[:2]

        # ROI (centered, slightly below middle for typical toaster view)
        cx, cy = w // 2, h // 2 + 50
        x1, y1 = cx - BOX_WIDTH // 2, cy - BOX_HEIGHT // 2
        x2, y2 = cx + BOX_WIDTH // 2, cy + BOX_HEIGHT // 2

        roi = color_frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        pred_idx, confidence, all_probs, expected_state = predict(model, roi)
        pred_class = CLASSES[pred_idx]

        # Display frame (optionally grayscale)
        if grayscale_mode:
            gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            frame = color_frame.copy()

        color = CLASS_COLORS[pred_class]
        target_name = CLASSES[target_idx]
        target_color = CLASS_COLORS[target_name]

        # ROI box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Prediction label
        cv2.putText(frame, f"{CLASS_LABELS[pred_class]}  {confidence:.0f}%", (x1, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # ---- Right side: probability bars ----
        bar_x = x2 + 25
        bar_w = 160
        bar_h = 28
        cv2.putText(frame, "Vorhersage:", (bar_x, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for i, (cls, prob) in enumerate(zip(CLASSES, all_probs)):
            by = y1 + 15 + i * (bar_h + 8)
            cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w, by + bar_h), (50, 50, 50), -1)
            fill = int(prob * bar_w)
            cv2.rectangle(frame, (bar_x, by), (bar_x + fill, by + bar_h), CLASS_COLORS[cls], -1)
            cv2.putText(frame, f"{cls}: {prob*100:.0f}%", (bar_x + 5, by + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            if i == target_idx:
                cv2.rectangle(frame, (bar_x - 2, by - 2), (bar_x + bar_w + 2, by + bar_h + 2), (255, 255, 255), 2)

        # ---- Bottom: status info ----
        info_y = y2 + 30
        cv2.putText(frame, f"ZIEL: {CLASS_LABELS[target_name]}  [Taste 1-5 zum aendern]", (x1, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, target_color, 2)
        info_y += 28
        session_txt = "AKTIV" if session_active else "WARTE (Leertaste)"
        cv2.putText(frame, f"Session: {session_txt}  |  Erwartungswert: {expected_state:.2f}", (x1, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        info_y += 22
        cv2.putText(frame, shelly_status, (x1, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ---- Auto-off logic ----
        if session_active and not power_off_sent:
            reached = target_reached(target_idx, pred_idx, expected_state, confidence, min_conf)
            if reached:
                consecutive += 1
            else:
                consecutive = 0

            if consecutive > 0:
                progress_pct = min(100, int(consecutive / stable_needed * 100))
                cv2.putText(frame, f"Ziel: {consecutive}/{stable_needed} ({progress_pct}%)",
                            (w // 2 - 180, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, target_color, 2)

            if consecutive >= stable_needed:
                if shelly is not None:
                    ok, msg = shelly.set_power(False)
                    shelly_status = "Shelly: AUS  (Toast fertig!)" if ok else f"Shelly OFF Fehler: {msg}"
                else:
                    shelly_status = "TOAST FERTIG! (kein Shelly konfiguriert)"
                power_off_sent = True
                session_active = False
                consecutive = 0

        if power_off_sent:
            cv2.putText(frame, "STROM AUS - TOAST FERTIG!", (w // 2 - 250, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Safety warning
        if pred_class == 'verbrannt' and confidence > 50:
            cv2.putText(frame, "!!! ACHTUNG VERBRANNT !!!", (w // 2 - 200, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        cv2.imshow("Toast AI - Live Erkennung", frame)

        # ---- Key handling ----
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

        if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            target_idx = int(chr(key)) - 1
            consecutive = 0
            power_off_sent = False

        elif key == ord(' '):
            power_off_sent = False
            consecutive = 0
            session_active = True
            if shelly is not None:
                ok, msg = shelly.set_power(True)
                shelly_status = "Shelly: EIN (Session gestartet)" if ok else f"Shelly ON Fehler: {msg}"
            else:
                shelly_status = "Session gestartet (kein Shelly)"

        elif key == ord('o'):
            if shelly is not None:
                ok, msg = shelly.set_power(True)
                shelly_status = "Shelly: EIN" if ok else f"Shelly ON Fehler: {msg}"

        elif key == ord('f'):
            if shelly is not None:
                ok, msg = shelly.set_power(False)
                shelly_status = "Shelly: AUS" if ok else f"Shelly OFF Fehler: {msg}"
            session_active = False
            consecutive = 0
            power_off_sent = False

        elif key == ord('r'):
            consecutive = 0
            power_off_sent = False

        elif key == ord('g'):
            grayscale_mode = not grayscale_mode

    cap.release()
    cv2.destroyAllWindows()
    print("\nBeendet.")


if __name__ == "__main__":
    main()
