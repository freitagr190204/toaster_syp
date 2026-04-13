import argparse
import cv2
import os
import datetime
import time
import numpy as np
from pathlib import Path

# ================= PFADE =================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Output-Ordner
VIDEOS_FOLDER = DATA_DIR / "videos"
IMAGES_FOLDER = DATA_DIR / "raw_images"  # Wird von 1_crop_images.py verwendet

# Ordner erstellen
VIDEOS_FOLDER.mkdir(parents=True, exist_ok=True)
IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)

# Größe des Ausrichtungs-Rechtecks (Region of Interest)
# Passt diese Werte an die Größe eures Toaster-Fensters an
BOX_WIDTH = 550
BOX_HEIGHT = 450

# Bräunungsstufen (Helligkeit 0-255, je dunkler desto brauner)
TOAST_LEVELS = {
    "🍞 ROH": (200, 255),        # Sehr hell - noch nicht getoastet
    "🥪 LEICHT": (150, 200),     # Leicht gebräunt
    "🍞 PERFEKT": (100, 150),    # Goldbraun - ideal!
    "🔥 DUNKEL": (50, 100),      # Zu dunkel
    "💀 VERBRANNT": (0, 50)      # Verbrannt!
}

# ================= KAMERA SETUP =================
def init_camera(preferred_ids=None):
    print("Suche Kamera...")
    # Wenn eine Kamera-ID explizit übergeben wurde, nur diese testen.
    if preferred_ids is None or len(preferred_ids) == 0:
        # Standard: Versuche Kamera 1 (Extern), dann 0 (Intern), dann 2
        preferred_ids = [1, 0, 2]

    print(f"Reihenfolge: {preferred_ids}")
    for idx in preferred_ids:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                # Versuchen den Autofokus abzustellen (funktioniert bei den meisten Webcams)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
                print(f"✅ Kamera {idx} bereit! (Auflösung: 1280x720)")
                return cap, idx
        cap.release()
    print("❌ Keine Kamera gefunden!")
    return None, -1

def get_toast_level(brightness):
    """Bestimmt die Bräunungsstufe basierend auf der Helligkeit"""
    for level, (low, high) in TOAST_LEVELS.items():
        if low <= brightness < high:
            return level
    return "🍞 ROH"

def get_level_color(brightness):
    """Gibt eine Farbe basierend auf der Bräunung zurück (BGR)"""
    if brightness >= 200:
        return (200, 200, 200)  # Grau - roh
    elif brightness >= 150:
        return (0, 255, 255)    # Gelb - leicht
    elif brightness >= 100:
        return (0, 255, 0)      # Grün - perfekt!
    elif brightness >= 50:
        return (0, 165, 255)    # Orange - dunkel
    else:
        return (0, 0, 255)      # Rot - verbrannt!

parser = argparse.ArgumentParser(description="Toast Recorder (Video + Helligkeitsanalyse)")
parser.add_argument("--camera-id", type=int, default=None, help="Feste Kamera-ID (z.B. 1 für USB-Kamera).")
args = parser.parse_args()

if args.camera_id is not None:
    cap, cam_idx = init_camera([args.camera_id])
else:
    cap, cam_idx = init_camera()
if cap is None:
    exit()

# ================= VARIABLEN =================
recording = False
video_writer = None
start_time = 0
last_snapshot_time = 0
session_id = ""
grayscale_mode = False  # Schwarz/Weiß Modus
show_analysis = True    # Bräunungsanalyse anzeigen

print("\n=== TOAST-RECORDER GESTARTET ===")
print("Steuerung im Videofenster:")
print(" [R] - Aufnahme START / STOP")
print(" [S] - Manueller Screenshot")
print(" [G] - Schwarz/Weiß Modus umschalten")
print(" [A] - Bräunungsanalyse ein/aus")
print(" [E] - Belichtung heller (+)")
print(" [D] - Belichtung dunkler (-)")
print(" [ESC] - Beenden")

# ================= HAUPTSCHLEIFE =================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamerafehler. Beende...")
        break

    # Originalbild für die Speicherung kopieren (ohne Text/Rahmen)
    clean_frame = frame.copy()
    h, w = frame.shape[:2]

    # --- Schwarz/Weiß Modus ---
    if grayscale_mode:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        clean_frame = frame.copy()

    # --- Overlay & Fadenkreuz zeichnen ---
    center_x, center_y = w // 2, h // 2
    x1 = center_x - BOX_WIDTH // 2
    y1 = center_y - BOX_HEIGHT // 2
    x2 = center_x + BOX_WIDTH // 2
    y2 = center_y + BOX_HEIGHT // 2

    # ROI (Region of Interest) für Bräunungsanalyse
    roi = frame[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(roi_gray)
    toast_level = get_toast_level(avg_brightness)
    level_color = get_level_color(avg_brightness)

    # Den Bereich für das Toaster-Fenster markieren
    cv2.rectangle(frame, (x1, y1), (x2, y2), level_color, 3)
    cv2.putText(frame, "Toaster-Fenster hier ausrichten", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- Bräunungsanalyse anzeigen ---
    if show_analysis:
        # Helligkeitsbalken zeichnen
        bar_x = x2 + 20
        bar_width = 30
        bar_height = BOX_HEIGHT
        
        # Hintergrund des Balkens
        cv2.rectangle(frame, (bar_x, y1), (bar_x + bar_width, y2), (50, 50, 50), -1)
        
        # Aktuelle Helligkeit markieren
        brightness_y = int(y2 - (avg_brightness / 255) * bar_height)
        cv2.rectangle(frame, (bar_x, brightness_y), (bar_x + bar_width, y2), level_color, -1)
        cv2.line(frame, (bar_x - 5, brightness_y), (bar_x + bar_width + 5, brightness_y), (255, 255, 255), 2)
        
        # Beschriftungen
        cv2.putText(frame, "HELL", (bar_x - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "DUNKEL", (bar_x - 15, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Toast-Status anzeigen
        cv2.putText(frame, f"{toast_level}", (x1, y2 + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, level_color, 2)
        cv2.putText(frame, f"Helligkeit: {avg_brightness:.0f}/255", (x1, y2 + 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Modus-Anzeige ---
    mode_text = "SW" if grayscale_mode else "FARBE"
    cv2.putText(frame, f"[{mode_text}]", (w - 100, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Aufnahme-Logik ---
    if recording:
        # 1. Video schreiben
        video_writer.write(clean_frame)
        
        # Laufzeit berechnen
        elapsed = int(time.time() - start_time)
        mins, secs = divmod(elapsed, 60)
        
        # Roter Punkt und Timer
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, f"REC {mins:02d}:{secs:02d}", (50, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # # 2. Automatischer Screenshot alle 5 Sekunden
        # if elapsed > 0 and elapsed % 5 == 0 and (time.time() - last_snapshot_time) >= 1:
        #     img_name = f"toaster_bilder/{session_id}_sec_{elapsed:03d}.png"
        #     cv2.imwrite(img_name, clean_frame)
        #     last_snapshot_time = time.time()
        #     print(f"📸 Auto-Bild gespeichert: Sekunde {elapsed}")

    else:
        cv2.putText(frame, "Bereit. [R] druecken fuer Aufnahme", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # --- Anzeige ---
    cv2.imshow("🍞 Toaster Cam", frame)

    # --- Tastensteuerung ---
    key = cv2.waitKey(1) & 0xFF

    if key == 27: # ESC
        break
    elif key == ord('r'):
        recording = not recording
        if recording:
            session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            vid_name = str(VIDEOS_FOLDER / f"toastlauf_{session_id}.avi")
            video_writer = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'XVID'), 30, (w, h))
            start_time = time.time()
            last_snapshot_time = start_time
            print(f"🎥 AUFNAHME LÄUFT: {vid_name}")
        else:
            video_writer.release()
            print("⏹ AUFNAHME GESTOPPT.")

    elif key == ord('s'):
        # Manueller Screenshot
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = str(IMAGES_FOLDER / f"toast_{ts}.png")
        cv2.imwrite(img_path, clean_frame)
        print(f"📸 Screenshot gespeichert: {img_path}")

    elif key == ord('e'): # Belichtung hoch
        current_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        cap.set(cv2.CAP_PROP_EXPOSURE, current_exp + 1)
        print(f"Belichtung erhöht auf: {current_exp + 1}")

    elif key == ord('d'): # Belichtung runter
        current_exp = cap.get(cv2.CAP_PROP_EXPOSURE)
        cap.set(cv2.CAP_PROP_EXPOSURE, current_exp - 1)
        print(f"Belichtung gesenkt auf: {current_exp - 1}")

    elif key == ord('g'): # Schwarz/Weiß Modus umschalten
        grayscale_mode = not grayscale_mode
        mode = "SCHWARZ/WEISS" if grayscale_mode else "FARBE"
        print(f"🎨 Modus: {mode}")

    elif key == ord('a'): # Analyse ein/aus
        show_analysis = not show_analysis
        status = "EIN" if show_analysis else "AUS"
        print(f"📊 Bräunungsanalyse: {status}")

# ================= AUFRÄUMEN =================
cap.release()
if video_writer is not None:
    video_writer.release()
cv2.destroyAllWindows()