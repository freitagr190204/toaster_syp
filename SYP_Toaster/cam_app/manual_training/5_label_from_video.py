"""
Skript 5: Videos durchschauen und dabei labeln
===============================================
Spielt alle Videos aus data/videos/ ab.
Du drueckst [1]-[5] um den aktuellen Braeunungsgrad zu setzen.
Alle paar Frames wird automatisch ein Bild mit dem aktuellen Label
direkt in data/labeled_dataset/{klasse}/ gespeichert.

Das ist VIEL schneller als einzelne Bilder zu labeln!

Steuerung:
  [1] roh   [2] leicht   [3] perfekt   [4] dunkel   [5] verbrannt
  [LEERTASTE]  Pause / Weiter
  [S]          Aktuelles Bild manuell speichern
  [N]          Naechstes Video
  [ESC]        Beenden

Workflow: recording.py -> 5_label_from_video.py -> 3_train_model.py -> 4_live_prediction.py
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# ================= PFADE =================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
VIDEOS_DIR = DATA_DIR / "videos"
OUTPUT_BASE = DATA_DIR / "labeled_dataset"

# ================= EINSTELLUNGEN =================
CLASSES = ['roh', 'leicht', 'perfekt', 'dunkel', 'verbrannt']
IMG_SIZE = 224

# Crop-Bereich (gleich wie in recording.py / 4_live_prediction.py)
BOX_WIDTH = 550
BOX_HEIGHT = 450

CLASS_COLORS = {
    0: (200, 200, 200),
    1: (0, 255, 255),
    2: (0, 255, 0),
    3: (0, 165, 255),
    4: (0, 0, 255),
}

SAVE_EVERY_N_FRAMES = 15  # ~2 Bilder pro Sekunde bei 30fps Video


def create_folders():
    for cls in CLASSES:
        (OUTPUT_BASE / cls).mkdir(parents=True, exist_ok=True)


def process_video(video_path, start_label=0, save_interval=SAVE_EVERY_N_FRAMES):
    """Spielt ein Video ab und labelt Frames interaktiv."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Kann Video nicht oeffnen: {video_path.name}")
        return 0, start_label

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration_s = total_frames / fps

    print(f"\n  Video: {video_path.name}")
    print(f"  Frames: {total_frames} | FPS: {fps:.0f} | Dauer: {duration_s:.0f}s")

    current_label = start_label
    saved_count = 0
    frame_idx = 0
    paused = False
    auto_save = True
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
        else:
            pass

        h, w = frame.shape[:2]

        # ROI (same as prediction script)
        cx, cy = w // 2, h // 2 + 50
        x1, y1 = max(0, cx - BOX_WIDTH // 2), max(0, cy - BOX_HEIGHT // 2)
        x2, y2 = min(w, cx + BOX_WIDTH // 2), min(h, cy + BOX_HEIGHT // 2)

        roi = frame[y1:y2, x1:x2]

        # Auto-save every N frames
        if auto_save and not paused and (frame_idx % save_interval == 0):
            resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            cls_name = CLASSES[current_label]
            fname = f"vid_{timestamp}_{video_path.stem}_f{frame_idx:06d}.png"
            save_path = OUTPUT_BASE / cls_name / fname
            cv2.imwrite(str(save_path), resized)
            saved_count += 1

        # Display
        display = frame.copy()
        color = CLASS_COLORS[current_label]
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)

        # Info bar at top
        cls_name = CLASSES[current_label]
        progress = frame_idx / max(total_frames, 1) * 100
        time_pos = frame_idx / fps

        cv2.rectangle(display, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(display, f"LABEL: {current_label+1}={cls_name.upper()}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"Gespeichert: {saved_count} | Frame: {frame_idx}/{total_frames} ({progress:.0f}%) | {time_pos:.1f}s/{duration_s:.0f}s",
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Pause indicator
        if paused:
            cv2.putText(display, "PAUSE", (w - 130, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # Key legend at bottom
        cv2.rectangle(display, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.putText(display, "[1]roh [2]leicht [3]perfekt [4]dunkel [5]verbrannt | [SPACE]Pause [N]Next [ESC]Quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # Progress bar
        bar_y = h - 38
        bar_w = int(progress / 100 * w)
        cv2.rectangle(display, (0, bar_y), (bar_w, bar_y + 3), color, -1)

        cv2.imshow("Video Labeler", display)

        # Key handling
        wait_ms = 1 if not paused else 50
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == 27:  # ESC
            cap.release()
            return saved_count, current_label

        if key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            current_label = int(chr(key)) - 1
            print(f"    Label -> {CLASSES[current_label]}")

        elif key == ord(' '):
            paused = not paused

        elif key == ord('s'):
            resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            cls_name = CLASSES[current_label]
            fname = f"vid_{timestamp}_{video_path.stem}_manual_{frame_idx:06d}.png"
            save_path = OUTPUT_BASE / cls_name / fname
            cv2.imwrite(str(save_path), resized)
            saved_count += 1
            print(f"    Manuell gespeichert: {cls_name}/{fname}")

        elif key == ord('n'):
            break

    cap.release()
    return saved_count, current_label


def main():
    parser = argparse.ArgumentParser(description="Videos anschauen und dabei labeln")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY_N_FRAMES,
                        help=f"Alle N Frames automatisch speichern. Default: {SAVE_EVERY_N_FRAMES}")
    args = parser.parse_args()

    save_interval = args.save_every

    print("=" * 55)
    print("    VIDEO LABELER - Toast-Braeunung")
    print("=" * 55)

    create_folders()

    videos = sorted(VIDEOS_DIR.glob("*.avi")) + sorted(VIDEOS_DIR.glob("*.mp4"))
    if not videos:
        print(f"Keine Videos in {VIDEOS_DIR} gefunden!")
        return

    print(f"\n{len(videos)} Videos gefunden:")
    for v in videos:
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"  {v.name} ({size_mb:.1f} MB)")

    print(f"\nAuto-Save alle {save_interval} Frames (~{30/save_interval:.1f} Bilder/Sek)")
    print("\nStarte mit Label '1=roh'. Aendere mit Tasten [1]-[5] waehrend das Video laeuft.")
    print("Typisch: Am Anfang [1], dann [2] wenn es leicht braun wird, [3] bei perfekt, usw.\n")

    total_saved = 0
    current_label = 0

    for i, video_path in enumerate(videos):
        print(f"\n--- Video {i+1}/{len(videos)} ---")
        saved, current_label = process_video(video_path, start_label=current_label, save_interval=save_interval)
        total_saved += saved
        print(f"  -> {saved} Bilder gespeichert (Gesamt: {total_saved})")

    cv2.destroyAllWindows()

    # Summary
    print("\n" + "=" * 55)
    print(f"FERTIG! {total_saved} Bilder insgesamt gespeichert.")
    print("\nVerteilung in labeled_dataset/:")
    for cls in CLASSES:
        folder = OUTPUT_BASE / cls
        count = len(list(folder.glob("*.png"))) + len(list(folder.glob("*.jpg")))
        print(f"  {cls}: {count}")
    print(f"\nNaechster Schritt: python manual_training/3_train_model.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
