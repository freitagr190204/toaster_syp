"""
Skript 1: Toast-Bilder automatisch zuschneiden
===============================================
Schneidet die Bilder auf den relevanten Toast-Bereich zu.

Workflow: recording.py -> 1_crop_images.py -> 2_label_images.py -> 3_train_model.py -> 4_live_prediction.py
"""

import cv2
import os
import sys
import numpy as np
from pathlib import Path

# ================= PFADE =================
# Basis-Ordner (cam_app)
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Input/Output Ordner
INPUT_FOLDER = DATA_DIR / "raw_images"        # Rohbilder von recording.py
OUTPUT_FOLDER = DATA_DIR / "cropped_images"   # Zugeschnittene Bilder

# ================= EINSTELLUNGEN =================
# Crop-Bereich (anpassen basierend auf eurer Kamera-Position!)
# Diese Werte müsst ihr eventuell anpassen
CROP_X = 250      # Start X (links)
CROP_Y = 150      # Start Y (oben)  
CROP_WIDTH = 750  # Breite des Ausschnitts
CROP_HEIGHT = 550 # Höhe des Ausschnitts

# Zielgröße für ML (quadratisch ist besser für CNNs)
TARGET_SIZE = (224, 224)

# ================= HAUPTPROGRAMM =================
def crop_and_resize(image_path, output_path):
    """Schneidet Bild zu und skaliert es auf Zielgröße"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ Konnte nicht laden: {image_path}")
        return False
    
    h, w = img.shape[:2]
    
    # Sicherstellen, dass Crop-Bereich im Bild liegt
    x1 = max(0, CROP_X)
    y1 = max(0, CROP_Y)
    x2 = min(w, CROP_X + CROP_WIDTH)
    y2 = min(h, CROP_Y + CROP_HEIGHT)
    
    # Zuschneiden
    cropped = img[y1:y2, x1:x2]
    
    # Auf Zielgröße skalieren
    resized = cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    
    # Speichern
    cv2.imwrite(str(output_path), resized)
    return True

def preview_crop_area():
    """Zeigt ein Beispielbild mit dem Crop-Bereich an"""
    images = list(Path(INPUT_FOLDER).glob("*.png")) + list(Path(INPUT_FOLDER).glob("*.jpg"))
    if not images:
        print("❌ Keine Bilder im Input-Ordner gefunden!")
        return
    
    img = cv2.imread(str(images[0]))
    h, w = img.shape[:2]
    
    # Crop-Bereich einzeichnen
    cv2.rectangle(img, (CROP_X, CROP_Y), 
                  (CROP_X + CROP_WIDTH, CROP_Y + CROP_HEIGHT), 
                  (0, 255, 0), 3)
    cv2.putText(img, "Crop-Bereich (Gruene Box anpassen!)", (CROP_X, CROP_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Vorschau anzeigen
    cv2.imshow("Crop-Vorschau - Druecke Q zum Schliessen", img)
    print("\n📸 Vorschau wird angezeigt...")
    print("   Falls der gruene Bereich nicht den Toast zeigt,")
    print("   passe CROP_X, CROP_Y, CROP_WIDTH, CROP_HEIGHT an!")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print("="*50)
    print("🍞 TOAST-BILDER ZUSCHNEIDEN")
    print("="*50)
    print(f"\n📁 Input:  {INPUT_FOLDER}")
    print(f"📁 Output: {OUTPUT_FOLDER}")
    
    # Ordner erstellen falls nicht vorhanden
    INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Erst Vorschau zeigen
    print("\n1. Zeige Crop-Vorschau...")
    preview_crop_area()
    
    # User fragen ob fortfahren
    response = input("\nMit diesen Einstellungen fortfahren? (j/n): ")
    if response.lower() != 'j':
        print("Abgebrochen. Bitte CROP_X, CROP_Y etc. im Skript anpassen.")
        return
    
    # Alle Bilder verarbeiten
    images = list(Path(INPUT_FOLDER).glob("*.png")) + list(Path(INPUT_FOLDER).glob("*.jpg"))
    
    if not images:
        print(f"❌ Keine Bilder in '{INPUT_FOLDER}' gefunden!")
        return
    
    print(f"\n2. Verarbeite {len(images)} Bilder...")
    
    success = 0
    for img_path in images:
        output_path = Path(OUTPUT_FOLDER) / f"cropped_{img_path.name}"
        if crop_and_resize(img_path, output_path):
            success += 1
            print(f"   ✅ {img_path.name}")
    
    print(f"\n{'='*50}")
    print(f"✅ Fertig! {success}/{len(images)} Bilder zugeschnitten")
    print(f"   Ausgabe in: {OUTPUT_FOLDER}/")
    print("="*50)

if __name__ == "__main__":
    main()
