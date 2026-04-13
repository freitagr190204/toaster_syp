"""
Skript 2: Toast-Bilder manuell labeln
======================================
Sortiert Bilder in Ordner nach Bräunungsstufe.
Einfaches GUI zum schnellen Labeln mit Tastatur.

Workflow: recording.py -> 1_crop_images.py -> 2_label_images.py -> 3_train_model.py -> 4_live_prediction.py
"""

import cv2
import os
import shutil
from pathlib import Path

# ================= PFADE =================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Input/Output Ordner
# Eingabe jetzt direkt aus den Rohbildern von recording.py
INPUT_FOLDER = DATA_DIR / "raw_images"
OUTPUT_BASE = DATA_DIR / "labeled_dataset"   # Gelabelte Daten für Training

# ================= EINSTELLUNGEN =================
LABELS = {
    '1': 'roh',
    '2': 'leicht', 
    '3': 'perfekt',
    '4': 'dunkel',
    '5': 'verbrannt',
    's': 'skip'  # Überspringen (schlechtes Bild)
}

# ================= HAUPTPROGRAMM =================
def create_label_folders():
    """Erstellt Ordnerstruktur für Labels"""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    for label in LABELS.values():
        if label != 'skip':
            (OUTPUT_BASE / label).mkdir(exist_ok=True)
    print(f"📁 Ordner erstellt in: {OUTPUT_BASE}")

def get_already_labeled():
    """Gibt Set von bereits gelabelten Dateinamen zurück"""
    labeled = set()
    if OUTPUT_BASE.exists():
        for label_folder in OUTPUT_BASE.iterdir():
            if label_folder.is_dir():
                for img in label_folder.glob("*"):
                    labeled.add(img.name)
    return labeled

def label_images():
    """Hauptfunktion zum Labeln"""
    print("="*50)
    print("🏷️  TOAST-BILDER LABELN")
    print("="*50)
    print("\nTasten:")
    for key, label in LABELS.items():
        emoji = {'roh': '🍞', 'leicht': '🥪', 'perfekt': '✅', 
                 'dunkel': '🔥', 'verbrannt': '💀', 'skip': '⏭️'}.get(label, '')
        print(f"   [{key}] = {label} {emoji}")
    print("   [ESC] = Beenden")
    print("="*50)
    
    # Ordner erstellen
    create_label_folders()
    
    # Bilder laden
    images = sorted(list(Path(INPUT_FOLDER).glob("*.png")) + 
                   list(Path(INPUT_FOLDER).glob("*.jpg")))
    
    if not images:
        print(f"❌ Keine Bilder in '{INPUT_FOLDER}' gefunden!")
        print("   Bitte zuerst mit recording.py Bilder in data/raw_images aufnehmen.")
        return
    
    # Bereits gelabelte überspringen
    already_labeled = get_already_labeled()
    images = [img for img in images if img.name not in already_labeled]
    
    if not images:
        print("✅ Alle Bilder bereits gelabelt!")
        return
    
    print(f"\n📸 {len(images)} Bilder zu labeln...\n")
    
    labeled_count = {label: 0 for label in LABELS.values()}
    
    for i, img_path in enumerate(images):
        # Bild laden und anzeigen
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Bild größer anzeigen
        display = cv2.resize(img, (448, 448), interpolation=cv2.INTER_LINEAR)
        
        # Info-Text hinzufügen
        cv2.putText(display, f"Bild {i+1}/{len(images)}: {img_path.name}", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "1=roh 2=leicht 3=perfekt 4=dunkel 5=verbrannt s=skip", 
                    (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.imshow("Toast Labeling - Druecke 1-5 oder S", display)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == 27:  # ESC
                print("\n⏹️  Labeln beendet.")
                cv2.destroyAllWindows()
                print_summary(labeled_count)
                return
            
            key_char = chr(key).lower()
            if key_char in LABELS:
                label = LABELS[key_char]
                
                if label != 'skip':
                    # Bild in entsprechenden Ordner kopieren
                    dest = Path(OUTPUT_BASE) / label / img_path.name
                    shutil.copy(img_path, dest)
                    labeled_count[label] += 1
                    print(f"   [{label.upper()}] {img_path.name}")
                else:
                    labeled_count['skip'] += 1
                    print(f"   [SKIP] {img_path.name}")
                break
    
    cv2.destroyAllWindows()
    print("\n✅ Alle Bilder gelabelt!")
    print_summary(labeled_count)

def print_summary(counts):
    """Zeigt Zusammenfassung"""
    print("\n" + "="*50)
    print("📊 ZUSAMMENFASSUNG")
    print("="*50)
    total = 0
    for label, count in counts.items():
        if count > 0:
            emoji = {'roh': '🍞', 'leicht': '🥪', 'perfekt': '✅', 
                     'dunkel': '🔥', 'verbrannt': '💀', 'skip': '⏭️'}.get(label, '')
            print(f"   {emoji} {label}: {count}")
            if label != 'skip':
                total += count
    print(f"\n   Total gelabelt: {total}")
    print(f"   Gespeichert in: {OUTPUT_BASE}/")
    print("="*50)

if __name__ == "__main__":
    label_images()
