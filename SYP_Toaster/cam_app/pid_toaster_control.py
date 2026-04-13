"""
PID-Regler für automatische Toaster-Steuerung
VOLLAUTOMATISCH - Lernt selbstständig aus Aufnahmen ohne manuelles Labeling!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, TensorDataset
import cv2
import numpy as np
from pathlib import Path
import time
import json
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import pickle


# ==================== KONFIGURATION ====================
class Config:
    """Zentrale Konfiguration - hier alles einstellen"""
    
    # Pfade
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    AUTO_DATA_DIR = BASE_DIR / "auto_training_data"  # Automatisch gesammelte Daten
    MODEL_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "pid_results"
    
    # PID Parameter (anpassbar für dein Toaster-Modell)
    PID_KP = 1.5  # Proportional Gain
    PID_KI = 0.3  # Integral Gain
    PID_KD = 0.5  # Derivative Gain
    
    # Zeitkonstanten (in Sekunden)
    SAMPLE_TIME = 1.0  # Abtastzeit für PID
    MAX_TOAST_TIME = 180.0  # Maximale Toastzeit (3 Minuten)
    MIN_TOAST_TIME = 30.0  # Minimale Toastzeit
    
    # Zielwert (0=roh, 1=leicht, 2=perfekt, 3=dunkel, 4=verbrannt)
    TARGET_STATE = 2  # "perfekt"
    
    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 10  # Weniger Epochs für schnelleres Online-Learning
    
    # Automatisches Lernen
    AUTO_LABEL_INTERVALS = [0, 30, 60, 90, 120]  # Sekunden für automatische Labels
    # 0-30s = roh, 30-60s = leicht, 60-90s = perfekt, 90-120s = dunkel, >120s = verbrannt
    FRAMES_PER_SECOND = 2  # Wie viele Frames pro Sekunde speichern
    MIN_SAMPLES_FOR_TRAINING = 50  # Minimum Samples bevor Training startet
    RETRAIN_INTERVAL = 100  # Alle N neuen Samples neu trainieren
    
    # Kamera
    CAMERA_ID = 0
    CAMERA_IDS = [1, 0, 2]  # Bevorzugte Reihenfolge
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    DISABLE_AUTOFOCUS = True
    IMAGE_SIZE = (224, 224)
    
    # Toast-Klassen
    CLASSES = ['roh', 'leicht', 'perfekt', 'dunkel', 'verbrannt']
    
    def __init__(self):
        self.RESULTS_DIR.mkdir(exist_ok=True)
        self.AUTO_DATA_DIR.mkdir(exist_ok=True)
        self.MODEL_DIR.mkdir(exist_ok=True)


def open_camera(preferred_ids, width=None, height=None, disable_autofocus=False, use_dshow=True):
    """Öffnet automatisch die erste verfügbare Kamera aus der Liste."""
    for cam_id in preferred_ids:
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW) if use_dshow else cv2.VideoCapture(cam_id)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                if width is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height is not None:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                if disable_autofocus:
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                print(f"✅ Kamera {cam_id} gefunden und verbunden.")
                return cap
        cap.release()
    return None


# ==================== PID REGLER ====================
class PIDController:
    """PID-Regler für Toastzeit-Steuerung"""
    
    def __init__(self, kp, ki, kd, sample_time=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.sample_time = sample_time
        
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
        
        self.error_history = []
        self.output_history = []
        self.time_history = []
        
    def compute(self, setpoint, measurement):
        """
        Berechnet PID-Ausgabe
        setpoint: Zielwert (z.B. 2 für "perfekt")
        measurement: Aktueller Messwert (geschätzte Bräunungsstufe)
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt < self.sample_time:
            return None
        
        # Fehler berechnen
        error = setpoint - measurement
        
        # Proportional
        p_term = self.kp * error
        
        # Integral
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative
        d_term = self.kd * (error - self.last_error) / dt if dt > 0 else 0
        
        # PID-Ausgabe
        output = p_term + i_term + d_term
        
        # Speichern für Analyse
        self.error_history.append(error)
        self.output_history.append(output)
        self.time_history.append(current_time)
        
        self.last_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset PID-Zustand"""
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()
        
    def save_history(self, filepath):
        """Speichert PID-Historie"""
        data = {
            'errors': self.error_history,
            'outputs': self.output_history,
            'times': self.time_history,
            'parameters': {'kp': self.kp, 'ki': self.ki, 'kd': self.kd}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def plot_response(self, save_path=None):
        """Visualisiert PID-Antwort"""
        if not self.time_history:
            print("Keine Daten zum Plotten vorhanden")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        times = np.array(self.time_history) - self.time_history[0]
        
        # Fehler über Zeit
        ax1.plot(times, self.error_history, 'b-', linewidth=2)
        ax1.set_ylabel('Fehler (Soll - Ist)', fontsize=12)
        ax1.set_title('PID Regler Antwort', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # PID-Ausgabe über Zeit
        ax2.plot(times, self.output_history, 'g-', linewidth=2)
        ax2.set_xlabel('Zeit (s)', fontsize=12)
        ax2.set_ylabel('PID Ausgabe (Zeitkorrektur)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ==================== NEURONALES NETZ ====================
class ToastClassifier(nn.Module):
    """CNN für Toast-Klassifikation"""
    
    def __init__(self, num_classes=5):
        super().__init__()
        # Verwende vortrainiertes ResNet18
        self.model = models.resnet18(pretrained=True)
        # Ersetze letzte Schicht
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


# ==================== AUTO DATA COLLECTOR ====================
class AutoDataCollector:
    """Sammelt automatisch Daten während Toast-Vorgängen"""
    
    def __init__(self, config):
        self.config = config
        self.data_file = config.AUTO_DATA_DIR / "training_data.pkl"
        self.samples = []  # Liste von (gray_image_array, label, timestamp)
        self.label_time_thresholds = None
        self.load_existing_data()
    
    def load_existing_data(self):
        """Lädt ALLE Trainingsdaten aus dem auto_training_data Verzeichnis"""
        if not self.config.AUTO_DATA_DIR.exists():
            print(f"⚠️ Verzeichnis {self.config.AUTO_DATA_DIR} existiert nicht")
            return
        
        # Finde alle .pkl Dateien im Verzeichnis
        pkl_files = list(self.config.AUTO_DATA_DIR.glob("*.pkl"))
        
        if not pkl_files:
            print(f"ℹ️ Keine Trainingsdaten in {self.config.AUTO_DATA_DIR} gefunden")
            return
        
        print(f"📂 Lade Trainingsdaten aus {len(pkl_files)} Datei(en)...")
        
        for pkl_file in sorted(pkl_files):
            try:
                with open(pkl_file, 'rb') as f:
                    file_samples = pickle.load(f)
                    if isinstance(file_samples, list):
                        self.samples.extend(file_samples)
                        print(f"  ✅ {pkl_file.name}: {len(file_samples)} Samples")
                    else:
                        print(f"  ⚠️ {pkl_file.name}: Unerwartetes Format, überspringe")
            except Exception as e:
                print(f"  ❌ Fehler beim Laden von {pkl_file.name}: {e}")
        
        print(f"✅ Gesamt: {len(self.samples)} Samples geladen")
        # Normalisiere Format und berechne Zeit-Schwellen
        self._update_sample_format()
        self._compute_label_time_thresholds()

    def _compute_label_time_thresholds(self):
        """
        Berechnet dynamische Zeit-Schwellen aus vorhandenen gelabelten Daten.
        Nutzt Median-Zeit je Klasse und setzt Schwellen zwischen den Klassen.
        Fallback auf AUTO_LABEL_INTERVALS wenn zu wenige Daten vorhanden sind.
        """
        num_classes = len(self.config.CLASSES)
        times_by_label = {i: [] for i in range(num_classes)}

        for sample in self.samples:
            if len(sample) != 3:
                continue
            _, label, elapsed = sample
            if isinstance(label, (int, np.integer)) and 0 <= label < num_classes:
                times_by_label[label].append(float(elapsed))

        # Prüfe, ob wir genug Daten haben (mindestens 2 Klassen mit Werten)
        available_labels = [i for i in range(num_classes) if len(times_by_label[i]) > 0]
        if len(available_labels) < 2:
            self.label_time_thresholds = None
            return

        # Median-Zeit pro Klasse
        medians = []
        for i in range(num_classes):
            if times_by_label[i]:
                medians.append(np.median(times_by_label[i]))
            else:
                medians.append(None)

        # Fehlende Medians linear interpolieren zwischen vorhandenen
        last_idx = None
        for i in range(num_classes):
            if medians[i] is not None:
                if last_idx is None:
                    # Backfill nach links
                    for j in range(0, i):
                        medians[j] = medians[i]
                else:
                    # Interpolieren zwischen last_idx und i
                    start = medians[last_idx]
                    end = medians[i]
                    span = i - last_idx
                    for j in range(1, span):
                        medians[last_idx + j] = start + (end - start) * (j / span)
                last_idx = i

        # Falls am Ende None übrig bleibt, forward fill
        if last_idx is not None:
            for j in range(last_idx + 1, num_classes):
                medians[j] = medians[last_idx]

        # Schwellen zwischen Klassen definieren
        thresholds = [0.0]
        for i in range(num_classes - 1):
            thresholds.append((medians[i] + medians[i + 1]) / 2.0)
        thresholds.append(float('inf'))

        self.label_time_thresholds = thresholds
    
    def save_data(self):
        """Speichert Trainingsdaten in separate Datei mit Timestamp"""
        # Speichere mit Timestamp für mehrere Sessions
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = self.config.AUTO_DATA_DIR / f"training_data_{timestamp}.pkl"
        
        with open(save_file, 'wb') as f:
            pickle.dump(self.samples, f)
        print(f"💾 Alle Samples ({len(self.samples)} total) gespeichert in:")
        print(f"   {save_file}")
    
    def auto_label_from_time(self, elapsed_time):
        """
        Automatisches Labeling basierend auf verstrichener Zeit
        Dynamisch aus vorhandenen .pkl berechnet (falls vorhanden).
        Fallback: AUTO_LABEL_INTERVALS.
        """
        if self.label_time_thresholds:
            for i in range(len(self.label_time_thresholds) - 1):
                if self.label_time_thresholds[i] <= elapsed_time < self.label_time_thresholds[i + 1]:
                    return i
            return len(self.label_time_thresholds) - 2

        intervals = self.config.AUTO_LABEL_INTERVALS
        for i in range(len(intervals) - 1):
            if intervals[i] <= elapsed_time < intervals[i + 1]:
                return i
        return len(intervals) - 1  # verbrannt
    
    def _update_sample_format(self):
        """
        Konvertiert alte Samples zu reiner Grayscale-Format (gray, label, time)
        - Alte Farb-Samples: (image, label, time) -> (image_gray, label, time)
        - Gemischte Samples: (image, gray, label, time) -> (gray, label, time)
        """
        updated_samples = []
        for sample in self.samples:
            if len(sample) == 3:
                # Format: (image, label, time) oder (gray, label, time)
                first_elem, label, elapsed = sample
                # Prüfe ob es ein Farbbild oder schon Grayscale ist
                if len(first_elem.shape) == 3:  # Farbbild (RGB/BGR)
                    gray = cv2.cvtColor(first_elem, cv2.COLOR_BGR2GRAY)
                else:  # Schon Grayscale
                    gray = first_elem
                updated_samples.append((gray, label, elapsed))
            elif len(sample) == 4:
                # Format: (image, gray, label, time) - nimm Grayscale
                image, gray, label, elapsed = sample
                updated_samples.append((gray, label, elapsed))
            else:
                # Fallback: nimme wie es ist
                updated_samples.append(sample)
        self.samples = updated_samples
    
    def collect_session(self, duration=120):
        """
        Sammelt automatisch Daten während einer Toast-Session
        duration: Dauer der Aufnahme in Sekunden
        """
        print("\n" + "="*60)
        print("📹 AUTOMATISCHE DATEN-AUFNAHME")
        print("="*60)
        print(f"⏱️  Dauer: {duration}s")
        print(f"📊 Frame-Rate: {self.config.FRAMES_PER_SECOND} FPS")
        print("💡 Labels werden automatisch basierend auf Zeit vergeben")
        print("\nDrücke 'q' zum vorzeitigen Abbruch\n")
        
        cap = open_camera(
            self.config.CAMERA_IDS,
            width=self.config.CAMERA_WIDTH,
            height=self.config.CAMERA_HEIGHT,
            disable_autofocus=self.config.DISABLE_AUTOFOCUS,
            use_dshow=True
        )
        if not cap:
            print("❌ Kamera konnte nicht geöffnet werden!")
            return 0
        
        start_time = time.time()
        frame_interval = 1.0 / self.config.FRAMES_PER_SECOND
        last_capture = 0
        samples_collected = 0
        gui_available = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            elapsed = time.time() - start_time
            
            # Frame erfassen
            if elapsed - last_capture >= frame_interval:
                # Auto-Label basierend auf Zeit
                label = self.auto_label_from_time(elapsed)
                
                # Resize zu Grayscale und speichern
                resized = cv2.resize(frame, self.config.IMAGE_SIZE)
                resized_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                self.samples.append((resized_gray, label, elapsed))
                
                samples_collected += 1
                last_capture = elapsed
            
            # Visualisierung - NUR GRAYSCALE!
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_display = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            current_label = self.auto_label_from_time(elapsed)
            
            cv2.putText(gray_display, f"Zeit: {elapsed:.1f}s / {duration}s",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(gray_display, f"Aktuelles Label: {self.config.CLASSES[current_label]}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(gray_display, f"Samples: {samples_collected}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Fortschrittsbalken
            progress = min(elapsed / duration, 1.0)
            bar_width = int(progress * (gray_display.shape[1] - 20))
            cv2.rectangle(gray_display, (10, gray_display.shape[0] - 30),
                         (10 + bar_width, gray_display.shape[0] - 10),
                         (0, 255, 0), -1)
            
            if gui_available:
                try:
                    cv2.imshow('Automatische Daten-Aufnahme - S/W', gray_display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    print("⚠️  OpenCV GUI nicht verfügbar. Weiter im Headless-Modus.")
                    gui_available = False

            if elapsed >= duration:
                break
        
        cap.release()
        if gui_available:
            cv2.destroyAllWindows()
        
        # Speichern
        self.save_data()
        
        print(f"\n✅ {samples_collected} neue Samples gesammelt")
        print(f"📊 Gesamt: {len(self.samples)} Samples")
        
        return samples_collected


# ==================== AUTO DATASET ====================
class AutoToastDataset(Dataset):
    """Dataset aus automatisch gesammelten Daten"""
    
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Handle both old format (image, label, time) and new format (gray, label, time)
        sample = self.samples[idx]
        if len(sample) == 3:
            # Format: (Grayscale, Label, Time)
            image_gray = sample[0]
        elif len(sample) == 4:
            # Altes Format mit Farbbild: (image_color, image_gray, label, time)
            image_gray = sample[1]
        else:
            raise ValueError(f"Unbekanntes Sample-Format mit {len(sample)} Feldern")
        
        label = sample[-2]  # Zweitletztes Element
        
        # Konvertiere Grayscale zu 3-Kanal RGB für Kompatibilität
        if len(image_gray.shape) == 2:  # Ist Grayscale
            image_rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image_gray
        
        # Convert to PIL Image
        image = Image.fromarray(image_rgb)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ==================== AUTOMATIC TRAINER ====================
class AutomaticTrainer:
    """Vollautomatischer Trainer - lernt aus gesammelten Daten"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Verwende Device: {self.device}")
        
        # Transformationen
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.model = None
        self.history = {'train_loss': [], 'train_acc': []}
        self.data_collector = AutoDataCollector(config)
    
    def prepare_auto_data(self):
        """Bereitet automatisch gesammelte Daten vor"""
        if len(self.data_collector.samples) < self.config.MIN_SAMPLES_FOR_TRAINING:
            raise ValueError(
                f"Zu wenige Samples! Benötigt: {self.config.MIN_SAMPLES_FOR_TRAINING}, "
                f"Vorhanden: {len(self.data_collector.samples)}"
            )
        
        # Update sample format falls nötig (alte + neue Daten kompatibel machen)
        self.data_collector._update_sample_format()
        
        print(f"📁 Verwende {len(self.data_collector.samples)} automatisch gesammelte Samples")
        
        dataset = AutoToastDataset(self.data_collector.samples, transform=self.transform)
        
        # Train/Val Split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        return len(dataset)
    
    def train(self, save_path=None, load_existing=True):
        """Automatisches Training auf gesammelten Daten"""
        print("\n" + "="*60)
        print("🚀 STARTE AUTOMATISCHES TRAINING")
        print("="*60)
        
        # Daten vorbereiten
        total_samples = self.prepare_auto_data()
        
        # Modell erstellen oder laden
        self.model = ToastClassifier(num_classes=len(self.config.CLASSES))
        
        # Lade existierendes Modell falls vorhanden (Online Learning)
        if load_existing and save_path is None:
            save_path = self.config.MODEL_DIR / "pid_toast_model.pt"
        
        if load_existing and save_path and Path(save_path).exists():
            try:
                self.model.load_state_dict(torch.load(save_path, map_location=self.device))
                print("✅ Existierendes Modell geladen - kontinuierliches Lernen")
            except Exception as e:
                print(f"⚠️ Konnte Modell nicht laden: {e} - starte neu")
        
        self.model = self.model.to(self.device)
        
        # Optimizer & Loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        # Training Loop
        best_acc = 0.0
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            print(f"\n📊 Epoch {epoch+1}/{self.config.EPOCHS}")
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = 100. * correct / total
            
            self.history['train_loss'].append(epoch_loss)
            self.history['train_acc'].append(epoch_acc)
            
            print(f"✅ Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                if save_path is None:
                    save_path = self.config.MODEL_DIR / "pid_toast_model.pt"
                torch.save(self.model.state_dict(), save_path)
                print(f"💾 Neues bestes Modell gespeichert! Acc: {best_acc:.2f}%")
        
        print("\n" + "="*60)
        print(f"🎉 TRAINING ABGESCHLOSSEN! Beste Accuracy: {best_acc:.2f}%")
        print("="*60)
        
        return self.model
    
    def plot_training_history(self, save_path=None):
        """Plottet Training-Historie"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        ax1.plot(epochs, self.history['train_loss'], 'b-', linewidth=2, label='Training Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, self.history['train_acc'], 'g-', linewidth=2, label='Training Accuracy')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# ==================== LIVE PID CONTROLLER ====================
class LiveToasterController:
    """Live-Steuerung mit Kamera + PID"""
    
    def __init__(self, config, model_path=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Lade Modell
        self.model = ToastClassifier(num_classes=len(config.CLASSES))
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Modell geladen von: {model_path}")
        else:
            print("⚠️ Kein Modell gefunden - bitte zuerst trainieren!")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # PID Controller
        self.pid = PIDController(
            kp=config.PID_KP,
            ki=config.PID_KI,
            kd=config.PID_KD,
            sample_time=config.SAMPLE_TIME
        )
        
        self.current_toast_time = 60.0  # Startwert
        self.session_data = []
    
    def predict_toast_state(self, image):
        """Klassifiziert Toast-Zustand"""
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
        
        return predicted.item(), confidence.item(), probs[0].cpu().numpy()
    
    def get_continuous_state(self, probs):
        """
        Berechnet kontinuierlichen Zustand aus Wahrscheinlichkeiten
        Gewichteter Durchschnitt der Klassen für smoothe PID-Regelung
        """
        state_values = np.arange(len(self.config.CLASSES))
        continuous_state = np.sum(state_values * probs)
        return continuous_state
    
    def run_live_control(self, duration=60):
        """
        Live-Kontrolle mit Kamera
        duration: Dauer des Tests in Sekunden
        """
        print("\n" + "="*60)
        print("🎥 STARTE LIVE PID-KONTROLLE")
        print("="*60)
        print(f"Zielwert: {self.config.CLASSES[self.config.TARGET_STATE]}")
        print(f"PID-Parameter: Kp={self.config.PID_KP}, Ki={self.config.PID_KI}, Kd={self.config.PID_KD}")
        print("Drücke 'q' zum Beenden\n")
        
        cap = open_camera(
            self.config.CAMERA_IDS,
            width=self.config.CAMERA_WIDTH,
            height=self.config.CAMERA_HEIGHT,
            disable_autofocus=self.config.DISABLE_AUTOFOCUS,
            use_dshow=True
        )
        
        if not cap:
            print("❌ Kamera konnte nicht geöffnet werden!")
            return
        
        self.pid.reset()
        start_time = time.time()
        gui_available = True
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Vorhersage
            predicted_class, confidence, probs = self.predict_toast_state(frame)
            continuous_state = self.get_continuous_state(probs)
            
            # PID-Berechnung
            pid_output = self.pid.compute(self.config.TARGET_STATE, continuous_state)
            
            if pid_output is not None:
                # Update Toast-Zeit basierend auf PID
                self.current_toast_time += pid_output
                self.current_toast_time = np.clip(
                    self.current_toast_time,
                    self.config.MIN_TOAST_TIME,
                    self.config.MAX_TOAST_TIME
                )
                
                # Speichere Session-Daten
                self.session_data.append({
                    'time': time.time() - start_time,
                    'state': continuous_state,
                    'class': predicted_class,
                    'confidence': confidence,
                    'toast_time': self.current_toast_time,
                    'pid_output': pid_output
                })
            
            # Visualisierung - NUR GRAYSCALE!
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            
            # Info-Text
            y_pos = 30
            cv2.putText(display_frame, f"Zustand: {self.config.CLASSES[predicted_class]} ({confidence*100:.1f}%)",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
            cv2.putText(display_frame, f"Kontinuierlich: {continuous_state:.2f}",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_pos += 30
            cv2.putText(display_frame, f"Ziel: {self.config.CLASSES[self.config.TARGET_STATE]} ({self.config.TARGET_STATE})",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 30
            cv2.putText(display_frame, f"Toast-Zeit: {self.current_toast_time:.1f}s",
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            if pid_output is not None:
                y_pos += 30
                cv2.putText(display_frame, f"PID Korrektur: {pid_output:+.2f}s",
                           (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
            
            # Wahrscheinlichkeiten als Balken
            bar_height = 20
            bar_y_start = display_frame.shape[0] - (len(self.config.CLASSES) * (bar_height + 5)) - 10
            for i, (class_name, prob) in enumerate(zip(self.config.CLASSES, probs)):
                y = bar_y_start + i * (bar_height + 5)
                bar_width = int(prob * 300)
                color = (0, 255, 0) if i == self.config.TARGET_STATE else (200, 200, 200)
                cv2.rectangle(display_frame, (10, y), (10 + bar_width, y + bar_height), color, -1)
                cv2.putText(display_frame, f"{class_name}: {prob*100:.1f}%",
                           (320, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if gui_available:
                try:
                    cv2.imshow('PID Toaster Control - S/W', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    print("⚠️  OpenCV GUI nicht verfügbar. Weiter im Headless-Modus.")
                    gui_available = False
            
            if time.time() - start_time > duration:
                break
        
        cap.release()
        if gui_available:
            cv2.destroyAllWindows()
        
        print("\n✅ Live-Kontrolle beendet")
        return self.session_data
    
    def save_session(self, filepath=None):
        """Speichert Session-Daten"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.config.RESULTS_DIR / f"session_{timestamp}.json"
        
        data = {
            'config': {
                'pid_params': {'kp': self.config.PID_KP, 'ki': self.config.PID_KI, 'kd': self.config.PID_KD},
                'target_state': self.config.TARGET_STATE,
                'classes': self.config.CLASSES
            },
            'session_data': self.session_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Session gespeichert: {filepath}")


# ==================== HAUPT-INTERFACE ====================
class ToasterPIDSystem:
    """Vollautomatisches System - Lernt selbstständig!"""
    
    def __init__(self):
        self.config = Config()
        self.data_collector = AutoDataCollector(self.config)
        print("🍞 Vollautomatisches Toaster PID System initialisiert")
        print(f"📁 Arbeitsverzeichnis: {self.config.BASE_DIR}")
        print(f"📊 Gespeicherte Samples: {len(self.data_collector.samples)}")
    
    def collect_training_data(self, duration=120):
        """Sammelt automatisch Trainingsdaten"""
        print("\n🎯 Starte automatische Daten-Sammlung...")
        print("💡 Das System nimmt automatisch auf und vergibt Labels basierend auf Zeit!")
        
        new_samples = self.data_collector.collect_session(duration=duration)
        
        # Automatisch neu trainieren wenn genug neue Daten
        if new_samples > 0 and len(self.data_collector.samples) >= self.config.MIN_SAMPLES_FOR_TRAINING:
            print("\n🤖 Genug Daten vorhanden - automatisches Re-Training?")
            choice = input("Jetzt trainieren? (j/n): ").strip().lower()
            if choice == 'j':
                self.auto_train()
    
    def auto_train(self):
        """Automatisches Training auf gesammelten Daten"""
        print("\n🎯 Starte automatisches Training auf gesammelten Daten...")
        
        trainer = AutomaticTrainer(self.config)
        
        if len(trainer.data_collector.samples) < self.config.MIN_SAMPLES_FOR_TRAINING:
            print(f"❌ Zu wenige Daten! Benötigt: {self.config.MIN_SAMPLES_FOR_TRAINING}")
            print(f"   Vorhanden: {len(trainer.data_collector.samples)}")
            print("   → Bitte zuerst Daten sammeln (Option 1)")
            return None
        
        model = trainer.train()
        
        # Plotte Trainings-Historie
        plot_path = self.config.RESULTS_DIR / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        trainer.plot_training_history(save_path=plot_path)
        print(f"📊 Training-Plot gespeichert: {plot_path}")
        
        return model
    
    def run_live_pid(self, model_path=None, duration=60):
        """Live PID-Kontrolle"""
        if model_path is None:
            model_path = self.config.MODEL_DIR / "pid_toast_model.pt"
        
        if not Path(model_path).exists():
            print("❌ Kein trainiertes Modell gefunden!")
            print("   → Bitte zuerst Daten sammeln und trainieren")
            return None
        
        controller = LiveToasterController(self.config, model_path)
        session_data = controller.run_live_control(duration=duration)
        
        # Speichere Session
        controller.save_session()
        
        # Plotte PID-Antwort
        plot_path = self.config.RESULTS_DIR / f"pid_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        controller.pid.plot_response(save_path=plot_path)
        print(f"📊 PID-Plot gespeichert: {plot_path}")
        
        return session_data
    
    def tune_pid(self):
        """Interaktives PID-Tuning"""
        print("\n🎛️ PID-Tuning Modus")
        print("Aktuelle Parameter:")
        print(f"  Kp = {self.config.PID_KP}")
        print(f"  Ki = {self.config.PID_KI}")
        print(f"  Kd = {self.config.PID_KD}")
        print("\nTipps:")
        print("  - Kp erhöhen → schnellere Reaktion, aber mehr Überschwingen")
        print("  - Ki erhöhen → eliminiert stationären Fehler")
        print("  - Kd erhöhen → dämpft Überschwingen")
        
        try:
            kp = float(input("\nNeuer Kp-Wert (Enter für aktuell): ") or self.config.PID_KP)
            ki = float(input("Neuer Ki-Wert (Enter für aktuell): ") or self.config.PID_KI)
            kd = float(input("Neuer Kd-Wert (Enter für aktuell): ") or self.config.PID_KD)
            
            self.config.PID_KP = kp
            self.config.PID_KI = ki
            self.config.PID_KD = kd
            
            print(f"\n✅ PID-Parameter aktualisiert: Kp={kp}, Ki={ki}, Kd={kd}")
        except ValueError:
            print("❌ Ungültige Eingabe!")
    
    def auto_workflow(self):
        """Komplett automatischer Workflow - Ein Knopf für alles!"""
        print("\n" + "="*60)
        print("🤖 VOLLAUTOMATISCHER WORKFLOW")
        print("="*60)
        print("Das System wird:")
        print("1. 📹 Automatisch Daten aufnehmen (120s)")
        print("2. 🧠 Automatisch trainieren")
        print("3. 🎥 Live PID-Test durchführen (60s)")
        print("="*60)
        
        input("\n▶️  Drücke Enter zum Starten...")
        
        # Schritt 1: Daten sammeln
        print("\n[1/3] Datensammlung...")
        self.data_collector.collect_session(duration=120)
        
        # Schritt 2: Trainieren
        print("\n[2/3] Training...")
        if len(self.data_collector.samples) >= self.config.MIN_SAMPLES_FOR_TRAINING:
            self.auto_train()
        else:
            print(f"⚠️ Nicht genug Daten ({len(self.data_collector.samples)}/{self.config.MIN_SAMPLES_FOR_TRAINING})")
            print("   Überspringe Training - verwende existierendes Modell falls vorhanden")
        
        # Schritt 3: Live-Test
        print("\n[3/3] Live-Test...")
        input("▶️  Drücke Enter für Live-Test...")
        self.run_live_pid(duration=60)
        
        print("\n" + "="*60)
        print("✅ VOLLAUTOMATISCHER WORKFLOW ABGESCHLOSSEN!")
        print("="*60)
    
    def show_stats(self):
        """Zeigt Statistiken über gesammelte Daten"""
        print("\n" + "="*60)
        print("📊 DATEN-STATISTIKEN")
        print("="*60)
        print(f"Gesamt-Samples: {len(self.data_collector.samples)}")
        
        if len(self.data_collector.samples) > 0:
            # Zähle Labels
            label_counts = {}
            for _, label, _ in self.data_collector.samples:
                label_name = self.config.CLASSES[label]
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
            
            print("\nVerteilung:")
            for label_name in self.config.CLASSES:
                count = label_counts.get(label_name, 0)
                percentage = (count / len(self.data_collector.samples)) * 100
                bar = "█" * int(percentage / 2)
                print(f"  {label_name:10s}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        print(f"\nStatus: {'✅ Bereit für Training' if len(self.data_collector.samples) >= self.config.MIN_SAMPLES_FOR_TRAINING else '⚠️ Mehr Daten benötigt'}")
        print(f"Benötigt: {self.config.MIN_SAMPLES_FOR_TRAINING} Samples")
        print("="*60)
    
    def main_menu(self):
        """Hauptmenü - Komplett automatisiert!"""
        while True:
            print("\n" + "="*60)
            print("🍞 VOLLAUTOMATISCHES TOASTER PID SYSTEM")
            print("="*60)
            print("1. 🤖 VOLLAUTOMATIK (Alles auf einmal!)")
            print("2. 📹 Daten aufnehmen (automatisch gelabelt)")
            print("3. 🧠 Training starten")
            print("4. 🎥 Live PID-Test")
            print("5. 📊 Statistiken anzeigen")
            print("6. 🎛️  PID-Parameter anpassen")
            print("7. ❌ Beenden")
            print("="*60)
            print(f"💾 Gespeicherte Samples: {len(self.data_collector.samples)}")
            print("="*60)
            
            choice = input("\nWähle Option (1-7): ").strip()
            
            if choice == '1':
                self.auto_workflow()
            
            elif choice == '2':
                duration = input("Aufnahme-Dauer in Sekunden (Enter für 120s): ").strip()
                duration = int(duration) if duration else 120
                self.collect_training_data(duration=duration)
            
            elif choice == '3':
                self.auto_train()
            
            elif choice == '4':
                duration = input("Test-Dauer in Sekunden (Enter für 60s): ").strip()
                duration = int(duration) if duration else 60
                self.run_live_pid(duration=duration)
            
            elif choice == '5':
                self.show_stats()
            
            elif choice == '6':
                self.tune_pid()
            
            elif choice == '7':
                self.tune_pid()
            
            elif choice == '4':
                print("\n🔄 Vollständiger Workflow...")
                self.auto_train()
                input("\n⏸️  Training abgeschlossen. Enter drücken für Live-Test...")
                self.run_live_pid(duration=30)
            
            elif choice == '5':
                print("\n👋 Auf Wiedersehen!")
                break
            
            else:
                print("❌ Ungültige Auswahl!")


# ==================== MAIN ====================
if __name__ == "__main__":
    system = ToasterPIDSystem()
    system.main_menu()
