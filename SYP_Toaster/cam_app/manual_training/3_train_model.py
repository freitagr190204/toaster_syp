"""
Skript 3: Toast-Klassifikations-CNN trainieren
===============================================
Trainiert ein einfaches CNN zur Toast-Bräunungserkennung.

Workflow: recording.py -> 1_crop_images.py -> 2_label_images.py -> 3_train_model.py -> 4_live_prediction.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ================= PFADE =================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Input/Output
DATASET_PATH = DATA_DIR / "labeled_dataset"   # Von Skript 2
MODEL_SAVE_PATH = MODELS_DIR / "toast_model.pt"
TRAINING_PLOT_PATH = MODELS_DIR / "training_history.png"

# Training
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
IMG_SIZE = 224

# Klassen
CLASSES = ['roh', 'leicht', 'perfekt', 'dunkel', 'verbrannt']

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= DATASET =================
class ToastDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
        
        # Alle Bilder laden
        for class_name in CLASSES:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"📁 Dataset geladen: {len(self.samples)} Bilder")
        for cls in CLASSES:
            count = len(list((self.root_dir / cls).glob("*"))) if (self.root_dir / cls).exists() else 0
            print(f"   {cls}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Bild laden (OpenCV -> RGB)
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Zu Tensor
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        
        if self.transform:
            img = self.transform(img)
        
        return img, label

# ================= CNN MODELL =================
class ToastCNN(nn.Module):
    """Einfaches CNN für Toast-Klassifikation"""
    def __init__(self, num_classes=5):
        super(ToastCNN, self).__init__()
        
        # Convolutional Layers
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ================= TRAINING =================
def train_model():
    print("="*50)
    print("🧠 TOAST-CNN TRAINING")
    print("="*50)
    print(f"Device: {device}")
    print(f"📁 Dataset: {DATASET_PATH}")
    print(f"📁 Model:   {MODEL_SAVE_PATH}")
    
    # Ordner erstellen
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dataset prüfen
    if not DATASET_PATH.exists():
        print(f"❌ Dataset nicht gefunden: {DATASET_PATH}")
        print("   Bitte zuerst Skript 2 (2_label_images.py) ausführen!")
        return
    
    # Data Augmentation für Training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])
    
    # Dataset laden
    dataset = ToastDataset(DATASET_PATH, transform=train_transform)
    
    if len(dataset) < 10:
        print(f"⚠️  Nur {len(dataset)} Bilder - mehr Daten empfohlen!")
    
    # Train/Val Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n📊 Training: {train_size} | Validation: {val_size}")
    
    # Modell erstellen
    model = ToastCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training Loop
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    
    print(f"\n🚀 Starte Training für {EPOCHS} Epochen...\n")
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        val_acc = 100 * correct / total if total > 0 else 0
        
        scheduler.step(val_loss)
        
        # History speichern
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Bestes Modell speichern
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': CLASSES,
                'accuracy': best_acc
            }, MODEL_SAVE_PATH)
            marker = " ⭐ BEST"
        else:
            marker = ""
        
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.1f}%{marker}")
    
    print("\n" + "="*50)
    print(f"✅ Training abgeschlossen!")
    print(f"   Beste Accuracy: {best_acc:.1f}%")
    print(f"   Modell gespeichert: {MODEL_SAVE_PATH}")
    print("="*50)
    
    # Plot erstellen
    plot_history(history)
    
    return model

def plot_history(history):
    """Erstellt Training-Plots"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss Plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy Plot
    ax2.plot(history['val_acc'], label='Val Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(str(TRAINING_PLOT_PATH), dpi=150)
    plt.show()
    print(f"📈 Plot gespeichert: {TRAINING_PLOT_PATH}")

# ================= INFERENZ =================
def predict_image(image_path):
    """Klassifiziert ein einzelnes Bild"""
    # Modell laden
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
    model = ToastCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Bild laden
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Vorhersage
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100
    
    return CLASSES[pred_idx], confidence

if __name__ == "__main__":
    train_model()
