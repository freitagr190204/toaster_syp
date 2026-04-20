"""
Skript 3: Toast-Klassifikation trainieren (ResNet-18 + ImageNet-Gewichte)
=======================================================================
Optimiert fuer echte Kamera / echte Beleuchtung:
  - Pretrained ResNet-18 Backbone (robuste Low-Level-Features)
  - ImageNet-Normalisierung (muss mit 4_live_prediction.py uebereinstimmen)
  - Starke Augmentation nur im Training; Validation ohne Zufalls-Augmentation
  - Stratified Train/Val Split (jede Klasse anteilig in beiden Sets)
  - Gewichteter Loss + Balanced Sampling + Label Smoothing

Workflow: recording.py -> 1_crop_images.py -> 2_label_images.py -> 3_train_model.py -> 4_live_prediction.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import cv2
import numpy as np
from collections import Counter
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
import toast_net

# ================= PFADE =================
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATASET_PATH = DATA_DIR / "labeled_dataset"
MODEL_SAVE_PATH = MODELS_DIR / "toast_model.pt"
TRAINING_PLOT_PATH = MODELS_DIR / "training_history.png"

# ================= TRAINING CONFIG =================
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.00035
IMG_SIZE = 224
EARLY_STOP_PATIENCE = 12
VAL_FRACTION = 0.2
LABEL_SMOOTHING = 0.08
SPLIT_SEED = 42

CLASSES = ["roh", "leicht", "perfekt", "dunkel", "verbrannt"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= DATASET =================
class ToastDataset(Dataset):
    def __init__(self, samples: list, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        return [label for _, label in self.samples]

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        if self.transform:
            img = self.transform(img)

        return img, label


def discover_samples(root_dir: Path) -> list:
    samples = []
    class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    for class_name in CLASSES:
        class_dir = root_dir / class_name
        if class_dir.exists():
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    samples.append((img_path, class_to_idx[class_name]))
    return samples


def stratified_train_val_indices(labels: list, val_fraction: float, seed: int) -> tuple:
    """Indices into labels list: train_idx, val_idx per class stratified."""
    labels_arr = np.array(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in range(len(CLASSES)):
        idx = np.where(labels_arr == c)[0]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n_val = int(round(len(idx) * val_fraction))
        if len(idx) == 1:
            n_val = 0
        else:
            n_val = min(max(n_val, 0), len(idx) - 1)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def split_resnet_params(model: nn.Module):
    backbone, head = [], []
    for name, p in model.named_parameters():
        (head if name.startswith("fc") else backbone).append(p)
    return backbone, head


# ================= TRAINING =================
def compute_class_weights(labels):
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(len(CLASSES)):
        c = counts.get(i, 1)
        weights.append(total / (len(CLASSES) * c))
    return torch.FloatTensor(weights)


def create_balanced_sampler(labels):
    counts = Counter(labels)
    weight_per_sample = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weight_per_sample, num_samples=len(labels), replacement=True)


def train_model():
    print("=" * 55)
    print("    TOAST-RESNET TRAINING")
    print("=" * 55)
    print(f"Device: {device}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model:   {MODEL_SAVE_PATH}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        print(f"Dataset nicht gefunden: {DATASET_PATH}")
        print("   Bitte zuerst 2_label_images.py ausfuehren!")
        return

    all_samples = discover_samples(DATASET_PATH)
    all_labels = [lab for _, lab in all_samples]
    print(f"Dataset geladen: {len(all_samples)} Bilder")
    for cls in CLASSES:
        count = sum(1 for _, l in all_samples if l == CLASSES.index(cls))
        print(f"   {cls}: {count}")

    if len(all_samples) < 10:
        print(f"Nur {len(all_samples)} Bilder - mehr Daten empfohlen!")
        return

    train_idx, val_idx = stratified_train_val_indices(all_labels, VAL_FRACTION, SPLIT_SEED)
    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]

    norm = transforms.Normalize(toast_net.IMAGENET_MEAN, toast_net.IMAGENET_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.12),
            transforms.RandomRotation(18),
            transforms.ColorJitter(brightness=0.55, contrast=0.55, saturation=0.4, hue=0.07),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.86, 1.14)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.35),
            transforms.RandomGrayscale(p=0.18),
            norm,
            transforms.RandomErasing(p=0.22, scale=(0.02, 0.14), ratio=(0.4, 2.8)),
        ]
    )
    val_transform = transforms.Compose([norm])

    train_dataset = ToastDataset(train_samples, transform=train_transform)
    val_dataset = ToastDataset(val_samples, transform=val_transform)

    train_labels = train_dataset.get_labels()
    sampler = create_balanced_sampler(train_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"\nTraining: {len(train_dataset)} | Validation: {len(val_dataset)} (stratified)")
    print(f"Klassen-Verteilung (Train): {Counter(train_labels)}")

    model = toast_net.build_toast_classifier(len(CLASSES), pretrained_backbone=True).to(device)
    backbone_params, head_params = split_resnet_params(model)
    class_weights = compute_class_weights(train_labels).to(device)
    print(f"Klassen-Gewichte: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    train_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    val_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        [
            {"params": backbone_params, "lr": LEARNING_RATE * 0.22},
            {"params": head_params, "lr": LEARNING_RATE},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    no_improve = 0

    print(f"\nStarte Training fuer max. {EPOCHS} Epochen (Early Stop nach {EARLY_STOP_PATIENCE})...\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = train_criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += val_criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= max(len(val_loader), 1)
        val_acc = 100 * correct / total if total > 0 else 0

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "accuracy": best_acc,
                    "arch": "resnet18",
                },
                MODEL_SAVE_PATH,
            )
            marker = " << BEST"
        else:
            no_improve += 1

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"TLoss: {train_loss:.4f} | VLoss: {val_loss:.4f} | "
            f"VAcc: {val_acc:.1f}% | LR: {lr:.6f}{marker}"
        )

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly Stop nach {epoch+1} Epochen (keine Verbesserung seit {EARLY_STOP_PATIENCE})")
            break

    print("\n" + "=" * 55)
    print(f"Training abgeschlossen! Beste Accuracy: {best_acc:.1f}%")
    print(f"Modell gespeichert: {MODEL_SAVE_PATH}")
    print("=" * 55)

    plot_history(history)
    return model


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["val_acc"], label="Val Accuracy", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(str(TRAINING_PLOT_PATH), dpi=150)
    print(f"Plot gespeichert: {TRAINING_PLOT_PATH}")


def predict_image(image_path):
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    if checkpoint.get("arch") != "resnet18":
        raise RuntimeError("Altes Modell-Format: bitte neu trainieren mit diesem Skript (ResNet18).")
    model = toast_net.build_toast_classifier(len(CLASSES), pretrained_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    img = toast_net.normalize_imagenet(img)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100

    return CLASSES[pred_idx], confidence


if __name__ == "__main__":
    train_model()
