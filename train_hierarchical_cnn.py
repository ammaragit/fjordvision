import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
IMG_SIZE = 128

DATA_ROOT = "data/cnn"
SAVE_PATH = "cnn_weights/best_hierarchical_cnn.pt"
METRICS_DIR = "metrics"

LEVELS = ["binary", "class", "genus", "species"]

os.makedirs("cnn_weights", exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# =========================
# DATASET
# =========================
class HierarchicalDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.class_maps = {}
        self.samples = []

        for level in LEVELS:
            classes = sorted(os.listdir(os.path.join(root, level)))
            self.class_maps[level] = {c: i for i, c in enumerate(classes)}

        # assume all levels share same image names
        sample_set = set()

        for cls in self.class_maps["binary"].keys():
            cls_path = os.path.join(root, "binary", cls)
            for img in os.listdir(cls_path):
                sample_set.add(img)

        self.samples = sorted(list(sample_set))


        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name = self.samples[idx]
        targets = {}
        img_path = None

        for level in LEVELS:
            for cls, cls_idx in self.class_maps[level].items():
                p = os.path.join(self.root, level, cls, img_name)
                if os.path.exists(p):
                    img_path = p
                    targets[level] = cls_idx

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        targets = {k: torch.tensor(v, dtype=torch.long) for k, v in targets.items()}
        return image, targets

# =========================
# MODEL (TOP-DOWN HIERARCHY)
# =========================
class HierarchicalCNN(nn.Module):
    def __init__(self, n_binary, n_class, n_genus, n_species):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(128, 256)

        self.binary_head  = nn.Linear(256, n_binary)
        self.class_head   = nn.Linear(256 + n_binary, n_class)
        self.genus_head   = nn.Linear(256 + n_class, n_genus)
        self.species_head = nn.Linear(256 + n_genus, n_species)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        x = F.relu(self.fc(x))

        outputs = {}

        outputs["binary"] = self.binary_head(x)
        b = F.softmax(outputs["binary"], dim=1)

        outputs["class"] = self.class_head(torch.cat([x, b], dim=1))
        c = F.softmax(outputs["class"], dim=1)

        outputs["genus"] = self.genus_head(torch.cat([x, c], dim=1))
        g = F.softmax(outputs["genus"], dim=1)

        outputs["species"] = self.species_head(torch.cat([x, g], dim=1))

        return outputs

# =========================
# HIERARCHICAL LOSS
# =========================
def hierarchical_loss(outputs, targets, weights=None):
    if weights is None:
        weights = {l: 1.0 for l in LEVELS}

    loss = 0.0
    for level in LEVELS:
        loss += weights[level] * F.cross_entropy(
            outputs[level],
            targets[level]
        )
    return loss

# =========================
# EVALUATION
# =========================
def evaluate(model, loader, dataset, split="val"):
    model.eval()

    preds = {l: [] for l in LEVELS}
    trues = {l: [] for l in LEVELS}

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            targets = {k: v.to(DEVICE) for k, v in targets.items()}

            outputs = model(images)

            for level in LEVELS:
                p = torch.argmax(outputs[level], dim=1)
                preds[level].extend(p.cpu().numpy())
                trues[level].extend(targets[level].cpu().numpy())

    accuracies = {}

    for level in LEVELS:
        acc = accuracy_score(trues[level], preds[level]) * 100
        accuracies[level] = acc

        # classification report
   
        labels = list(range(len(dataset.class_maps[level])))

        report = classification_report(
            trues[level],
            preds[level],
            labels=labels,
            target_names=list(dataset.class_maps[level].keys()),
            zero_division=0,
            output_dict=True
        )
        pd.DataFrame(report).transpose().to_csv(
            f"{METRICS_DIR}/{split}_{level}_classification_report.csv"
        )

        # confusion matrix
        cm = confusion_matrix(trues[level], preds[level])
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=dataset.class_maps[level].keys(),
            yticklabels=dataset.class_maps[level].keys()
        )
        plt.title(f"{level.capitalize()} Confusion Matrix ({split})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(f"{METRICS_DIR}/{split}_{level}_confusion_matrix.png")
        plt.close()

    return accuracies

# =========================
# TRAINING
# =========================
def train():
    train_ds = HierarchicalDataset(os.path.join(DATA_ROOT, "train"))
    val_ds   = HierarchicalDataset(os.path.join(DATA_ROOT, "val"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = HierarchicalCNN(
        len(train_ds.class_maps["binary"]),
        len(train_ds.class_maps["class"]),
        len(train_ds.class_maps["genus"]),
        len(train_ds.class_maps["species"])
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses, val_losses = [], []
    val_acc_history = {l: [] for l in LEVELS}

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, targets in train_loader:
            images = images.to(DEVICE)
            targets = {k: v.to(DEVICE) for k, v in targets.items()}

            loss = hierarchical_loss(model(images), targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE)
                targets = {k: v.to(DEVICE) for k, v in targets.items()}
                val_loss += hierarchical_loss(model(images), targets).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        accs = evaluate(model, val_loader, val_ds)
        for l in LEVELS:
            val_acc_history[l].append(accs[l])

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print("✔ Saved best model")

    # =========================
    # CURVES
    # =========================
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Hierarchical Loss Curve")
    plt.savefig(f"{METRICS_DIR}/loss_curve.png")
    plt.close()

    plt.figure()
    for l in LEVELS:
        plt.plot(val_acc_history[l], label=f"{l} Acc")
    plt.legend()
    plt.title("Validation Accuracy per Level")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epoch")
    plt.savefig(f"{METRICS_DIR}/accuracy_curve.png")
    plt.close()

    print("✅ Training & evaluation complete")

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    train()
