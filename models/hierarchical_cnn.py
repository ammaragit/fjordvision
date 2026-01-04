import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalCNN(nn.Module):
    def __init__(self, n_binary, n_class, n_genus, n_species):
        super().__init__()

        # Shared backbone
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

        # Hierarchy heads
        self.binary_head  = nn.Linear(256, n_binary)
        self.class_head   = nn.Linear(256 + n_binary, n_class)
        self.genus_head   = nn.Linear(256 + n_class, n_genus)
        self.species_head = nn.Linear(256 + n_genus, n_species)

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)
        x = F.relu(self.fc(x))

        out = {}

        # Level 1
        out["binary"] = self.binary_head(x)
        b = F.softmax(out["binary"], dim=1)

        # Level 2 (conditioned)
        out["class"] = self.class_head(torch.cat([x, b], dim=1))
        c = F.softmax(out["class"], dim=1)

        # Level 3
        out["genus"] = self.genus_head(torch.cat([x, c], dim=1))
        g = F.softmax(out["genus"], dim=1)

        # Level 4
        out["species"] = self.species_head(torch.cat([x, g], dim=1))

        return out
