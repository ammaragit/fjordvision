# models/hierarchical_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalCNN(nn.Module):
    def __init__(self, num_binary, num_class, num_genus, num_species):
        super(HierarchicalCNN, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_binary + num_class + num_genus + num_species)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        # Split outputs into respective levels
        num_binary = x[:, :self.num_binary]
        num_class = x[:, self.num_binary:self.num_binary + self.num_class]
        num_genus = x[:, self.num_binary + self.num_class:self.num_binary + self.num_class + self.num_genus]
        num_species = x[:, self.num_binary + self.num_class + self.num_genus:]
        
        return {
            "binary": num_binary,
            "class": num_class,
            "genus": num_genus,
            "species": num_species
        }