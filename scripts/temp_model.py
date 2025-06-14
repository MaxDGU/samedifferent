import torch
import torch.nn as nn
import torch.nn.functional as F

# This is a temporary, hybrid model definition created to match the
# exact architecture found in the 'model_seed_*.pt' files.
# It combines the 2 convolutional layers from the 'conv2' architecture
# with the ModuleList-based FC layers from the 'conv6lr' architecture.

class SameDifferentCNN(nn.Module):
    def __init__(self):
        super(SameDifferentCNN, self).__init__()
        
        # --- Convolutional layers from 'baselines/models/conv2.py' ---
        self.conv1 = nn.Conv2d(3, 6, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(6)
        
        self.conv2 = nn.Conv2d(6, 12, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout2d = nn.Dropout2d(0.1)
        
        # --- FC layers and Classifier from 'meta_baseline/models/conv6lr.py' ---
        # but adapted for the output of the conv2 layers.
        
        self._to_linear = None
        self._initialize_size() # Calculate the flattened size after conv layers
        
        # This structure matches the keys 'fc_layers.0.weight', etc.
        self.fc_layers = nn.ModuleList([
            nn.Linear(self._to_linear, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024)
        ])
        
        # This structure matches the keys 'layer_norms.0.weight', etc.
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(1024),
            nn.LayerNorm(1024),
            nn.LayerNorm(1024)
        ])
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(3)
        ])
        
        # The final classifier layer
        self.classifier = nn.Linear(1024, 2)

        # MAML-related parameters found in the checkpoint are not part of the
        # base model definition (e.g., temperature, lr_*.
        # They will be ignored during loading with strict=False.

    def _initialize_size(self):
        # Calculate the size of the output of the conv layers for the FC layers
        x = torch.randn(1, 3, 128, 128)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.reshape(x.size(0), -1)
        self._to_linear = x.size(1)

    def forward(self, x):
        # Conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
        
        x = x.reshape(x.size(0), -1)
        
        # FC block
        for fc, ln, dropout in zip(self.fc_layers, self.layer_norms, self.dropouts):
            x = dropout(F.relu(ln(fc(x))))
            
        x = self.classifier(x)
        return x 