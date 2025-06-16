import torch
import torch.nn as nn
import torch.nn.functional as F

# These are specific model definitions to match the architectures
# found in the older "Meta-PB" checkpoints. Each is different.

class PB_Conv2(nn.Module):
    """Matches the Meta-PB-Conv2 checkpoints."""
    def __init__(self, num_classes=2):
        super(PB_Conv2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, 2)
        self.bn2 = nn.BatchNorm2d(12)
        self._to_linear = 10800 # Calculated for 32x32 input
        self.fc_layers = nn.ModuleList([
            nn.Linear(self._to_linear, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
        ])
        self.classifier = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        # Check for size mismatch and print info
        if x.size(1) != self._to_linear:
            print(f"Warning: PB_Conv2 size mismatch. Expected {self._to_linear}, got {x.size(1)}")
            # Fallback to reshape dynamically if needed, though this shouldn't happen
            # if the input size is consistent.
            x = x.view(x.size(0), self._to_linear)

        for fc in self.fc_layers:
            x = self.relu(fc(x))
        x = self.classifier(x)
        return x

class PB_Conv4(nn.Module):
    """Matches the Meta-PB-Conv4 checkpoints."""
    def __init__(self, num_classes=2):
        super(PB_Conv4, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 4)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 2)
        self._to_linear = 15360 # Calculated for 32x32 input
        self.fc_layers = nn.ModuleList([
            nn.Linear(self._to_linear, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
        ])
        self.classifier = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
            
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        if x.size(1) != self._to_linear:
            print(f"Warning: PB_Conv4 size mismatch. Expected {self._to_linear}, got {x.size(1)}")
        for fc in self.fc_layers:
            x = self.relu(fc(x))
        x = self.classifier(x)
        return x

class PB_Conv6(nn.Module):
    """Matches the Meta-PB-Conv6 checkpoints."""
    def __init__(self, num_classes=2):
        super(PB_Conv6, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self._to_linear = 1024 # Calculated for 32x32 input
        self.fc_layers = nn.ModuleList([
            nn.Linear(self._to_linear, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
        ])
        self.classifier = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        if x.size(1) != self._to_linear:
            print(f"Warning: PB_Conv6 size mismatch. Expected {self._to_linear}, got {x.size(1)}")
        for fc in self.fc_layers:
            x = self.relu(fc(x))
        x = self.classifier(x)
        return x 