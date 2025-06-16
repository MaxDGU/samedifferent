import torch
import torch.nn as nn
import torch.nn.functional as F

# These are specific model definitions to match the architectures
# found in the older "Meta-PB" checkpoints. Each architecture is
# different and the layer sizes are hardcoded based on error logs.

class PB_Conv2(nn.Module):
    """Matches the Meta-PB-Conv2 checkpoints.
    Error log indicates fc_layers.0.weight input is 13068.
    """
    def __init__(self, num_classes=2):
        super(PB_Conv2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, 2)
        self.bn2 = nn.BatchNorm2d(12)
        
        self.fc_layers = nn.ModuleList([
            nn.Linear(13068, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
        ])
        self.classifier = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        for fc in self.fc_layers:
            x = self.relu(fc(x))
        x = self.classifier(x)
        return x

class PB_Conv4(nn.Module):
    """Matches the Meta-PB-Conv4 checkpoints.
    Error log indicates fc_layers.0.weight input is 7776.
    And conv layers are larger.
    """
    def __init__(self, num_classes=2):
        super(PB_Conv4, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 4)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 2)
        self.bn2 = nn.BatchNorm2d(24)

        self.fc_layers = nn.ModuleList([
            nn.Linear(7776, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
        ])
        self.classifier = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
            
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        for fc in self.fc_layers:
            x = self.relu(fc(x))
        x = self.classifier(x)
        return x

class PB_Conv6(nn.Module):
    """Matches the Meta-PB-Conv6 checkpoints.
    Error log indicates this is actually a Conv2-like architecture.
    """
    def __init__(self, num_classes=2):
        super(PB_Conv6, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 12, 2)
        self.bn2 = nn.BatchNorm2d(12)
        
        self.fc_layers = nn.ModuleList([
            nn.Linear(13068, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024),
        ])
        self.classifier = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        for fc in self.fc_layers:
            x = self.relu(fc(x))
        x = self.classifier(x)
        return x 