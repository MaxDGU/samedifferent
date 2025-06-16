import torch
import torch.nn as nn
import torch.nn.functional as F

# This is a temporary, hybrid model definition created to match the
# exact architecture found in the 'model_seed_*.pt' PB files.
# It appears to be a different ConvNet structure than the other experiments.

class SameDifferentCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SameDifferentCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, kernel_size=2, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(6)
        
        self.conv2 = nn.Conv2d(6, 12, kernel_size=2, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(12)
        
        self._to_linear = None
        # Dynamically calculate the size of the output of the conv layers
        self._initialize_size() 
        
        self.fc1 = nn.Linear(self._to_linear, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)
        self.classifier = nn.Linear(20, num_classes)

    def _initialize_size(self):
        # A dummy forward pass to calculate the flattened size after conv layers
        with torch.no_grad():
            x = torch.randn(1, 3, 32, 32) # Assuming 32x32 input for calculation
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            self._to_linear = x.view(x.size(0), -1).size(1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.classifier(x)
        return x 