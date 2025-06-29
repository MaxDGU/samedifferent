import torch
import torch.nn as nn
import torch.nn.functional as F

class SameDifferentCNN(nn.Module):
    def __init__(self):
        super(SameDifferentCNN, self).__init__()
        
        # 6-layer CNN with increasing filter counts
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=False)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, track_running_stats=False)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256, track_running_stats=False)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512, track_running_stats=False)
        
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(1024, track_running_stats=False)
        
        self.pool = nn.MaxPool2d(2)
        self.dropout2d = nn.Dropout2d(0.3)
        
        self._to_linear = None
        self._initialize_size()
        
        # FC layers with decreasing sizes
        self.fc_layers = nn.ModuleList([
            nn.Linear(self._to_linear, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 256)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(1024),
            nn.LayerNorm(512),
            nn.LayerNorm(256)
        ])
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(3)
        ])
        
        self.classifier = nn.Linear(256, 2)
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Learnable per-layer learning rates
        self.lr_conv = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.01) for _ in range(6)
        ])
        self.lr_fc = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.01) for _ in range(3)
        ])
        self.lr_classifier = nn.Parameter(torch.ones(1) * 0.01)
        
        self._initialize_weights()
    
    def _initialize_size(self):
        # This function calculates the input size of the first fully connected
        # layer by performing a forward pass with a dummy tensor.
        # It assumes an input image size of 128x128.
        x = torch.randn(1, 3, 128, 128)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = x.reshape(x.size(0), -1)
        self._to_linear = x.size(1)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear) and m != self.classifier:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
        
        if self.classifier.bias is not None:
            nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        # Convolutional blocks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        x = self.dropout2d(x)
        
        # Flatten for FC layers
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers
        for i, (fc, ln, drop) in enumerate(zip(self.fc_layers, self.layer_norms, self.dropouts)):
            x = drop(F.relu(ln(fc(x))))
            
        # Classifier
        x = self.classifier(x) / self.temperature
        return x
    
    def get_layer_lrs(self):
        """Returns the learnable LRs for each layer group."""
        return {
            'conv': [lr.item() for lr in self.lr_conv],
            'fc': [lr.item() for lr in self.lr_fc],
            'classifier': self.lr_classifier.item()
        } 