import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random
import glob
from PIL import Image
from torchvision import transforms
import json
import argparse
from .utils import train_epoch, validate_epoch, EarlyStopping, SameDifferentDataset

class SameDifferentCNN(nn.Module):
    def __init__(self, initial_kernel_size=2, initial_filters=6, dropout_rate_fc=0.3):
        super(SameDifferentCNN, self).__init__()
        
        # First layer: 2x2 filters with 6 filters
        self.conv1 = nn.Conv2d(3, initial_filters, kernel_size=initial_kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(initial_filters)
        
        # Subsequent layers: 2x2 filters with doubling filter counts
        self.conv2 = nn.Conv2d(initial_filters, initial_filters * 2, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(initial_filters * 2)
        
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout2d = nn.Dropout2d(0.1)  # Add spatial dropout
        
        # Calculate the size of flattened features
        self._to_linear = None
        self._initialize_size()
        
        # Three FC layers with 1024 units each
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        
        # Dropouts for FC layers
        self.dropout1 = nn.Dropout(dropout_rate_fc)
        self.dropout2 = nn.Dropout(dropout_rate_fc)
        self.dropout3 = nn.Dropout(dropout_rate_fc)
        
        # Final classification layer
        self.classifier = nn.Linear(1024, 2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_size(self):
        # Temporarily set batchnorm layers to eval mode
        self.bn1.eval()
        self.bn2.eval()
        
        x = torch.randn(1, 3, 128, 128)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.reshape(x.size(0), -1)
        self._to_linear = x.size(1)
        
        # Set batchnorm layers back to train mode
        self.bn1.train()
        self.bn2.train()
    
    def _initialize_weights(self):
        # Initialize convolutional layers
        for m in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        # Initialize fully connected layers
        for fc in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(fc.weight)
            nn.init.constant_(fc.bias, 0)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x):
        # Convolutional layers with spatial dropout
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2d(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
       
        x = x.reshape(x.size(0), -1)
        
        # FC layers with dropout
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.dropout3(F.relu(self.fc3(x)))
        
        return self.classifier(x)