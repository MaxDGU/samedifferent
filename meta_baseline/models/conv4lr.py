import os
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import learn2learn as l2l
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import math
import random
import glob
import json
import gc
import sys
import argparse
from PIL import Image
from torchvision import transforms
#reused methods across all models 
from .utils_meta import train_epoch, validate, EarlyStopping, SameDifferentDataset, collate_episodes

# Import vanilla model architecture
class SameDifferentCNN(nn.Module):
    """
    A 4-layer CNN aligned with the paper's specification.
    - First layer: 12 filters, 4x4 kernel.
    - Subsequent layers: 2x2 kernels, doubling filters.
    - Pooling after each conv layer.
    - Three 1024-unit FC layers with LayerNorm and Dropout.
    """
    def __init__(self, dropout_rate_fc=0.5):
        super(SameDifferentCNN, self).__init__()
        
        # Convolutional layers based on the paper's description
        self.conv1 = nn.Conv2d(3, 12, kernel_size=4, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=2, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, kernel_size=2, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 96, kernel_size=2, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(96)
        
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Calculate the size of flattened features dynamically
        self._to_linear = None
        self._initialize_size()
        
        # Three fully connected layers with 1024 units, with LayerNorm and Dropout
        self.fc_layers = nn.ModuleList([
            nn.Linear(self._to_linear, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 1024)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(1024),
            nn.LayerNorm(1024),
            nn.LayerNorm(1024)
        ])
        
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate_fc) for _ in range(3)
        ])
        
        # Final classification layer
        self.classifier = nn.Linear(1024, 2)
        
        self._initialize_weights()

    def _initialize_size(self):
        # Dummy forward pass to calculate the flattened size after conv layers
        x = torch.randn(1, 3, 128, 128)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.reshape(x.size(0), -1)
        self._to_linear = x.size(1)

    def _initialize_weights(self):
        # Using Xavier initialization as specified in the paper
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Convolutional pathway
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = x.reshape(x.size(0), -1)
        
        # Fully connected pathway with LayerNorm and Dropout
        for fc, ln, dropout in zip(self.fc_layers, self.layer_norms, self.dropouts):
            x = dropout(F.relu(ln(fc(x))))
        
        x = self.classifier(x)
        return x # Output raw logits

def main(seed=None, output_dir=None, pb_data_dir='data/pb/pb'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/pb/pb',
                        help='Directory containing the PB dataset')
    parser.add_argument('--output_dir', type=str, default='results/meta_baselines',
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--support_size', type=int, default=10,
                        help='Number of support examples per class')
    parser.add_argument('--adaptation_steps', type=int, default=5,
                        help='Number of adaptation steps during training')
    parser.add_argument('--test_adaptation_steps', type=int, default=15,
                        help='Number of adaptation steps during testing')
    parser.add_argument('--inner_lr', type=float, default=0.05,
                        help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                        help='Outer loop learning rate')
    args = parser.parse_args()
    
    try:
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Running on CPU, but this will be slow and may not work correctly.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        print(f"Using device: {device}")
        
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
        
        if args.seed is not None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.cuda.manual_seed(args.seed)
        
        arch_dir = os.path.join(args.output_dir, 'conv4', f'seed_{args.seed}')
        os.makedirs(arch_dir, exist_ok=True)
        
        print("\nCreating datasets...")
        train_tasks = ['regular', 'lines', 'open', 'wider_line', 'scrambled',
                       'random_color', 'arrows', 'irregular', 'filled', 'original']
        train_dataset = SameDifferentDataset(args.data_dir, train_tasks, 'train', support_sizes=[args.support_size])
        val_dataset = SameDifferentDataset(args.data_dir, train_tasks, 'val', support_sizes=[args.support_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=4, pin_memory=True, collate_fn=collate_episodes)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                              num_workers=4, pin_memory=True, collate_fn=collate_episodes)
        
        print("\nCreating conv4 model")
        model = SameDifferentCNN().to(device)
        
        print(f"Model created and initialized on {device}")
        
        maml = l2l.algorithms.MAML(
            model,
            lr=args.inner_lr,
            first_order=False,
            allow_unused=True,
            allow_nograd=True
        )
        
        optimizer = torch.optim.Adam(maml.parameters(), lr=args.outer_lr)
        
        # Training loop
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(
                maml, train_loader, optimizer, device,
                adaptation_steps=args.adaptation_steps
            )
            
            # Validate
            val_loss, val_acc = validate(
                maml, val_loader, device,
                adaptation_steps=args.adaptation_steps
            )
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(arch_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'maml_state_dict': maml.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc
                }, model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break
        
        # Test on all tasks
        test_results = {}
        for task in train_tasks:
            print(f"\nTesting on task: {task}")
            test_dataset = SameDifferentDataset(args.data_dir, [task], 'test', support_sizes=[args.support_size])
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True, collate_fn=collate_episodes)
            
            test_loss, test_acc = validate(
                maml, test_loader, device,
                adaptation_steps=args.test_adaptation_steps
            )
            
            test_results[task] = {
                'loss': test_loss,
                'accuracy': test_acc
            }
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Save final results
        results = {
            'args': vars(args),
            'test_results': test_results,
            'best_val_acc': best_val_acc
        }
        
        with open(os.path.join(arch_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return test_results, best_val_acc
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == '__main__':
    main()


