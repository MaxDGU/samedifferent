#!/usr/bin/env python3
"""
Sample Efficiency Comparison (Simplified Version)

This is a simplified version that can be tested locally without learn2learn.
It focuses on the vanilla SGD implementation and provides the framework for
the full comparison.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also try current working directory (in case running from project root)
cwd = os.getcwd()
if cwd not in sys.path and os.path.exists(os.path.join(cwd, 'meta_baseline')):
    sys.path.insert(0, cwd)

# Import required modules
from meta_baseline.models.conv6lr import SameDifferentCNN
import h5py

# Define PB tasks
ALL_PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]

class VanillaPBDataset(torch.utils.data.Dataset):
    """
    Memory-efficient dataset that loads H5 data on-demand for vanilla SGD.
    Stores file paths and indices, loads actual data lazily in __getitem__.
    """
    def __init__(self, tasks, split='train', data_dir='data/meta_h5/pb', support_sizes=[4, 6, 8, 10]):
        self.data_dir = data_dir
        self.split = split
        self.sample_info = []  # List of (filepath, episode_idx, sample_type, sample_idx_in_episode)
        
        print(f"Indexing vanilla dataset for {split} split...")
        total_samples = 0
        
        for task in tasks:
            for support_size in support_sizes:
                filename = f"{task}_support{support_size}_{split}.h5"
                filepath = os.path.join(data_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"Warning: File not found: {filepath}")
                    continue
                
                print(f"Indexing {filename}...")
                with h5py.File(filepath, 'r') as f:
                    num_episodes = f['support_images'].shape[0]
                    support_size_actual = f['support_images'].shape[1]
                    query_size_actual = f['query_images'].shape[1]
                    
                    # Index support samples
                    for episode_idx in range(num_episodes):
                        for sample_idx in range(support_size_actual):
                            self.sample_info.append((filepath, episode_idx, 'support', sample_idx))
                            total_samples += 1
                    
                    # Index query samples  
                    for episode_idx in range(num_episodes):
                        for sample_idx in range(query_size_actual):
                            self.sample_info.append((filepath, episode_idx, 'query', sample_idx))
                            total_samples += 1
        
        print(f"Indexed {total_samples} individual samples for vanilla SGD ({split} split)")
    
    def __len__(self):
        return len(self.sample_info)
    
    def __getitem__(self, idx):
        if idx >= len(self.sample_info):
            raise IndexError("Index out of range")
        
        filepath, episode_idx, sample_type, sample_idx = self.sample_info[idx]
        
        # Load data on-demand
        with h5py.File(filepath, 'r') as f:
            if sample_type == 'support':
                image = f['support_images'][episode_idx, sample_idx]  # Shape: (H, W, C)
                label = f['support_labels'][episode_idx, sample_idx]  # Scalar
            else:  # query
                image = f['query_images'][episode_idx, sample_idx]  # Shape: (H, W, C)
                label = f['query_labels'][episode_idx, sample_idx]  # Scalar
        
        # Convert to tensor and normalize
        # Convert from HWC to CHW format and normalize to [0, 1]
        image = torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0
        label = torch.tensor(int(label), dtype=torch.long)
        
        return image, label

def validate_vanilla_model(model, val_loader, device, loss_fn):
    """Validate vanilla SGD model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, labels in val_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            
            # Handle output format
            if outputs.dim() > 1 and outputs.shape[1] > 1:
                outputs = outputs[:, 1] - outputs[:, 0]
            else:
                outputs = outputs.squeeze()
            
            loss = loss_fn(outputs, labels.float())
            total_loss += loss.item()
            
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    avg_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    
    return avg_loss, avg_acc

def train_vanilla_sgd(args, device, save_dir):
    """Train Vanilla SGD."""
    print("Training Vanilla SGD...")
    
    # Create model
    model = SameDifferentCNN().to(device)
    
    # Create datasets - flatten episodic data into individual samples for vanilla SGD
    train_dataset = VanillaPBDataset(tasks=ALL_PB_TASKS, split='train', data_dir=args.data_dir)
    val_dataset = VanillaPBDataset(tasks=ALL_PB_TASKS, split='val', data_dir=args.data_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.vanilla_batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.vanilla_batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.vanilla_lr)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Training tracking
    data_points_seen = []
    val_accuracies = []
    total_data_points = 0
    
    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc=f"Vanilla SGD Epoch {epoch+1}")):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(data)
            if outputs.dim() > 1 and outputs.shape[1] > 1:
                outputs = outputs[:, 1] - outputs[:, 0]
            else:
                outputs = outputs.squeeze()
            
            loss = loss_fn(outputs, labels.float())
            loss.backward()
            optimizer.step()
            
            # Track metrics
            epoch_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).long()
            epoch_correct += (predicted == labels).sum().item()
            epoch_total += labels.size(0)
            total_data_points += labels.size(0)
            
            # Validate every few batches
            if batch_idx % args.val_frequency == 0:
                val_loss, val_acc = validate_vanilla_model(model, val_loader, device, loss_fn)
                data_points_seen.append(total_data_points)
                val_accuracies.append(val_acc)
                print(f"  Batch {batch_idx}: Data points seen: {total_data_points}, Val Acc: {val_acc:.2f}%")
        
        # End of epoch validation
        val_loss, val_acc = validate_vanilla_model(model, val_loader, device, loss_fn)
        data_points_seen.append(total_data_points)
        val_accuracies.append(val_acc)
        
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = 100.0 * epoch_correct / epoch_total
        
        print(f"Vanilla SGD Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save results
    results = {
        'method': 'Vanilla SGD',
        'data_points_seen': data_points_seen,
        'val_accuracies': val_accuracies,
        'total_data_points': total_data_points
    }
    
    with open(os.path.join(save_dir, 'vanilla_sgd_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def plot_results(results, save_dir):
    """Create plot of sample efficiency."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(results['data_points_seen'], results['val_accuracies'], 
            color='green', linestyle='-', linewidth=2, 
            label=results['method'], marker='o', markersize=4)
    
    plt.xlabel('Number of Data Points Seen', fontsize=14)
    plt.ylabel('Validation Accuracy (%)', fontsize=14)
    plt.title('Sample Efficiency: Vanilla SGD Training (Conv6 Architecture)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(save_dir, 'vanilla_sgd_sample_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'vanilla_sgd_sample_efficiency.pdf'), bbox_inches='tight')
    
    print(f"Plot saved to {os.path.join(save_dir, 'vanilla_sgd_sample_efficiency.png')}")
    
    return plt

def main():
    parser = argparse.ArgumentParser(description='Sample Efficiency Comparison (Simplified Version)')
    
    # Data directories
    parser.add_argument('--data_dir', type=str, default='data/meta_h5/pb', 
                       help='Directory for HDF5 data')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, 
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    
    # Vanilla SGD parameters
    parser.add_argument('--vanilla_batch_size', type=int, default=32, 
                       help='Vanilla SGD batch size')
    parser.add_argument('--vanilla_lr', type=float, default=1e-3, 
                       help='Vanilla SGD learning rate')
    
    # Validation and saving
    parser.add_argument('--val_frequency', type=int, default=1000, 
                       help='Validation frequency (in batches)')
    parser.add_argument('--save_dir', type=str, default='results/sample_efficiency_simple', 
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, f"seed_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Run vanilla SGD experiment
    print("Running Vanilla SGD experiment...")
    vanilla_results = train_vanilla_sgd(args, device, save_dir)
    
    # Create plot
    plot_results(vanilla_results, save_dir)
    
    # Save combined results
    combined_results = {
        'args': vars(args),
        'results': [vanilla_results],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_dir, 'combined_results.json'), 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"Sample efficiency test completed. Results saved to {save_dir}")
    print(f"Final validation accuracy: {vanilla_results['val_accuracies'][-1]:.2f}%")
    print(f"Total data points seen: {vanilla_results['total_data_points']:,}")

if __name__ == '__main__':
    main() 