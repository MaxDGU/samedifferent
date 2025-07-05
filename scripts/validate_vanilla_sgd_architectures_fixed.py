#!/usr/bin/env python3
"""
Vanilla SGD Architecture Validation Script - FIXED

This script validates vanilla SGD performance across different CNN architectures 
(conv2, conv4, conv6) using the CORRECT data loading approach from the working scripts.

Uses VanillaPBDataset that loads from data/meta_h5/pb/, not data/vanilla_h5/
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import h5py

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import existing baselines code
from baselines.models import Conv2CNN, Conv4CNN, Conv6CNN
from baselines.models.utils import EarlyStopping

# Use the CORRECT VanillaPBDataset that actually works
class VanillaPBDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that loads H5 data on-demand for vanilla SGD."""
    def __init__(self, tasks, split='train', data_dir='data/meta_h5/pb', support_sizes=[4, 6, 8, 10]):
        self.data_dir = data_dir
        self.split = split
        self.sample_info = []
        
        print(f"Indexing vanilla dataset for {split} split with tasks: {tasks}")
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
        
        with h5py.File(filepath, 'r') as f:
            if sample_type == 'support':
                image = f['support_images'][episode_idx, sample_idx]
                label = f['support_labels'][episode_idx, sample_idx]
            else:
                image = f['query_images'][episode_idx, sample_idx]
                label = f['query_labels'][episode_idx, sample_idx]
        
        image = torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0
        label = torch.tensor(int(label), dtype=torch.long)
        
        return image, label

def validate_epoch(model, loader, criterion, device):
    """Validate model performance using existing pattern."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in tqdm(loader, desc="Validation"):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            
            # Handle output format (same as working scripts)
            if outputs.dim() > 1 and outputs.shape[1] > 1:
                outputs = outputs[:, 1] - outputs[:, 0]
            else:
                outputs = outputs.squeeze()

            loss = criterion(outputs, labels.float())
            running_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > 0.5).long()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch using existing pattern."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data, labels in tqdm(loader, desc="Training"):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        
        # Handle output format (same as working scripts)
        if outputs.dim() > 1 and outputs.shape[1] > 1:
            outputs = outputs[:, 1] - outputs[:, 0]
        else:
            outputs = outputs.squeeze()
        
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_architecture(arch_name, model_class, args, device):
    """Train a specific architecture with vanilla SGD."""
    print(f"\n{'='*60}")
    print(f"Training {arch_name} Architecture with Vanilla SGD")
    print(f"{'='*60}")
    
    # Create model 
    model = model_class().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create datasets using the CORRECT approach
    ALL_PB_TASKS = [
        'regular', 'lines', 'open', 'wider_line', 'scrambled',
        'random_color', 'arrows', 'irregular', 'filled', 'original'
    ]
    
    # Use the CORRECT data directory and dataset class
    train_dataset = VanillaPBDataset(tasks=ALL_PB_TASKS, split='train', data_dir=args.data_dir)
    val_dataset = VanillaPBDataset(tasks=ALL_PB_TASKS, split='val', data_dir=args.data_dir)
    test_dataset = VanillaPBDataset(tasks=ALL_PB_TASKS, split='test', data_dir=args.data_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Setup training (same as working scripts)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    early_stopper = EarlyStopping(patience=args.patience, verbose=True, 
                                 path=f'{arch_name.lower()}_best_model.pt')
    
    # Training loop
    best_val_acc = 0.0
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"Training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save metrics
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)
        
        # Early stopping
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # Final test evaluation
    print(f"\nLoading best model for final testing...")
    model.load_state_dict(torch.load(f'{arch_name.lower()}_best_model.pt'))
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    
    print(f"\n{arch_name} Final Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Clean up model file
    os.remove(f'{arch_name.lower()}_best_model.pt')
    
    return {
        'architecture': arch_name,
        'parameters': total_params,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'metrics': metrics
    }

def plot_results(results, save_dir):
    """Create bar chart comparing architectures."""
    architectures = [r['architecture'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    val_accs = [r['best_val_acc'] for r in results]
    
    x = np.arange(len(architectures))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, test_accs, width, label='Test Accuracy', 
                   alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, val_accs, width, label='Best Val Accuracy', 
                   alpha=0.8, color='orange')
    
    ax.set_xlabel('Architecture', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Vanilla SGD Performance Across Architectures\n(Validation of Bar Chart Results)', 
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(architectures)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'vanilla_sgd_architecture_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'vanilla_sgd_architecture_comparison.pdf'), 
                bbox_inches='tight')
    print(f"\nPlot saved to {os.path.join(save_dir, 'vanilla_sgd_architecture_comparison.png')}")
    
    return plt

def main():
    parser = argparse.ArgumentParser(description='Validate Vanilla SGD Across Architectures')
    
    # Data and save paths - CORRECTED
    parser.add_argument('--data_dir', type=str, default='data/meta_h5/pb',
                       help='Directory for meta H5 dataset (CORRECTED)')
    parser.add_argument('--save_dir', type=str, default='results/vanilla_sgd_validation',
                       help='Directory to save results')
    
    # Training parameters (same as working scripts)
    parser.add_argument('--epochs', type=int, default=100,
                       help='Max number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Architecture selection
    parser.add_argument('--architectures', nargs='+', default=['conv2', 'conv4', 'conv6'],
                       choices=['conv2', 'conv4', 'conv6'],
                       help='Architectures to test')
    
    args = parser.parse_args()
    
    # Set device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seeds (same as working scripts)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, f"seed_{args.seed}")
    os.makedirs(save_dir, exist_ok=True)
    
    # Architecture mapping
    arch_models = {
        'conv2': Conv2CNN,
        'conv4': Conv4CNN,
        'conv6': Conv6CNN
    }
    
    # Train each architecture
    results = []
    
    for arch_name in args.architectures:
        if arch_name in arch_models:
            result = train_architecture(arch_name.upper(), arch_models[arch_name], args, device)
            results.append(result)
        else:
            print(f"Warning: Unknown architecture {arch_name}")
    
    # Save results
    results_summary = {
        'experiment': 'vanilla_sgd_architecture_validation_FIXED',
        'description': 'Validation of vanilla SGD performance using CORRECT data loading from working scripts',
        'args': vars(args),
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_dir, 'validation_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Create comparison plot
    plot_results(results, save_dir)
    
    # Print summary
    print(f"\n{'='*80}")
    print("VANILLA SGD ARCHITECTURE VALIDATION SUMMARY - FIXED")
    print(f"{'='*80}")
    print("Using CORRECT data loading approach:")
    print("- Data directory: data/meta_h5/pb/ (CORRECTED)")
    print("- Dataset class: VanillaPBDataset (from working scripts)")
    print("- Architecture definitions from baselines/models/")
    print(f"{'='*80}")
    
    for result in results:
        print(f"{result['architecture']:6} | Parameters: {result['parameters']:,} | "
              f"Test Acc: {result['test_acc']:6.2f}% | Val Acc: {result['best_val_acc']:6.2f}%")
    
    print(f"\nResults saved to: {save_dir}")
    print(f"Expected: ~50% accuracy (chance level) for all architectures")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
