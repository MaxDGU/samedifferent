#!/usr/bin/env python3
"""
Sample Efficiency Comparison: FOMAML vs Second-Order MAML vs Vanilla SGD

This script compares the sample efficiency of three training methods on the conv6 architecture:
1. First-Order MAML (FOMAML)
2. Second-Order MAML 
3. Vanilla SGD

The comparison tracks validation accuracy as a function of total data points seen during training.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import copy
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
import learn2learn as l2l
from meta_baseline.models.conv6lr import SameDifferentCNN
from meta_baseline.models.utils_meta import SameDifferentDataset, collate_episodes
import h5py

# Define PB tasks
ALL_PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]

# Support and query sizes for meta-learning
VARIABLE_SUPPORT_SIZES = [4, 6, 8, 10]
FIXED_QUERY_SIZE = 2

class VanillaPBDataset(torch.utils.data.Dataset):
    """
    Simple dataset that flattens meta-learning H5 episodes into individual samples for vanilla SGD.
    """
    def __init__(self, tasks, split='train', data_dir='data/meta_h5/pb', support_sizes=[4, 6, 8, 10]):
        self.images = []
        self.labels = []
        
        print(f"Loading vanilla dataset for {split} split...")
        for task in tasks:
            for support_size in support_sizes:
                filename = f"{task}_support{support_size}_{split}.h5"
                filepath = os.path.join(data_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"Warning: File not found: {filepath}")
                    continue
                
                print(f"Loading {filename} for vanilla dataset...")
                with h5py.File(filepath, 'r') as f:
                    # Load support images and labels
                    if 'support_images' in f and 'support_labels' in f:
                        support_images = f['support_images'][:]  # Shape: (episodes, support_size, H, W, C)
                        support_labels = f['support_labels'][:]  # Shape: (episodes, support_size)
                        
                        # Flatten episodes and support examples into individual samples
                        num_episodes, support_size_actual, H, W, C = support_images.shape
                        support_images = support_images.reshape(-1, H, W, C)  # (episodes * support_size, H, W, C)
                        support_labels = support_labels.reshape(-1)  # (episodes * support_size,)
                        
                        self.images.append(support_images)
                        self.labels.append(support_labels)
                    
                    # Load query images and labels
                    if 'query_images' in f and 'query_labels' in f:
                        query_images = f['query_images'][:]  # Shape: (episodes, query_size, H, W, C)
                        query_labels = f['query_labels'][:]  # Shape: (episodes, query_size)
                        
                        # Flatten episodes and query examples into individual samples
                        num_episodes, query_size_actual, H, W, C = query_images.shape
                        query_images = query_images.reshape(-1, H, W, C)  # (episodes * query_size, H, W, C)
                        query_labels = query_labels.reshape(-1)  # (episodes * query_size,)
                        
                        self.images.append(query_images)
                        self.labels.append(query_labels)
        
        if self.images:
            # Concatenate all images and labels
            self.images = np.concatenate(self.images, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
            
            print(f"Loaded {len(self.images)} individual samples for vanilla SGD ({split} split)")
        else:
            print(f"No data loaded for vanilla SGD ({split} split)")
            self.images = np.array([])
            self.labels = np.array([])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if len(self.images) == 0:
            raise IndexError("No data available")
        
        # Get image and label
        image = self.images[idx]  # Shape: (H, W, C)
        label = self.labels[idx]  # Scalar
        
        # Convert to tensor and normalize
        # Convert from HWC to CHW format and normalize to [0, 1]
        image = torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0
        label = torch.tensor(int(label), dtype=torch.long)
        
        return image, label

def accuracy(predictions, targets):
    """Binary classification accuracy using raw logits."""
    with torch.no_grad():
        # Compare raw logits to 0.0, not softmax probabilities to 0.5
        predicted_labels = (predictions[:, 1] > 0.0).float()
        
        # Safely handle targets of different dimensions
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
        
        return (predicted_labels == targets).float().mean()

def fast_adapt(batch, learner, loss_fn, adaptation_steps, device):
    """Perform fast adaptation for meta-learning."""
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    
    # Separate support and query
    support_size = data.size(0) // 2  # Assuming equal support and query sizes
    support_data, query_data = data[:support_size], data[support_size:]
    support_labels, query_labels = labels[:support_size], labels[support_size:]
    
    # Adapt on support set
    for step in range(adaptation_steps):
        support_preds = learner(support_data)
        support_loss = loss_fn(support_preds, support_labels)
        learner.adapt(support_loss)
    
    # Evaluate on query set
    query_preds = learner(query_data)
    query_loss = loss_fn(query_preds, query_labels)
    query_acc = accuracy(query_preds, query_labels)
    
    return query_loss, query_acc

def validate_meta_model(maml, val_loader, device, adaptation_steps, loss_fn):
    """Validate meta-learning model."""
    maml.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            learner = maml.clone()
            val_loss, val_acc = fast_adapt(batch, learner, loss_fn, adaptation_steps, device)
            total_loss += val_loss.item()
            total_acc += val_acc.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_acc

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

def train_fomaml(args, device, save_dir):
    """Train First-Order MAML."""
    print("Training First-Order MAML...")
    
    # Create model and MAML wrapper
    model = SameDifferentCNN().to(device)
    maml = l2l.algorithms.MAML(model, lr=args.inner_lr, first_order=True, allow_unused=True)
    
    # Create datasets
    train_dataset = SameDifferentDataset(
        data_dir=args.data_dir,
        tasks=ALL_PB_TASKS,
        split='train',
        support_sizes=VARIABLE_SUPPORT_SIZES
    )
    
    val_dataset = SameDifferentDataset(
        data_dir=args.data_dir,
        tasks=ALL_PB_TASKS,
        split='val',
        support_sizes=VARIABLE_SUPPORT_SIZES
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.meta_batch_size, 
                             shuffle=True, collate_fn=collate_episodes)
    val_loader = DataLoader(val_dataset, batch_size=args.meta_batch_size, 
                           shuffle=False, collate_fn=collate_episodes)
    
    # Optimizer and loss
    optimizer = optim.Adam(maml.parameters(), lr=args.outer_lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training tracking
    data_points_seen = []
    val_accuracies = []
    total_data_points = 0
    
    # Training loop
    for epoch in range(args.epochs):
        maml.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"FOMAML Epoch {epoch+1}")):
            optimizer.zero_grad()
            
            batch_loss = 0.0
            batch_acc = 0.0
            
            for task_batch in batch:
                learner = maml.clone()
                task_loss, task_acc = fast_adapt(task_batch, learner, loss_fn, args.adaptation_steps, device)
                batch_loss += task_loss
                batch_acc += task_acc
                
                # Count data points (support + query for each task)
                data, _ = task_batch
                total_data_points += data.size(0)
            
            # Average over tasks in batch
            batch_loss /= len(batch)
            batch_acc /= len(batch)
            
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()
            num_batches += 1
            
            # Validate every few batches
            if batch_idx % args.val_frequency == 0:
                val_loss, val_acc = validate_meta_model(maml, val_loader, device, args.adaptation_steps, loss_fn)
                data_points_seen.append(total_data_points)
                val_accuracies.append(val_acc * 100)  # Convert to percentage
                print(f"  Batch {batch_idx}: Data points seen: {total_data_points}, Val Acc: {val_acc*100:.2f}%")
        
        # End of epoch validation
        val_loss, val_acc = validate_meta_model(maml, val_loader, device, args.adaptation_steps, loss_fn)
        data_points_seen.append(total_data_points)
        val_accuracies.append(val_acc * 100)
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = epoch_acc / num_batches if num_batches > 0 else 0.0
        
        print(f"FOMAML Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    
    # Save results
    results = {
        'method': 'FOMAML',
        'data_points_seen': data_points_seen,
        'val_accuracies': val_accuracies,
        'total_data_points': total_data_points
    }
    
    with open(os.path.join(save_dir, 'fomaml_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def train_second_order_maml(args, device, save_dir):
    """Train Second-Order MAML."""
    print("Training Second-Order MAML...")
    
    # Create model and MAML wrapper
    model = SameDifferentCNN().to(device)
    maml = l2l.algorithms.MAML(model, lr=args.inner_lr, first_order=False, allow_unused=True)
    
    # Create datasets (same as FOMAML)
    train_dataset = SameDifferentDataset(
        data_dir=args.data_dir,
        tasks=ALL_PB_TASKS,
        split='train',
        support_sizes=VARIABLE_SUPPORT_SIZES
    )
    
    val_dataset = SameDifferentDataset(
        data_dir=args.data_dir,
        tasks=ALL_PB_TASKS,
        split='val',
        support_sizes=VARIABLE_SUPPORT_SIZES
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.meta_batch_size, 
                             shuffle=True, collate_fn=collate_episodes)
    val_loader = DataLoader(val_dataset, batch_size=args.meta_batch_size, 
                           shuffle=False, collate_fn=collate_episodes)
    
    # Optimizer and loss
    optimizer = optim.Adam(maml.parameters(), lr=args.outer_lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # Training tracking
    data_points_seen = []
    val_accuracies = []
    total_data_points = 0
    
    # Training loop (same structure as FOMAML)
    for epoch in range(args.epochs):
        maml.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Second-Order MAML Epoch {epoch+1}")):
            optimizer.zero_grad()
            
            batch_loss = 0.0
            batch_acc = 0.0
            
            for task_batch in batch:
                learner = maml.clone()
                task_loss, task_acc = fast_adapt(task_batch, learner, loss_fn, args.adaptation_steps, device)
                batch_loss += task_loss
                batch_acc += task_acc
                
                # Count data points
                data, _ = task_batch
                total_data_points += data.size(0)
            
            # Average over tasks in batch
            batch_loss /= len(batch)
            batch_acc /= len(batch)
            
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()
            num_batches += 1
            
            # Validate every few batches
            if batch_idx % args.val_frequency == 0:
                val_loss, val_acc = validate_meta_model(maml, val_loader, device, args.adaptation_steps, loss_fn)
                data_points_seen.append(total_data_points)
                val_accuracies.append(val_acc * 100)
                print(f"  Batch {batch_idx}: Data points seen: {total_data_points}, Val Acc: {val_acc*100:.2f}%")
        
        # End of epoch validation
        val_loss, val_acc = validate_meta_model(maml, val_loader, device, args.adaptation_steps, loss_fn)
        data_points_seen.append(total_data_points)
        val_accuracies.append(val_acc * 100)
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = epoch_acc / num_batches if num_batches > 0 else 0.0
        
        print(f"Second-Order MAML Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
    
    # Save results
    results = {
        'method': 'Second-Order MAML',
        'data_points_seen': data_points_seen,
        'val_accuracies': val_accuracies,
        'total_data_points': total_data_points
    }
    
    with open(os.path.join(save_dir, 'second_order_maml_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

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

def plot_comparison(results_list, save_dir):
    """Create comparison plot of sample efficiency."""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'red', 'green']
    linestyles = ['-', '--', '-.']
    
    for i, results in enumerate(results_list):
        plt.plot(results['data_points_seen'], results['val_accuracies'], 
                color=colors[i], linestyle=linestyles[i], linewidth=2, 
                label=results['method'], marker='o', markersize=4)
    
    plt.xlabel('Number of Data Points Seen', fontsize=14)
    plt.ylabel('Validation Accuracy (%)', fontsize=14)
    plt.title('Sample Efficiency Comparison: FOMAML vs Second-Order MAML vs Vanilla SGD\n(Conv6 Architecture)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(save_dir, 'sample_efficiency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'sample_efficiency_comparison.pdf'), bbox_inches='tight')
    
    print(f"Comparison plot saved to {os.path.join(save_dir, 'sample_efficiency_comparison.png')}")
    
    return plt

def main():
    parser = argparse.ArgumentParser(description='Sample Efficiency Comparison: FOMAML vs Second-Order MAML vs Vanilla SGD')
    
    # Data directories
    parser.add_argument('--data_dir', type=str, default='data/meta_h5/pb', 
                       help='Directory for HDF5 data (used for both meta-learning and vanilla SGD)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    
    # Meta-learning parameters
    parser.add_argument('--meta_batch_size', type=int, default=32, 
                       help='Meta batch size')
    parser.add_argument('--inner_lr', type=float, default=0.001, 
                       help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=0.0001, 
                       help='Outer loop learning rate')
    parser.add_argument('--adaptation_steps', type=int, default=5, 
                       help='Number of adaptation steps')
    
    # Vanilla SGD parameters
    parser.add_argument('--vanilla_batch_size', type=int, default=32, 
                       help='Vanilla SGD batch size')
    parser.add_argument('--vanilla_lr', type=float, default=1e-4, 
                       help='Vanilla SGD learning rate')
    
    # Validation and saving
    parser.add_argument('--val_frequency', type=int, default=10, 
                       help='Validation frequency (in batches)')
    parser.add_argument('--save_dir', type=str, default='results/sample_efficiency_comparison', 
                       help='Directory to save results')
    
    # Method selection
    parser.add_argument('--methods', nargs='+', default=['fomaml', 'second_order', 'vanilla'],
                       choices=['fomaml', 'second_order', 'vanilla'],
                       help='Methods to run')
    
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
    
    # Run experiments
    results_list = []
    
    if 'fomaml' in args.methods:
        fomaml_results = train_fomaml(args, device, save_dir)
        results_list.append(fomaml_results)
    
    if 'second_order' in args.methods:
        second_order_results = train_second_order_maml(args, device, save_dir)
        results_list.append(second_order_results)
    
    if 'vanilla' in args.methods:
        vanilla_results = train_vanilla_sgd(args, device, save_dir)
        results_list.append(vanilla_results)
    
    # Create comparison plot
    if len(results_list) > 1:
        plot_comparison(results_list, save_dir)
    
    # Save combined results
    combined_results = {
        'args': vars(args),
        'results': results_list,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_dir, 'combined_results.json'), 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"Sample efficiency comparison completed. Results saved to {save_dir}")

if __name__ == '__main__':
    main()
