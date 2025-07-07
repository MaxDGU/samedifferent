#!/usr/bin/env python3
"""
Sample Efficiency Comparison with Holdout Task for OOD Generalization (Long Version with Checkpointing)

This script tests meta-learning's advantage on out-of-distribution tasks by:
1. Holding out one task from training
2. Training all methods on remaining tasks  
3. Testing validation accuracy only on the held-out task

This version includes checkpointing functionality to handle job time limits.
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import learn2learn as l2l
from tqdm import tqdm
import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from meta_baseline.models.conv6lr import SameDifferentCNN
from meta_baseline.models.utils_meta import SameDifferentDataset

# Define PB tasks
ALL_PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]

# Support and query sizes for meta-learning
VARIABLE_SUPPORT_SIZES = [4, 6, 8, 10]
FIXED_QUERY_SIZE = 2

def identity_collate(batch):
    """A simple collate_fn that returns the list of items unchanged."""
    return batch

def save_checkpoint(checkpoint_path, model, optimizer, epoch, results, args):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'results': results,
        'args': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load training checkpoint and return results."""
    if not os.path.exists(checkpoint_path):
        return None, 0
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    results = checkpoint['results']
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"Resuming from epoch {start_epoch}")
    print(f"Previous results: {len(results['val_accuracies'])} validation points")
    
    return results, start_epoch

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

def accuracy(predictions, targets):
    """Binary classification accuracy using raw logits."""
    with torch.no_grad():
        predicted_labels = (predictions[:, 1] > 0.0).float()
        if targets.dim() > 1:
            targets = targets.squeeze(-1)
        return (predicted_labels == targets).float().mean()

def fast_adapt(batch, learner, loss_fn, adaptation_steps, device):
    """Perform fast adaptation for a single episode."""
    if isinstance(batch, (list, tuple)) and len(batch) == 2 and torch.is_tensor(batch[0]):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        support_size = data.size(0) // 2
        support_data, query_data = data[:support_size], data[support_size:]
        support_labels, query_labels = labels[:support_size], labels[support_size:]
    else:
        if not isinstance(batch, dict):
            raise TypeError(f"fast_adapt expected episode dict or (data, labels) tuple, got type {type(batch)}")
        
        support_data = batch['support_images'].to(device)
        support_labels = batch['support_labels'].to(device)
        query_data = batch['query_images'].to(device)
        query_labels = batch['query_labels'].to(device)

    # Adaptation
    for _ in range(adaptation_steps):
        support_preds = learner(support_data)
        support_loss = loss_fn(support_preds, support_labels)
        learner.adapt(support_loss)

    # Evaluation
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

    for batch in val_loader:
        batch_loss = 0.0
        batch_acc = 0.0

        for episode in batch:
            learner = maml.clone()
            learner.train()
            ep_loss, ep_acc = fast_adapt(episode, learner, loss_fn, adaptation_steps, device)
            batch_loss += ep_loss.detach().item()
            batch_acc += ep_acc.item()

        batch_size = len(batch)
        total_loss += batch_loss / batch_size
        total_acc += batch_acc / batch_size
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0.0

    return avg_loss, avg_acc

def train_second_order_maml_holdout(args, device, save_dir, train_tasks, holdout_task):
    """Train Second-Order MAML with holdout task for OOD validation."""
    print(f"Training Second-Order MAML (Holdout: {holdout_task})...")
    print(f"Training tasks: {train_tasks}")
    print(f"Parameters: inner_lr={args.inner_lr}, outer_lr={args.outer_lr}, adaptation_steps={args.adaptation_steps}")
    
    # Create model and MAML wrapper
    model = SameDifferentCNN().to(device)
    maml = l2l.algorithms.MAML(
        model, 
        lr=args.inner_lr, 
        first_order=False, 
        allow_unused=True,
        allow_nograd=True
    )
    
    for param in maml.parameters():
        param.requires_grad = True
    
    total_params = sum(p.numel() for p in maml.parameters())
    trainable_params = sum(p.numel() for p in maml.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create datasets - train on training tasks, validate on holdout task only
    train_dataset = SameDifferentDataset(
        data_dir=args.data_dir,
        tasks=train_tasks,
        split='train',
        support_sizes=VARIABLE_SUPPORT_SIZES
    )
    
    val_dataset = SameDifferentDataset(
        data_dir=args.data_dir,
        tasks=[holdout_task],
        split='val',
        support_sizes=VARIABLE_SUPPORT_SIZES
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.meta_batch_size, 
                             shuffle=True, collate_fn=identity_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.meta_batch_size, 
                           shuffle=False, collate_fn=identity_collate)
    
    optimizer = optim.Adam(maml.parameters(), lr=args.outer_lr)
    loss_fn = nn.CrossEntropyLoss()
    
    # Setup checkpointing
    checkpoint_path = os.path.join(save_dir, f'second_order_maml_holdout_{holdout_task}_checkpoint.pth')
    
    # Try to load checkpoint
    checkpoint_results, start_epoch = load_checkpoint(checkpoint_path, maml, optimizer, device)
    
    if checkpoint_results is not None:
        # Resume from checkpoint
        data_points_seen = checkpoint_results['data_points_seen']
        val_accuracies = checkpoint_results['val_accuracies']
        total_data_points = checkpoint_results['total_data_points']
        print(f"Resuming from checkpoint with {len(val_accuracies)} validation points")
    else:
        # Start fresh
        data_points_seen = []
        val_accuracies = []
        total_data_points = 0
        start_epoch = 0
    
    print(f"Starting training from epoch {start_epoch} with {len(train_loader)} batches per epoch")
    print(f"OOD Validation on holdout task: {holdout_task}")
    
    for epoch in range(start_epoch, args.epochs):
        maml.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Second-Order MAML Epoch {epoch+1} (Holdout: {holdout_task})")
        
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            batch_loss = 0.0
            batch_acc = 0.0
            
            for task_batch in batch:
                learner = maml.clone()
                task_loss, task_acc = fast_adapt(task_batch, learner, loss_fn, args.adaptation_steps, device)
                batch_loss += task_loss
                batch_acc += task_acc
                
                # Count data points
                if isinstance(task_batch, dict):
                    total_data_points += task_batch['support_images'].size(0) + task_batch['query_images'].size(0)
                else:
                    data, _ = task_batch
                    total_data_points += data.size(0)
            
            batch_loss /= len(batch)
            batch_acc /= len(batch)
            
            # Gradient checks
            grad_norm = 0.0
            for param in maml.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # Validation checks
            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                print(f"WARNING: Invalid loss detected at epoch {epoch+1}, batch {batch_idx}: {batch_loss}")
                continue
            
            if grad_norm == 0.0:
                print(f"WARNING: Zero gradient norm at epoch {epoch+1}, batch {batch_idx}")
            elif grad_norm > 100.0:
                print(f"WARNING: Large gradient norm {grad_norm:.2f} at epoch {epoch+1}, batch {batch_idx}")
            
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{batch_loss.item():.4f}',
                'Acc': f'{batch_acc.item()*100:.1f}%',
                'Data': f'{total_data_points:,}'
            })
            
            # Validate every few batches
            if batch_idx % args.val_frequency == 0:
                print(f"\n--- Second-Order MAML OOD Validation at Batch {batch_idx} (Holdout: {holdout_task}) ---")
                val_loss, val_acc = validate_meta_model(maml, val_loader, device, args.test_adaptation_steps, loss_fn)
                data_points_seen.append(total_data_points)
                val_accuracies.append(val_acc * 100)
                
                current_train_acc = epoch_acc / max(1, batch_idx + 1) * 100
                print(f"  Data points seen: {total_data_points:,}")
                print(f"  Current train accuracy (in-distribution): {current_train_acc:.2f}%")
                print(f"  Validation accuracy (OOD - {holdout_task}): {val_acc*100:.2f}%")
                print(f"  Validation loss: {val_loss:.4f}")
                if len(val_accuracies) > 1:
                    acc_improvement = val_accuracies[-1] - val_accuracies[-2]
                    print(f"  OOD accuracy improvement: {acc_improvement:+.2f}%")
                print("-" * 50)
        
        # End of epoch validation
        val_loss, val_acc = validate_meta_model(maml, val_loader, device, args.test_adaptation_steps, loss_fn)
        data_points_seen.append(total_data_points)
        val_accuracies.append(val_acc * 100)
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = epoch_acc / num_batches if num_batches > 0 else 0.0
        
        print(f"Second-Order MAML Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc*100:.2f}%, "
              f"OOD Val Loss: {val_loss:.4f}, OOD Val Acc: {val_acc*100:.2f}%")
        
        # Save checkpoint every few epochs
        if (epoch + 1) % 2 == 0:
            current_results = {
                'method': 'Second-Order MAML',
                'holdout_task': holdout_task,
                'train_tasks': train_tasks,
                'data_points_seen': data_points_seen,
                'val_accuracies': val_accuracies,
                'total_data_points': total_data_points,
                'ood_evaluation': True,
                'config': {
                    'inner_lr': args.inner_lr,
                    'outer_lr': args.outer_lr,
                    'adaptation_steps': args.adaptation_steps,
                    'meta_batch_size': args.meta_batch_size
                }
            }
            save_checkpoint(checkpoint_path, maml, optimizer, epoch, current_results, args)
    
    # Training summary
    print(f"\n{'='*60}")
    print(f"Second-Order MAML OOD Training Summary (Holdout: {holdout_task})")
    print(f"{'='*60}")
    print(f"Training tasks: {train_tasks}")
    print(f"Holdout (OOD) task: {holdout_task}")
    print(f"Total data points processed: {total_data_points:,}")
    print(f"Total validation points: {len(val_accuracies)}")
    if val_accuracies:
        initial_acc = val_accuracies[0]
        final_acc = val_accuracies[-1]
        max_acc = max(val_accuracies)
        print(f"Initial OOD validation accuracy: {initial_acc:.2f}%")
        print(f"Final OOD validation accuracy: {final_acc:.2f}%")
        print(f"Maximum OOD validation accuracy: {max_acc:.2f}%")
        print(f"Total OOD improvement: {final_acc - initial_acc:+.2f}%")
        
        if final_acc - initial_acc < 1.0:
            print("⚠️  WARNING: Very little OOD learning progress detected!")
        if max_acc < 60.0:
            print("⚠️  WARNING: Low maximum OOD accuracy achieved!")
    print(f"{'='*60}\n")
    
    # Save final results
    results = {
        'method': 'Second-Order MAML',
        'holdout_task': holdout_task,
        'train_tasks': train_tasks,
        'data_points_seen': data_points_seen,
        'val_accuracies': val_accuracies,
        'total_data_points': total_data_points,
        'ood_evaluation': True,
        'config': {
            'inner_lr': args.inner_lr,
            'outer_lr': args.outer_lr,
            'adaptation_steps': args.adaptation_steps,
            'meta_batch_size': args.meta_batch_size
        }
    }
    
    with open(os.path.join(save_dir, f'second_order_maml_holdout_{holdout_task}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Clean up checkpoint file
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint file removed: {checkpoint_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='OOD Sample Efficiency Comparison with Holdout Task (With Checkpointing)')
    
    # Data directories
    parser.add_argument('--data_dir', type=str, default='data/meta_h5/pb', 
                       help='Directory for HDF5 data')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=25, 
                       help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    
    # Holdout experiment parameters
    parser.add_argument('--holdout_task', type=str, default='scrambled',
                       choices=ALL_PB_TASKS,
                       help='Task to hold out for OOD validation')
    
    # Meta-learning parameters
    parser.add_argument('--meta_batch_size', type=int, default=16, 
                       help='Meta batch size')
    parser.add_argument('--inner_lr', type=float, default=0.05, 
                       help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=0.001, 
                       help='Outer loop learning rate')
    parser.add_argument('--adaptation_steps', type=int, default=10, 
                       help='Number of adaptation steps for training')
    parser.add_argument('--test_adaptation_steps', type=int, default=15, 
                       help='Number of adaptation steps for testing/validation')
    
    # Validation and saving
    parser.add_argument('--val_frequency', type=int, default=1000, 
                       help='Validation frequency (in batches)')
    parser.add_argument('--save_dir', type=str, default='results/sample_efficiency_ood', 
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create holdout experiment setup
    holdout_task = args.holdout_task
    train_tasks = [task for task in ALL_PB_TASKS if task != holdout_task]
    
    print(f"\n{'='*80}")
    print(f"OUT-OF-DISTRIBUTION SAMPLE EFFICIENCY EXPERIMENT (WITH CHECKPOINTING)")
    print(f"{'='*80}")
    print(f"Holdout (OOD) task: {holdout_task}")
    print(f"Training tasks ({len(train_tasks)}): {train_tasks}")
    print(f"Method: Second-Order MAML Only")
    print(f"This tests meta-learning's advantage on novel task generalization!")
    print(f"{'='*80}\n")
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, f'holdout_{holdout_task}', f'seed_{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    
    # Run Second-Order MAML
    print("🚀 Starting Second-Order MAML OOD training...")
    second_order_results = train_second_order_maml_holdout(args, device, save_dir, train_tasks, holdout_task)
    
    print("\n" + "="*80)
    print("🎉 OOD SAMPLE EFFICIENCY EXPERIMENT COMPLETED!")
    print("="*80)
    print(f"Holdout task: {holdout_task}")
    print(f"Results saved to: {save_dir}")
    
    # Print final comparison
    if second_order_results['val_accuracies']:
        final_acc = second_order_results['val_accuracies'][-1]
        max_acc = max(second_order_results['val_accuracies'])
        data_points = second_order_results['total_data_points']
        print(f"\n📊 FINAL OOD PERFORMANCE:")
        print("-" * 50)
        print(f"{'Second-Order MAML':20} | Final: {final_acc:6.2f}% | Max: {max_acc:6.2f}% | Data: {data_points:,}")
        print("-" * 50)
    
    print("="*80)

if __name__ == '__main__':
    main() 