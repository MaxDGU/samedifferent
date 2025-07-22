#!/usr/bin/env python3
"""
Computational Efficiency Comparison: FOMAML vs Second-Order MAML

This script measures the training-time vs test-time efficiency trade-off between
First-Order MAML and Second-Order MAML to test the hypothesis:
- FOMAML: More train-time efficient, less test-time efficient  
- Second-Order MAML: Less train-time efficient, more test-time efficient

The experiment tracks:
1. Training efficiency: time per batch, memory usage, forward/backward pass times
2. Test-time adaptation efficiency: accuracy vs adaptation steps, steps to target accuracy
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
import time
from datetime import datetime
from tqdm import tqdm
import psutil
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import required modules
import learn2learn as l2l
from meta_baseline.models.conv6lr import SameDifferentCNN
from meta_baseline.models.utils_meta import SameDifferentDataset

# Minimal task set for efficiency testing
EFFICIENCY_TASKS = ['regular', 'lines', 'open']
SUPPORT_SIZES = [4, 6]
QUERY_SIZE = 2

def identity_collate(batch):
    """Simple collate function that returns the batch unchanged."""
    return batch

def accuracy(predictions, targets):
    """Calculate accuracy from predictions and targets."""
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def get_memory_usage():
    """Get current memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    else:
        process = psutil.Process()
        return process.memory_info().rss / 1024**2

def reset_memory_stats():
    """Reset memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def get_peak_memory():
    """Get peak memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    else:
        return get_memory_usage()

def fast_adapt(episode, learner, loss_fn, adaptation_steps, device):
    """Perform fast adaptation for a single episode."""
    support_data = episode['support_images'].to(device)
    support_labels = episode['support_labels'].to(device)
    query_data = episode['query_images'].to(device)
    query_labels = episode['query_labels'].to(device)

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

def measure_training_efficiency(method_name, maml, train_loader, optimizer, loss_fn, args, device):
    """Measure training computational efficiency."""
    print(f"  📊 Measuring training efficiency for {method_name}")
    
    training_metrics = {
        'batch_times': [],
        'memory_peaks': [],
        'forward_times': [],
        'backward_times': [],
        'optimization_times': [],
        'total_times': []
    }
    
    maml.train()
    batch_count = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training {method_name}")):
        if batch_count >= args.max_training_batches:
            break
            
        # Reset memory tracking
        reset_memory_stats()
        total_start = time.perf_counter()
        
        optimizer.zero_grad()
        
        # Forward pass timing
        forward_start = time.perf_counter()
        batch_loss = 0.0
        batch_acc = 0.0
        
        for task_batch in batch:
            learner = maml.clone()
            task_loss, task_acc = fast_adapt(task_batch, learner, loss_fn, args.training_adaptation_steps, device)
            batch_loss += task_loss
            batch_acc += task_acc
        
        batch_loss /= len(batch)
        batch_acc /= len(batch)
        forward_time = time.perf_counter() - forward_start
        
        # Backward pass timing
        backward_start = time.perf_counter()
        batch_loss.backward()
        backward_time = time.perf_counter() - backward_start
        
        # Optimization timing
        opt_start = time.perf_counter()
        optimizer.step()
        opt_time = time.perf_counter() - opt_start
        
        total_time = time.perf_counter() - total_start
        peak_memory = get_peak_memory()
        
        # Record metrics
        training_metrics['batch_times'].append(total_time)
        training_metrics['memory_peaks'].append(peak_memory)
        training_metrics['forward_times'].append(forward_time)
        training_metrics['backward_times'].append(backward_time)
        training_metrics['optimization_times'].append(opt_time)
        training_metrics['total_times'].append(total_time)
        
        batch_count += 1
    
    # Compute averages
    for key in training_metrics:
        if training_metrics[key]:
            avg_val = np.mean(training_metrics[key])
            std_val = np.std(training_metrics[key])
            print(f"    {key}: {avg_val:.4f} ± {std_val:.4f}")
    
    return training_metrics

def measure_adaptation_efficiency(method_name, maml, test_episodes, args, device):
    """Measure test-time adaptation efficiency."""
    print(f"  ⚡ Measuring adaptation efficiency for {method_name}")
    
    adaptation_metrics = {
        'accuracy_curves': [],
        'time_to_target': {},
        'step_times': [],
        'final_accuracies': []
    }
    
    loss_fn = nn.CrossEntropyLoss()
    episode_count = 0
    
    for episode in tqdm(test_episodes, desc=f"Testing {method_name}"):
        if episode_count >= args.max_test_episodes:
            break
            
        step_accuracies = []
        step_times = []
        
        learner = maml.clone()
        learner.eval()
        
        support_data = episode['support_images'].to(device)
        support_labels = episode['support_labels'].to(device)
        query_data = episode['query_images'].to(device)
        query_labels = episode['query_labels'].to(device)
        
        # Test accuracy at each adaptation step
        for step in range(args.max_adaptation_steps + 1):  # +1 for step 0 (no adaptation)
            step_start = time.perf_counter()
            
            if step > 0:  # Perform adaptation step
                learner.train()
                support_preds = learner(support_data)
                support_loss = loss_fn(support_preds, support_labels)
                learner.adapt(support_loss)
                learner.eval()
            
            # Evaluate on query set
            with torch.no_grad():
                query_preds = learner(query_data)
                query_acc = accuracy(query_preds, query_labels)
                step_accuracies.append(query_acc.item() * 100)
            
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)
        
        adaptation_metrics['accuracy_curves'].append(step_accuracies)
        adaptation_metrics['step_times'].extend(step_times)
        adaptation_metrics['final_accuracies'].append(step_accuracies[-1])
        
        episode_count += 1
    
    # Compute average adaptation curve
    avg_accuracy_curve = np.mean(adaptation_metrics['accuracy_curves'], axis=0)
    std_accuracy_curve = np.std(adaptation_metrics['accuracy_curves'], axis=0)
    
    print(f"    Average accuracy curve: {[f'{acc:.1f}' for acc in avg_accuracy_curve]}")
    
    # Find steps needed to reach target accuracies
    for target_acc in args.target_accuracies:
        steps_needed = None
        for step, acc in enumerate(avg_accuracy_curve):
            if acc >= target_acc:
                steps_needed = step
                break
        adaptation_metrics['time_to_target'][target_acc] = steps_needed
        if steps_needed is not None:
            print(f"    Steps to reach {target_acc}%: {steps_needed}")
        else:
            print(f"    Never reached {target_acc}%")
    
    # Store summary statistics
    adaptation_metrics['avg_curve'] = avg_accuracy_curve.tolist()
    adaptation_metrics['std_curve'] = std_accuracy_curve.tolist()
    adaptation_metrics['avg_step_time'] = np.mean(adaptation_metrics['step_times'])
    adaptation_metrics['final_acc_mean'] = np.mean(adaptation_metrics['final_accuracies'])
    adaptation_metrics['final_acc_std'] = np.std(adaptation_metrics['final_accuracies'])
    
    return adaptation_metrics

def run_computational_comparison(args):
    """Main comparison function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\n📂 Loading datasets...")
    train_dataset = SameDifferentDataset(
        data_dir=args.data_dir,
        tasks=EFFICIENCY_TASKS,
        split='train',
        support_sizes=SUPPORT_SIZES
    )
    
    test_dataset = SameDifferentDataset(
        data_dir=args.data_dir,
        tasks=EFFICIENCY_TASKS,
        split='test',
        support_sizes=SUPPORT_SIZES
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.meta_batch_size, 
                             shuffle=True, collate_fn=identity_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, 
                            shuffle=False, collate_fn=identity_collate)
    
    # Convert test loader to episodes list for easier handling
    test_episodes = []
    for batch in test_loader:
        test_episodes.extend(batch)
    
    print(f"Training episodes: {len(train_dataset)}")
    print(f"Test episodes: {len(test_episodes)}")
    
    results = {}
    loss_fn = nn.CrossEntropyLoss()
    
    for method in ['fomaml', 'second_order']:
        print(f"\n🔬 Testing {method.upper()} Computational Efficiency")
        print("=" * 60)
        
        # Create model and MAML wrapper
        model = SameDifferentCNN().to(device)
        first_order = (method == 'fomaml')
        maml = l2l.algorithms.MAML(
            model, 
            lr=args.inner_lr, 
            first_order=first_order,
            allow_unused=True,
            allow_nograd=True
        )
        
        optimizer = optim.Adam(maml.parameters(), lr=args.outer_lr)
        
        # Phase 1: Training efficiency
        print("Phase 1: Training Efficiency Measurement")
        training_metrics = measure_training_efficiency(
            method, maml, train_loader, optimizer, loss_fn, args, device
        )
        
        # Phase 2: Test-time adaptation efficiency
        print("Phase 2: Test-Time Adaptation Efficiency Measurement")
        adaptation_metrics = measure_adaptation_efficiency(
            method, maml, test_episodes, args, device
        )
        
        results[method] = {
            'training': training_metrics,
            'adaptation': adaptation_metrics
        }
        
        print(f"✅ Completed {method.upper()} efficiency measurement")
    
    return results

def create_efficiency_plots(results, save_dir):
    """Create comprehensive efficiency comparison plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Training Efficiency Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    methods = list(results.keys())
    method_labels = [m.replace('_', '-').upper() for m in methods]
    
    # Plot 1: Training time per batch
    batch_times = [np.mean(results[m]['training']['batch_times']) for m in methods]
    batch_stds = [np.std(results[m]['training']['batch_times']) for m in methods]
    axes[0,0].bar(method_labels, batch_times, yerr=batch_stds, capsize=5)
    axes[0,0].set_title('Training Time per Batch', fontweight='bold')
    axes[0,0].set_ylabel('Time (seconds)')
    
    # Plot 2: Memory usage
    memory_usage = [np.mean(results[m]['training']['memory_peaks']) for m in methods]
    memory_stds = [np.std(results[m]['training']['memory_peaks']) for m in methods]
    axes[0,1].bar(method_labels, memory_usage, yerr=memory_stds, capsize=5)
    axes[0,1].set_title('Peak Memory Usage', fontweight='bold')
    axes[0,1].set_ylabel('Memory (MB)')
    
    # Plot 3: Forward vs Backward time breakdown
    forward_times = [np.mean(results[m]['training']['forward_times']) for m in methods]
    backward_times = [np.mean(results[m]['training']['backward_times']) for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    axes[1,0].bar(x - width/2, forward_times, width, label='Forward', alpha=0.8)
    axes[1,0].bar(x + width/2, backward_times, width, label='Backward', alpha=0.8)
    axes[1,0].set_title('Forward vs Backward Pass Time', fontweight='bold')
    axes[1,0].set_ylabel('Time (seconds)')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(method_labels)
    axes[1,0].legend()
    
    # Plot 4: Efficiency ratio (backward/forward time)
    efficiency_ratio = [backward_times[i]/forward_times[i] if forward_times[i] > 0 else 0 
                       for i in range(len(methods))]
    axes[1,1].bar(method_labels, efficiency_ratio)
    axes[1,1].set_title('Backward/Forward Time Ratio', fontweight='bold')
    axes[1,1].set_ylabel('Ratio')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Test-Time Adaptation Efficiency Plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Accuracy vs Adaptation Steps
    colors = ['#1f77b4', '#ff7f0e']
    for i, method in enumerate(methods):
        adaptation_data = results[method]['adaptation']
        avg_curve = adaptation_data['avg_curve']
        std_curve = adaptation_data['std_curve']
        steps = range(len(avg_curve))
        
        axes[0].plot(steps, avg_curve, marker='o', label=method_labels[i], 
                    color=colors[i], linewidth=2, markersize=6)
        axes[0].fill_between(steps, 
                           np.array(avg_curve) - np.array(std_curve),
                           np.array(avg_curve) + np.array(std_curve),
                           alpha=0.2, color=colors[i])
    
    axes[0].axhline(y=70, color='red', linestyle='--', linewidth=2, label='70% Target')
    axes[0].set_xlabel('Adaptation Steps', fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0].set_title('Test-Time Adaptation Efficiency', fontweight='bold', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Steps to Reach Target Accuracy
    target_accs = [60, 65, 70]
    method_steps = []
    for method in methods:
        steps_list = []
        for acc in target_accs:
            steps = results[method]['adaptation']['time_to_target'].get(acc, None)
            steps_list.append(steps if steps is not None else len(results[method]['adaptation']['avg_curve']))
        method_steps.append(steps_list)
    
    x = np.arange(len(target_accs))
    width = 0.35
    for i, (method, steps_list) in enumerate(zip(method_labels, method_steps)):
        axes[1].bar(x + i*width - width/2, steps_list, width, label=method, 
                   color=colors[i], alpha=0.8)
    
    axes[1].set_xlabel('Target Accuracy (%)', fontweight='bold')
    axes[1].set_ylabel('Steps Needed', fontweight='bold')
    axes[1].set_title('Steps to Reach Target Accuracy', fontweight='bold', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{acc}%' for acc in target_accs])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'adaptation_efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to {save_dir}")

def save_results(results, save_dir):
    """Save detailed results to JSON."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for method, data in results.items():
        json_results[method] = {}
        for phase, metrics in data.items():
            json_results[method][phase] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    json_results[method][phase][key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    json_results[method][phase][key] = [arr.tolist() for arr in value]
                else:
                    json_results[method][phase][key] = value
    
    # Save to JSON
    results_path = save_dir / 'computational_efficiency_results.json'
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Create summary report
    summary_path = save_dir / 'efficiency_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("COMPUTATIONAL EFFICIENCY COMPARISON SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for method in results.keys():
            method_label = method.replace('_', '-').upper()
            f.write(f"{method_label} RESULTS:\n")
            f.write("-" * 30 + "\n")
            
            # Training efficiency
            training = results[method]['training']
            f.write(f"Training Efficiency:\n")
            f.write(f"  Avg batch time: {np.mean(training['batch_times']):.4f}s\n")
            f.write(f"  Avg memory usage: {np.mean(training['memory_peaks']):.2f}MB\n")
            f.write(f"  Forward time: {np.mean(training['forward_times']):.4f}s\n")
            f.write(f"  Backward time: {np.mean(training['backward_times']):.4f}s\n")
            f.write(f"  Backward/Forward ratio: {np.mean(training['backward_times'])/np.mean(training['forward_times']):.2f}\n")
            
            # Adaptation efficiency
            adaptation = results[method]['adaptation']
            f.write(f"Adaptation Efficiency:\n")
            f.write(f"  Final accuracy: {adaptation['final_acc_mean']:.2f}% ± {adaptation['final_acc_std']:.2f}%\n")
            f.write(f"  Avg step time: {adaptation['avg_step_time']:.4f}s\n")
            
            for target_acc in [60, 65, 70]:
                steps = adaptation['time_to_target'].get(target_acc, None)
                if steps is not None:
                    f.write(f"  Steps to {target_acc}%: {steps}\n")
                else:
                    f.write(f"  Steps to {target_acc}%: Not reached\n")
            
            f.write("\n")
    
    print(f"📁 Results saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Computational Efficiency Comparison: FOMAML vs Second-Order MAML')
    
    # Data and experiment setup
    parser.add_argument('--data_dir', type=str, default='data/meta_h5/pb',
                       help='Directory for HDF5 data')
    parser.add_argument('--save_dir', type=str, default='results/computational_efficiency',
                       help='Directory to save results')
    
    # Training parameters
    parser.add_argument('--meta_batch_size', type=int, default=8,
                       help='Meta batch size for efficiency measurement')
    parser.add_argument('--inner_lr', type=float, default=0.05,
                       help='Inner loop learning rate')
    parser.add_argument('--outer_lr', type=float, default=0.001,
                       help='Outer loop learning rate')
    
    # Efficiency measurement parameters
    parser.add_argument('--max_training_batches', type=int, default=20,
                       help='Number of training batches to measure')
    parser.add_argument('--training_adaptation_steps', type=int, default=5,
                       help='Adaptation steps during training measurement')
    parser.add_argument('--max_test_episodes', type=int, default=50,
                       help='Number of test episodes for adaptation measurement')
    parser.add_argument('--max_adaptation_steps', type=int, default=15,
                       help='Maximum adaptation steps to test')
    parser.add_argument('--target_accuracies', nargs='+', type=float, 
                       default=[60.0, 65.0, 70.0],
                       help='Target accuracies to measure steps needed')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("🔬 COMPUTATIONAL EFFICIENCY COMPARISON")
    print("=" * 60)
    print(f"Testing hypothesis: FOMAML vs Second-Order MAML efficiency trade-off")
    print(f"Device: {device}")
    print(f"Training batches: {args.max_training_batches}")
    print(f"Test episodes: {args.max_test_episodes}")
    print(f"Max adaptation steps: {args.max_adaptation_steps}")
    print("=" * 60)
    
    # Run comparison
    results = run_computational_comparison(args)
    
    # Create visualizations
    create_efficiency_plots(results, args.save_dir)
    
    # Save results
    save_results(results, args.save_dir)
    
    print("\n🎉 COMPUTATIONAL EFFICIENCY COMPARISON COMPLETED!")
    print("=" * 60)
    
    # Print quick summary
    for method in results.keys():
        method_label = method.replace('_', '-').upper()
        training = results[method]['training']
        adaptation = results[method]['adaptation']
        
        print(f"{method_label}:")
        print(f"  Training time/batch: {np.mean(training['batch_times']):.3f}s")
        print(f"  Memory usage: {np.mean(training['memory_peaks']):.1f}MB")
        print(f"  Steps to 70%: {adaptation['time_to_target'].get(70.0, 'Not reached')}")
        print(f"  Final accuracy: {adaptation['final_acc_mean']:.1f}%")
        print()
    
    print(f"📊 Full results and plots saved to: {args.save_dir}")

if __name__ == '__main__':
    main() 