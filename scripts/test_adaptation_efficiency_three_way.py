#!/usr/bin/env python3
"""
Three-Way Adaptation Efficiency Comparison: FOMAML vs Second-Order MAML vs Vanilla SGD

This script compares adaptation efficiency of three methods on naturalistic dataset:
1. First-Order MAML (using pretrained meta-learning weights)
2. Second-Order MAML (using pretrained meta-learning weights) 
3. Vanilla SGD (using pretrained vanilla weights)

Meta-learning models: /scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6/seed_{42-46}/best_model.pt
Vanilla SGD models: /scratch/gpfs/mg7411/samedifferent/results/pb_baselines_vanilla_final/all_tasks/conv6/test_regular/seed_{47-51}/best_model.pt
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
from pathlib import Path
from torch.utils.data import DataLoader
from collections import OrderedDict
import h5py

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import required modules
import learn2learn as l2l
from meta_baseline.models.conv6lr import SameDifferentCNN

def accuracy(predictions, targets):
    """Calculate accuracy from predictions and targets."""
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def load_pretrained_model(model_path, device):
    """Load a pretrained model from checkpoint."""
    print(f"Loading model from: {model_path}")
    
    # Create model
    model = SameDifferentCNN().to(device)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
        
        # Remove 'module.' prefix if exists (from DataParallel)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
        
        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
            
        print(f"✅ Successfully loaded model")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model from {model_path}: {e}")
        return None

class NaturalisticDataset(torch.utils.data.Dataset):
    """Dataset for naturalistic data adaptation testing."""
    
    def __init__(self, h5_file_path, max_episodes=None):
        self.h5_file_path = h5_file_path
        self.episodes = []
        
        print(f"Loading naturalistic data from {h5_file_path}")
        
        # Load episode metadata
        with h5py.File(h5_file_path, 'r') as f:
            episode_keys = [k for k in f.keys() if k.startswith('episode_')]
            
            if max_episodes:
                episode_keys = episode_keys[:max_episodes]
            
            for episode_key in episode_keys:
                episode_group = f[episode_key]
                
                # Check if this episode has the required structure
                if 'support_images' in episode_group and 'query_images' in episode_group:
                    episode_info = {
                        'episode_key': episode_key,
                        'support_size': episode_group['support_images'].shape[0],
                        'query_size': episode_group['query_images'].shape[0]
                    }
                    self.episodes.append(episode_info)
        
        print(f"Loaded {len(self.episodes)} naturalistic episodes")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        episode_info = self.episodes[idx]
        
        with h5py.File(self.h5_file_path, 'r') as f:
            episode_group = f[episode_info['episode_key']]
            
            # Load images and labels
            support_images = torch.from_numpy(episode_group['support_images'][:]).float() / 255.0
            support_labels = torch.from_numpy(episode_group['support_labels'][:]).long()
            query_images = torch.from_numpy(episode_group['query_images'][:]).float() / 255.0
            query_labels = torch.from_numpy(episode_group['query_labels'][:]).long()
            
            # Convert from HWC to CHW if necessary
            if support_images.dim() == 4 and support_images.shape[-1] == 3:
                support_images = support_images.permute(0, 3, 1, 2)
            if query_images.dim() == 4 and query_images.shape[-1] == 3:
                query_images = query_images.permute(0, 3, 1, 2)
            
            return {
                'support_images': support_images,
                'support_labels': support_labels.squeeze(),
                'query_images': query_images,
                'query_labels': query_labels.squeeze()
            }

def fast_adapt_meta(episode, learner, loss_fn, adaptation_steps, device):
    """Perform fast adaptation for meta-learning methods (FOMAML/Second-Order)."""
    support_data = episode['support_images'].to(device)
    support_labels = episode['support_labels'].to(device)
    query_data = episode['query_images'].to(device)
    query_labels = episode['query_labels'].to(device)

    # Track accuracy at each adaptation step
    accuracies = []
    
    # Step 0: Before any adaptation
    with torch.no_grad():
        query_preds = learner(query_data)
        query_acc = accuracy(query_preds, query_labels)
        accuracies.append(query_acc.item())

    # Adaptation steps
    for step in range(adaptation_steps):
        support_preds = learner(support_data)
        support_loss = loss_fn(support_preds, support_labels)
        learner.adapt(support_loss)
        
        # Evaluate after this adaptation step
        with torch.no_grad():
            query_preds = learner(query_data)
            query_acc = accuracy(query_preds, query_labels)
            accuracies.append(query_acc.item())

    return accuracies

def fast_adapt_vanilla(episode, model, optimizer, loss_fn, adaptation_steps, device):
    """Perform fast adaptation for vanilla SGD method."""
    support_data = episode['support_images'].to(device)
    support_labels = episode['support_labels'].to(device)
    query_data = episode['query_images'].to(device)
    query_labels = episode['query_labels'].to(device)

    # Track accuracy at each adaptation step
    accuracies = []
    
    # Step 0: Before any adaptation
    model.eval()
    with torch.no_grad():
        query_preds = model(query_data)
        query_acc = accuracy(query_preds, query_labels)
        accuracies.append(query_acc.item())

    # Adaptation steps
    for step in range(adaptation_steps):
        model.train()
        optimizer.zero_grad()
        
        support_preds = model(support_data)
        support_loss = loss_fn(support_preds, support_labels)
        support_loss.backward()
        optimizer.step()
        
        # Evaluate after this adaptation step
        model.eval()
        with torch.no_grad():
            query_preds = model(query_data)
            query_acc = accuracy(query_preds, query_labels)
            accuracies.append(query_acc.item())

    return accuracies

def measure_meta_adaptation_efficiency(method_name, model, test_episodes, args, device):
    """Measure adaptation efficiency for meta-learning methods."""
    print(f"  ⚡ Measuring {method_name} adaptation efficiency")
    
    # Create MAML wrapper
    first_order = (method_name == 'fomaml')
    maml = l2l.algorithms.MAML(
        model, 
        lr=args.inner_lr, 
        first_order=first_order,
        allow_unused=True,
        allow_nograd=True
    )
    
    adaptation_metrics = {
        'accuracy_curves': [],
        'time_to_target': {},
        'step_times': [],
        'final_accuracies': []
    }
    
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"    Testing on {min(len(test_episodes), args.max_test_episodes)} episodes...")
    
    all_accuracy_curves = []
    step_times = []
    
    for episode_idx, episode in enumerate(tqdm(test_episodes, desc=f"Testing {method_name}")):
        if episode_idx >= args.max_test_episodes:
            break
            
        # Clone the model for this episode
        learner = maml.clone()
        
        # Measure adaptation time
        start_time = time.perf_counter()
        accuracies = fast_adapt_meta(episode, learner, loss_fn, args.max_adaptation_steps, device)
        adaptation_time = time.perf_counter() - start_time
        
        all_accuracy_curves.append(accuracies)
        step_times.append(adaptation_time / args.max_adaptation_steps)  # Time per step
        adaptation_metrics['final_accuracies'].append(accuracies[-1])
    
    # Compute average accuracy curve
    max_steps = max(len(curve) for curve in all_accuracy_curves)
    avg_curve = []
    std_curve = []
    
    for step in range(max_steps):
        step_accuracies = [curve[step] if step < len(curve) else curve[-1] 
                          for curve in all_accuracy_curves]
        avg_curve.append(np.mean(step_accuracies) * 100)  # Convert to percentage
        std_curve.append(np.std(step_accuracies) * 100)
    
    adaptation_metrics['avg_curve'] = avg_curve
    adaptation_metrics['std_curve'] = std_curve
    adaptation_metrics['avg_step_time'] = np.mean(step_times)
    
    # Calculate steps to reach target accuracies
    for target_acc in args.target_accuracies:
        steps_needed = None
        for step, acc in enumerate(avg_curve):
            if acc >= target_acc:
                steps_needed = step
                break
        adaptation_metrics['time_to_target'][target_acc] = steps_needed
    
    return adaptation_metrics

def measure_vanilla_adaptation_efficiency(model, test_episodes, args, device):
    """Measure adaptation efficiency for vanilla SGD method."""
    print(f"  ⚡ Measuring vanilla SGD adaptation efficiency")
    
    adaptation_metrics = {
        'accuracy_curves': [],
        'time_to_target': {},
        'step_times': [],
        'final_accuracies': []
    }
    
    loss_fn = nn.CrossEntropyLoss()
    
    print(f"    Testing on {min(len(test_episodes), args.max_test_episodes)} episodes...")
    
    all_accuracy_curves = []
    step_times = []
    
    for episode_idx, episode in enumerate(tqdm(test_episodes, desc="Testing vanilla SGD")):
        if episode_idx >= args.max_test_episodes:
            break
            
        # Create a fresh copy of the model for this episode
        episode_model = SameDifferentCNN().to(device)
        episode_model.load_state_dict(model.state_dict())
        
        # Create optimizer for this episode
        optimizer = optim.Adam(episode_model.parameters(), lr=args.vanilla_lr)
        
        # Measure adaptation time
        start_time = time.perf_counter()
        accuracies = fast_adapt_vanilla(episode, episode_model, optimizer, loss_fn, args.max_adaptation_steps, device)
        adaptation_time = time.perf_counter() - start_time
        
        all_accuracy_curves.append(accuracies)
        step_times.append(adaptation_time / args.max_adaptation_steps)  # Time per step
        adaptation_metrics['final_accuracies'].append(accuracies[-1])
    
    # Compute average accuracy curve
    max_steps = max(len(curve) for curve in all_accuracy_curves)
    avg_curve = []
    std_curve = []
    
    for step in range(max_steps):
        step_accuracies = [curve[step] if step < len(curve) else curve[-1] 
                          for curve in all_accuracy_curves]
        avg_curve.append(np.mean(step_accuracies) * 100)  # Convert to percentage
        std_curve.append(np.std(step_accuracies) * 100)
    
    adaptation_metrics['avg_curve'] = avg_curve
    adaptation_metrics['std_curve'] = std_curve
    adaptation_metrics['avg_step_time'] = np.mean(step_times)
    
    # Calculate steps to reach target accuracies
    for target_acc in args.target_accuracies:
        steps_needed = None
        for step, acc in enumerate(avg_curve):
            if acc >= target_acc:
                steps_needed = step
                break
        adaptation_metrics['time_to_target'][target_acc] = steps_needed
    
    return adaptation_metrics

def create_three_way_comparison_plots(results, save_dir, max_adaptation_steps):
    """Create bar charts comparing all three methods."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Colors for the three methods
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    method_labels = ['First-Order MAML', 'Second-Order MAML', 'Vanilla SGD']
    methods = ['fomaml', 'second_order', 'vanilla_sgd']
    
    # Plot 1: Final accuracy comparison
    ax = axes[0]
    final_accs = []
    final_stds = []
    
    for method in methods:
        if method in results:
            curves = results[method]['curves']
            if curves:
                final_accuracies = [curve[-1] for curve in curves]
                final_accs.append(np.mean(final_accuracies))
                final_stds.append(np.std(final_accuracies))
            else:
                final_accs.append(0)
                final_stds.append(0)
        else:
            final_accs.append(0)
            final_stds.append(0)
    
    bars = ax.bar(method_labels, final_accs, yerr=final_stds, 
                  color=colors, alpha=0.8, capsize=8, width=0.6)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, final_accs, final_stds)):
        height = bar.get_height()
        ax.annotate(f'{mean:.1f}±{std:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 8),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title('Final Adaptation Accuracy', fontweight='bold', fontsize=14, pad=20)
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Steps to reach 60% accuracy
    ax = axes[1]
    target_acc = 60.0
    steps_to_target = []
    
    for method in methods:
        if method in results:
            # Average steps across all seeds
            method_steps = []
            for metrics in results[method]['metrics']:
                steps = metrics['time_to_target'].get(target_acc, None)
                if steps is not None:
                    method_steps.append(steps)
                else:
                    method_steps.append(max_adaptation_steps)  # Max steps if not reached
            
            if method_steps:
                steps_to_target.append(np.mean(method_steps))
            else:
                steps_to_target.append(max_adaptation_steps)
        else:
            steps_to_target.append(max_adaptation_steps)
    
    bars = ax.bar(method_labels, steps_to_target, 
                  color=colors, alpha=0.8, width=0.6)
    
    # Add value labels on bars
    for i, (bar, steps) in enumerate(zip(bars, steps_to_target)):
        height = bar.get_height()
        if steps >= max_adaptation_steps:
            text = f'>{max_adaptation_steps-1}'
        else:
            text = f'{int(steps)}'
        ax.annotate(text,
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 8),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_title(f'Steps to Reach {target_acc}% Accuracy', fontweight='bold', fontsize=14, pad=20)
    ax.set_ylabel('Adaptation Steps', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at max steps
    ax.axhline(y=max_adaptation_steps, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Max Steps ({max_adaptation_steps})')
    ax.legend(loc='upper right')
    
    # Overall title
    fig.suptitle('Three-Way Adaptation Efficiency Comparison on Naturalistic Data', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # Save plots
    plot_path = save_dir / 'three_way_adaptation_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Three-way comparison plot saved to {plot_path}")
    
    pdf_path = save_dir / 'three_way_adaptation_comparison.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"📊 PDF version saved to {pdf_path}")
    
    plt.close()

def run_three_way_adaptation_comparison(args):
    """Run adaptation efficiency comparison for all three methods."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load naturalistic test dataset
    print("\n📂 Loading naturalistic test dataset...")
    test_dataset = NaturalisticDataset(args.naturalistic_data_path, max_episodes=args.max_test_episodes * 2)
    
    # Convert to list for easier handling
    test_episodes = []
    for i in range(len(test_dataset)):
        test_episodes.append(test_dataset[i])
    
    print(f"Loaded {len(test_episodes)} naturalistic episodes")
    
    # Results storage
    all_results = {
        'fomaml': {'curves': [], 'metrics': []},
        'second_order': {'curves': [], 'metrics': []},
        'vanilla_sgd': {'curves': [], 'metrics': []}
    }
    
    # Test meta-learning methods (seeds 42-46)
    meta_seeds = list(range(42, 47))
    print(f"\n🔬 Testing Meta-Learning Methods (Seeds {meta_seeds})")
    print("=" * 70)
    
    for seed in meta_seeds:
        model_path = f"{args.meta_model_base_path}/seed_{seed}/best_model.pt"
        
        print(f"\n🔬 Testing Seed {seed} on Naturalistic Data")
        print("=" * 70)
        
        # Load pretrained meta-learning model
        pretrained_model = load_pretrained_model(model_path, device)
        if pretrained_model is None:
            print(f"❌ Skipping meta seed {seed} due to loading error")
            continue
        
        print(f"✅ Successfully loaded pretrained model")
        
        # Test both meta-learning methods with the same pretrained weights
        for method in ['fomaml', 'second_order']:
            print(f"\n  📊 Testing {method.upper()} naturalistic adaptation")
            
            # Clone the pretrained model for this method
            model_copy = SameDifferentCNN().to(device)
            model_copy.load_state_dict(pretrained_model.state_dict())
            
            # Measure adaptation efficiency
            metrics = measure_meta_adaptation_efficiency(
                method, model_copy, test_episodes, args, device
            )
            
            all_results[method]['curves'].append(metrics['avg_curve'])
            all_results[method]['metrics'].append(metrics)
            
            print(f"  ✅ Completed {method.upper()} naturalistic testing for seed {seed}")
    
    # Test vanilla SGD method (seeds 47-51)
    vanilla_seeds = list(range(47, 52))
    print(f"\n🔬 Testing Vanilla SGD Method (Seeds {vanilla_seeds})")
    print("=" * 70)
    
    for seed in vanilla_seeds:
        model_path = f"{args.vanilla_model_base_path}/seed_{seed}/best_model.pt"
        
        print(f"\n🔬 Testing Vanilla SGD Seed {seed} on Naturalistic Data")
        print("=" * 70)
        
        # Load pretrained vanilla model
        pretrained_model = load_pretrained_model(model_path, device)
        if pretrained_model is None:
            print(f"❌ Skipping vanilla seed {seed} due to loading error")
            continue
        
        print(f"✅ Successfully loaded pretrained vanilla model")
        
        # Test vanilla SGD adaptation
        print(f"\n  📊 Testing VANILLA SGD naturalistic adaptation")
        metrics = measure_vanilla_adaptation_efficiency(
            pretrained_model, test_episodes, args, device
        )
        
        all_results['vanilla_sgd']['curves'].append(metrics['avg_curve'])
        all_results['vanilla_sgd']['metrics'].append(metrics)
        
        print(f"  ✅ Completed VANILLA SGD naturalistic testing for seed {seed}")
    
    return all_results

def save_results(results, save_dir):
    """Save detailed results to JSON."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for method, data in results.items():
        serializable_results[method] = {
            'curves': [curve.tolist() if hasattr(curve, 'tolist') else curve for curve in data['curves']],
            'metrics': []
        }
        
        for metrics in data['metrics']:
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_metrics[key] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in value.items()}
                else:
                    serializable_metrics[key] = value
            serializable_results[method]['metrics'].append(serializable_metrics)
    
    results_path = save_dir / 'three_way_adaptation_results.json'
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"📁 Detailed results saved to {results_path}")

def print_summary(results):
    """Print a summary of all results."""
    print("\n🎉 THREE-WAY ADAPTATION EFFICIENCY COMPARISON COMPLETED!")
    print("=" * 80)
    
    methods = ['fomaml', 'second_order', 'vanilla_sgd']
    method_names = ['First-Order MAML', 'Second-Order MAML', 'Vanilla SGD']
    
    for method, name in zip(methods, method_names):
        if method in results and results[method]['curves']:
            curves = results[method]['curves']
            final_accuracies = [curve[-1] for curve in curves]
            
            print(f"{name}:")
            print(f"  Final accuracy: {np.mean(final_accuracies):.1f}% ± {np.std(final_accuracies):.1f}%")
            
            # Average step times
            if results[method]['metrics']:
                step_times = [m['avg_step_time'] for m in results[method]['metrics']]
                print(f"  Avg step time: {np.mean(step_times):.4f}s")
            
            # Steps to 60%
            steps_60 = []
            for metrics in results[method]['metrics']:
                steps = metrics['time_to_target'].get(60.0, None)
                if steps is not None:
                    steps_60.append(steps)
            
            if steps_60:
                print(f"  Steps to 60%: {np.mean(steps_60):.1f} ± {np.std(steps_60):.1f}")
            else:
                print(f"  Steps to 60%: Never reached")
            
            print()

def main():
    parser = argparse.ArgumentParser(description='Three-Way Adaptation Efficiency Comparison')
    
    # Data paths
    parser.add_argument('--naturalistic_data_path', type=str, 
                       default='/scratch/gpfs/mg7411/samedifferent/data/naturalistic/test.h5',
                       help='Path to naturalistic test data')
    parser.add_argument('--meta_model_base_path', type=str, 
                       default='/scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6',
                       help='Base path to pretrained meta-learning models')
    parser.add_argument('--vanilla_model_base_path', type=str, 
                       default='/scratch/gpfs/mg7411/samedifferent/results/pb_baselines_vanilla_final/all_tasks/conv6/test_regular',
                       help='Base path to pretrained vanilla models')
    parser.add_argument('--save_dir', type=str, 
                       default='results/three_way_adaptation_efficiency',
                       help='Directory to save results')
    
    # Test parameters
    parser.add_argument('--inner_lr', type=float, default=0.01,
                       help='Inner loop learning rate for meta-learning methods')
    parser.add_argument('--vanilla_lr', type=float, default=0.001,
                       help='Learning rate for vanilla SGD adaptation')
    parser.add_argument('--max_test_episodes', type=int, default=100,
                       help='Number of test episodes per seed')
    parser.add_argument('--max_adaptation_steps', type=int, default=30,
                       help='Maximum adaptation steps to test')
    parser.add_argument('--target_accuracies', nargs='+', type=float, 
                       default=[55.0, 60.0, 65.0, 70.0],
                       help='Target accuracies to measure steps needed')
    
    args = parser.parse_args()
    
    print("🔬 THREE-WAY ADAPTATION EFFICIENCY COMPARISON")
    print("=" * 80)
    print(f"Testing adaptation speed on naturalistic data:")
    print(f"  • First-Order MAML (pretrained meta-learning)")
    print(f"  • Second-Order MAML (pretrained meta-learning)")
    print(f"  • Vanilla SGD (pretrained vanilla)")
    print(f"Meta models: {args.meta_model_base_path}")
    print(f"Vanilla models: {args.vanilla_model_base_path}")
    print(f"Naturalistic data: {args.naturalistic_data_path}")
    print(f"Max adaptation steps: {args.max_adaptation_steps}")
    print("=" * 80)
    
    # Run comparison
    results = run_three_way_adaptation_comparison(args)
    
    # Create visualizations
    create_three_way_comparison_plots(results, args.save_dir, args.max_adaptation_steps)
    
    # Save detailed results
    save_results(results, args.save_dir)
    
    # Print summary
    print_summary(results)
    
    print(f"\n📊 All results and plots saved to: {args.save_dir}")

if __name__ == '__main__':
    main()
