#!/usr/bin/env python3
"""
Test Adaptation Efficiency of Pretrained Meta-Learning Models

This script loads pretrained meta-learning models and measures their adaptation efficiency
to test the hypothesis: Second-Order MAML adapts faster than First-Order MAML.

The script loads weights from: /scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6/seed_{42-46}/best_model.pt
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

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import required modules
import learn2learn as l2l
from meta_baseline.models.conv6lr import SameDifferentCNN
from meta_baseline.models.utils_meta import SameDifferentDataset, collate_episodes

# Minimal task set for efficiency testing
EFFICIENCY_TASKS = ['regular', 'lines', 'open']
SUPPORT_SIZES = [4, 6]

def accuracy(predictions, targets):
    """Calculate accuracy from predictions and targets."""
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def load_pretrained_model(model_path, device):
    """Load a pretrained meta-learning model from checkpoint."""
    print(f"Loading pretrained model from: {model_path}")
    
    # Create model
    model = SameDifferentCNN().to(device)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
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
            
        print(f"✅ Successfully loaded pretrained model")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model from {model_path}: {e}")
        return None

def fast_adapt(episode, learner, loss_fn, adaptation_steps, device):
    """Perform fast adaptation for a single episode."""
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

def measure_adaptation_efficiency(method_name, model, test_episodes, args, device):
    """Measure test-time adaptation efficiency for a pretrained model."""
    print(f"  ⚡ Measuring adaptation efficiency for {method_name}")
    
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
    
    print(f"    Testing on {len(test_episodes)} episodes...")
    
    all_accuracy_curves = []
    step_times = []
    
    for episode_idx, episode in enumerate(tqdm(test_episodes, desc=f"Testing {method_name}")):
        # Clone the model for this episode
        learner = maml.clone()
        
        # Measure adaptation time
        start_time = time.perf_counter()
        accuracies = fast_adapt(episode, learner, loss_fn, args.max_adaptation_steps, device)
        adaptation_time = time.perf_counter() - start_time
        
        all_accuracy_curves.append(accuracies)
        step_times.append(adaptation_time / args.max_adaptation_steps)  # Time per step
        adaptation_metrics['final_accuracies'].append(accuracies[-1])
        
        if episode_idx >= args.max_test_episodes - 1:
            break
    
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
    adaptation_metrics['final_acc_mean'] = np.mean(adaptation_metrics['final_accuracies']) * 100
    adaptation_metrics['final_acc_std'] = np.std(adaptation_metrics['final_accuracies']) * 100
    
    # Find steps to reach target accuracies
    for target_acc in args.target_accuracies:
        steps_to_target = None
        for step, acc in enumerate(avg_curve):
            if acc >= target_acc:
                steps_to_target = step
                break
        adaptation_metrics['time_to_target'][target_acc] = steps_to_target
    
    # Print summary
    print(f"    Average accuracy curve: {[f'{acc:.1f}' for acc in avg_curve]}")
    for target_acc in args.target_accuracies:
        steps = adaptation_metrics['time_to_target'][target_acc]
        if steps is not None:
            print(f"    Steps to {target_acc}%: {steps}")
        else:
            print(f"    Never reached {target_acc}%")
    
    return adaptation_metrics

def run_pretrained_adaptation_comparison(args):
    """Run adaptation efficiency comparison using pretrained models."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define seeds to test
    seeds = list(range(42, 47))  # 42, 43, 44, 45, 46
    print(f"Testing seeds: {seeds}")
    
    # Load test dataset
    print("\n📂 Loading test dataset...")
    test_dataset = SameDifferentDataset(
        args.data_dir, 
        EFFICIENCY_TASKS, 
        'test',
        support_sizes=SUPPORT_SIZES
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_episodes
    )
    
    # Collect test episodes
    test_episodes = []
    for batch in test_loader:
        # Convert collated batch back to individual episodes
        batch_size = len(batch['task'])
        for i in range(batch_size):
            episode = {
                'support_images': batch['support_images'][i],
                'support_labels': batch['support_labels'][i],
                'query_images': batch['query_images'][i],
                'query_labels': batch['query_labels'][i],
                'task': batch['task'][i],
                'support_size': batch['support_size'][i]
            }
            test_episodes.append(episode)
    
    print(f"Loaded {len(test_episodes)} test episodes")
    
    # Results storage
    all_results = {
        'fomaml': {'curves': [], 'metrics': []},
        'second_order': {'curves': [], 'metrics': []}
    }
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Test each seed
    for seed in seeds:
        model_path = f"{args.model_base_path}/seed_{seed}/best_model.pt"
        
        print(f"\n🔬 Testing Seed {seed}")
        print("=" * 60)
        
        # Load pretrained model
        pretrained_model = load_pretrained_model(model_path, device)
        if pretrained_model is None:
            print(f"❌ Skipping seed {seed} due to loading error")
            continue
        
        # Test both methods with the same pretrained weights
        for method in ['fomaml', 'second_order']:
            print(f"\n  📊 Testing {method.upper()} adaptation efficiency")
            
            # Clone the pretrained model for this method
            model_copy = SameDifferentCNN().to(device)
            model_copy.load_state_dict(pretrained_model.state_dict())
            
            # Measure adaptation efficiency
            metrics = measure_adaptation_efficiency(
                method, model_copy, test_episodes[:args.max_test_episodes], args, device
            )
            
            all_results[method]['curves'].append(metrics['avg_curve'])
            all_results[method]['metrics'].append(metrics)
            
            print(f"  ✅ Completed {method.upper()} testing for seed {seed}")
    
    return all_results

def create_adaptation_plots(results, save_dir):
    """Create adaptation efficiency comparison plots."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract data for plotting
    methods = ['fomaml', 'second_order']
    method_labels = ['FOMAML', 'SECOND-ORDER']
    colors = ['#2E86AB', '#A23B72']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Average adaptation curves across all seeds
    for i, (method, label, color) in enumerate(zip(methods, method_labels, colors)):
        if not results[method]['curves']:
            continue
            
        # Average across all seeds
        all_curves = np.array(results[method]['curves'])
        mean_curve = np.mean(all_curves, axis=0)
        std_curve = np.std(all_curves, axis=0)
        
        steps = np.arange(len(mean_curve))
        
        axes[0].plot(steps, mean_curve, label=label, color=color, linewidth=2.5, marker='o', markersize=4)
        axes[0].fill_between(steps, mean_curve - std_curve, mean_curve + std_curve, 
                           color=color, alpha=0.2)
    
    axes[0].axhline(y=70, color='red', linestyle='--', linewidth=2, label='70% Target')
    axes[0].set_xlabel('Adaptation Steps', fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0].set_title('Test-Time Adaptation Efficiency', fontweight='bold', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([40, 85])
    
    # Plot 2: Steps to reach target accuracies
    target_accs = [55.0, 60.0, 65.0, 70.0]
    method_steps = []
    
    for method in methods:
        if not results[method]['metrics']:
            method_steps.append([len(target_accs)] * len(target_accs))  # Max steps if no data
            continue
            
        steps_list = []
        for target_acc in target_accs:
            # Average steps across seeds
            steps_for_target = []
            for metrics in results[method]['metrics']:
                steps = metrics['time_to_target'].get(target_acc, None)
                if steps is not None:
                    steps_for_target.append(steps)
                else:
                    steps_for_target.append(20)  # Max steps if not reached
            steps_list.append(np.mean(steps_for_target) if steps_for_target else 20)
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
    plt.savefig(save_dir / 'pretrained_adaptation_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Plots saved to {save_dir}")

def save_results(results, save_dir):
    """Save detailed results to JSON."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for method, data in results.items():
        json_results[method] = {
            'curves': [curve.tolist() if isinstance(curve, np.ndarray) else curve 
                      for curve in data['curves']],
            'metrics': []
        }
        
        for metrics in data['metrics']:
            json_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    json_metrics[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    json_metrics[key] = [arr.tolist() for arr in value]
                else:
                    json_metrics[key] = value
            json_results[method]['metrics'].append(json_metrics)
    
    # Save to JSON
    results_path = save_dir / 'pretrained_adaptation_results.json'
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Create summary report
    summary_path = save_dir / 'adaptation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("PRETRAINED MODEL ADAPTATION EFFICIENCY SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        for method in results.keys():
            method_label = method.replace('_', '-').upper()
            f.write(f"{method_label} RESULTS (averaged across seeds):\n")
            f.write("-" * 40 + "\n")
            
            if results[method]['metrics']:
                # Average metrics across seeds
                final_accs = [m['final_acc_mean'] for m in results[method]['metrics']]
                step_times = [m['avg_step_time'] for m in results[method]['metrics']]
                
                f.write(f"Final accuracy: {np.mean(final_accs):.2f}% ± {np.std(final_accs):.2f}%\n")
                f.write(f"Avg step time: {np.mean(step_times):.4f}s\n")
                
                for target_acc in [55.0, 60.0, 65.0, 70.0]:
                    steps_list = []
                    for metrics in results[method]['metrics']:
                        steps = metrics['time_to_target'].get(target_acc, None)
                        if steps is not None:
                            steps_list.append(steps)
                    
                    if steps_list:
                        f.write(f"Steps to {target_acc}%: {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}\n")
                    else:
                        f.write(f"Steps to {target_acc}%: Not reached\n")
            
            f.write("\n")
    
    print(f"📁 Results saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Test Adaptation Efficiency of Pretrained Meta-Learning Models')
    
    # Data and model paths
    parser.add_argument('--data_dir', type=str, default='data/meta_h5/pb',
                       help='Directory for HDF5 data')
    parser.add_argument('--model_base_path', type=str, 
                       default='/scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6',
                       help='Base path to pretrained model directories')
    parser.add_argument('--save_dir', type=str, default='results/pretrained_adaptation_efficiency',
                       help='Directory to save results')
    
    # Test parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for data loading')
    parser.add_argument('--inner_lr', type=float, default=0.01,
                       help='Inner loop learning rate for adaptation')
    parser.add_argument('--max_test_episodes', type=int, default=100,
                       help='Number of test episodes per seed')
    parser.add_argument('--max_adaptation_steps', type=int, default=20,
                       help='Maximum adaptation steps to test')
    parser.add_argument('--target_accuracies', nargs='+', type=float, 
                       default=[55.0, 60.0, 65.0, 70.0],
                       help='Target accuracies to measure steps needed')
    
    args = parser.parse_args()
    
    print("🔬 PRETRAINED MODEL ADAPTATION EFFICIENCY TEST")
    print("=" * 70)
    print(f"Testing hypothesis: Second-Order MAML adapts faster than First-Order MAML")
    print(f"Using pretrained models from: {args.model_base_path}")
    print(f"Max adaptation steps: {args.max_adaptation_steps}")
    print(f"Test episodes per seed: {args.max_test_episodes}")
    print("=" * 70)
    
    # Run comparison
    results = run_pretrained_adaptation_comparison(args)
    
    # Create visualizations
    create_adaptation_plots(results, args.save_dir)
    
    # Save results
    save_results(results, args.save_dir)
    
    print("\n🎉 PRETRAINED ADAPTATION EFFICIENCY TEST COMPLETED!")
    print("=" * 70)
    
    # Print quick summary
    for method in results.keys():
        method_label = method.replace('_', '-').upper()
        if results[method]['metrics']:
            final_accs = [m['final_acc_mean'] for m in results[method]['metrics']]
            step_times = [m['avg_step_time'] for m in results[method]['metrics']]
            
            print(f"{method_label}:")
            print(f"  Final accuracy: {np.mean(final_accs):.1f}% ± {np.std(final_accs):.1f}%")
            print(f"  Avg step time: {np.mean(step_times):.4f}s")
            
            # Check if reaches 70%
            reaches_70 = []
            for metrics in results[method]['metrics']:
                steps = metrics['time_to_target'].get(70.0, None)
                if steps is not None:
                    reaches_70.append(steps)
            
            if reaches_70:
                print(f"  Steps to 70%: {np.mean(reaches_70):.1f} ± {np.std(reaches_70):.1f}")
            else:
                print(f"  Steps to 70%: Not reached")
        print()
    
    print(f"📊 Full results and plots saved to: {args.save_dir}")

if __name__ == '__main__':
    main() 