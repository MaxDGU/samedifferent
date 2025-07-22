#!/usr/bin/env python3
"""
Test Adaptation Efficiency on Naturalistic Dataset

This script loads pretrained meta-learning models (trained on PB tasks) and measures 
their adaptation efficiency on the naturalistic same-different dataset.

This tests TRUE cross-domain generalization:
- Models trained on synthetic PB patterns
- Tested on real naturalistic object images
- Measures adaptation speed to novel domain

This should show clearer differences between FOMAML and Second-Order MAML
since the task is much harder and requires more adaptation steps.
"""

import os
import sys
import torch
import torch.nn as nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from torchvision import transforms

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import required modules
import learn2learn as l2l
from meta_baseline.models.conv6lr import SameDifferentCNN

class NaturalisticMetaDataset(Dataset):
    """
    Dataset for loading naturalistic meta-learning episodes from HDF5 files.
    Based on the existing naturalistic dataset implementations.
    """
    def __init__(self, h5_path, max_episodes=None):
        self.h5_path = Path(h5_path)
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                               std=[0.229, 0.224, 0.225])
        ])
        
        self._file = None
        self.episode_keys = []
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        
        try:
            self._file = h5py.File(self.h5_path, 'r')
            self.episode_keys = sorted([k for k in self._file.keys() if k.startswith('episode_')])
            
            if not self.episode_keys:
                raise ValueError(f"No episode groups found in {self.h5_path}")
            
            # Limit number of episodes if specified
            if max_episodes is not None:
                self.episode_keys = self.episode_keys[:max_episodes]
                
            print(f"Loaded {len(self.episode_keys)} episodes from {self.h5_path.name}")
            
        except Exception as e:
            print(f"Error opening {self.h5_path}: {e}")
            if self._file:
                self._file.close()
            raise
    
    def __len__(self):
        return len(self.episode_keys)
    
    def __getitem__(self, idx):
        if not hasattr(self, '_file') or not self._file:
            self._file = h5py.File(self.h5_path, 'r')
        
        episode_key = self.episode_keys[idx]
        
        try:
            ep_group = self._file[episode_key]
            support_images = ep_group['support_images'][()]  # [S, H, W, C] uint8
            support_labels = ep_group['support_labels'][()]  # [S] int32
            query_images = ep_group['query_images'][()]      # [Q, H, W, C] uint8
            query_labels = ep_group['query_labels'][()]      # [Q] int32
        except KeyError as e:
            print(f"Error accessing episode {episode_key}: {e}")
            raise
        
        # Transform images
        support_tensors = []
        for img in support_images:
            support_tensors.append(self.transform(img))
        
        query_tensors = []
        for img in query_images:
            query_tensors.append(self.transform(img))
        
        return {
            'support_images': torch.stack(support_tensors),
            'support_labels': torch.from_numpy(support_labels).long(),
            'query_images': torch.stack(query_tensors), 
            'query_labels': torch.from_numpy(query_labels).long(),
            'task': 'naturalistic',  # Single task type
            'support_size': len(support_images)
        }
    
    def close(self):
        if hasattr(self, '_file') and self._file:
            self._file.close()
            self._file = None
    
    def __del__(self):
        self.close()

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

def fast_adapt_naturalistic(episode, learner, loss_fn, adaptation_steps, device):
    """Perform fast adaptation for a naturalistic episode."""
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

def measure_naturalistic_adaptation_efficiency(method_name, model, test_episodes, args, device):
    """Measure test-time adaptation efficiency on naturalistic data."""
    print(f"  ⚡ Measuring naturalistic adaptation efficiency for {method_name}")
    
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
    
    print(f"    Testing on {len(test_episodes)} naturalistic episodes...")
    
    all_accuracy_curves = []
    step_times = []
    
    for episode_idx, episode in enumerate(tqdm(test_episodes, desc=f"Testing {method_name} on naturalistic")):
        # Clone the model for this episode
        learner = maml.clone()
        
        # Measure adaptation time
        start_time = time.perf_counter()
        accuracies = fast_adapt_naturalistic(episode, learner, loss_fn, args.max_adaptation_steps, device)
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

def run_naturalistic_adaptation_comparison(args):
    """Run adaptation efficiency comparison on naturalistic dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define seeds to test
    seeds = list(range(42, 47))  # 42, 43, 44, 45, 46
    print(f"Testing seeds: {seeds}")
    
    # Load naturalistic test dataset
    print("\n📂 Loading naturalistic test dataset...")
    naturalistic_dataset = NaturalisticMetaDataset(
        args.naturalistic_data_path,
        max_episodes=args.max_episodes_total
    )
    
    # Convert to episodes (since it's already in episode format)
    test_episodes = []
    for i in range(len(naturalistic_dataset)):
        test_episodes.append(naturalistic_dataset[i])
    
    print(f"Loaded {len(test_episodes)} naturalistic episodes")
    
    # Results storage
    all_results = {
        'fomaml': {'curves': [], 'metrics': []},
        'second_order': {'curves': [], 'metrics': []}
    }
    
    # Test each seed
    for seed in seeds:
        model_path = f"{args.model_base_path}/seed_{seed}/best_model.pt"
        
        print(f"\n🔬 Testing Seed {seed} on Naturalistic Data")
        print("=" * 70)
        
        # Load pretrained model
        pretrained_model = load_pretrained_model(model_path, device)
        if pretrained_model is None:
            print(f"❌ Skipping seed {seed} due to loading error")
            continue
        
        # Test both methods with the same pretrained weights
        for method in ['fomaml', 'second_order']:
            print(f"\n  📊 Testing {method.upper()} naturalistic adaptation")
            
            # Clone the pretrained model for this method
            model_copy = SameDifferentCNN().to(device)
            model_copy.load_state_dict(pretrained_model.state_dict())
            
            # Measure adaptation efficiency
            metrics = measure_naturalistic_adaptation_efficiency(
                method, model_copy, test_episodes[:args.max_test_episodes], args, device
            )
            
            all_results[method]['curves'].append(metrics['avg_curve'])
            all_results[method]['metrics'].append(metrics)
            
            print(f"  ✅ Completed {method.upper()} naturalistic testing for seed {seed}")
    
    return all_results

def create_naturalistic_adaptation_plots(results, save_dir):
    """Create adaptation efficiency comparison plots for naturalistic data."""
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
    
    axes[0].axhline(y=60, color='red', linestyle='--', linewidth=2, label='60% Target')
    axes[0].axhline(y=70, color='darkred', linestyle='--', linewidth=2, label='70% Target')
    axes[0].set_xlabel('Adaptation Steps', fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0].set_title('Naturalistic Adaptation Efficiency\n(PB→Naturalistic Transfer)', fontweight='bold', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([40, 85])
    
    # Plot 2: Steps to reach target accuracies
    target_accs = [50.0, 55.0, 60.0, 65.0, 70.0]
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
                    steps_for_target.append(30)  # Max steps if not reached
            steps_list.append(np.mean(steps_for_target) if steps_for_target else 30)
        method_steps.append(steps_list)
    
    x = np.arange(len(target_accs))
    width = 0.35
    for i, (method, steps_list) in enumerate(zip(method_labels, method_steps)):
        axes[1].bar(x + i*width - width/2, steps_list, width, label=method, 
                   color=colors[i], alpha=0.8)
    
    axes[1].set_xlabel('Target Accuracy (%)', fontweight='bold')
    axes[1].set_ylabel('Steps Needed', fontweight='bold')
    axes[1].set_title('Steps to Reach Target Accuracy\n(Lower = Better)', fontweight='bold', fontsize=14)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{acc}%' for acc in target_accs])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'naturalistic_adaptation_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Naturalistic adaptation plots saved to {save_dir}")

def save_naturalistic_results(results, save_dir):
    """Save detailed naturalistic adaptation results to JSON."""
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
    results_path = save_dir / 'naturalistic_adaptation_results.json'
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Create summary report
    summary_path = save_dir / 'naturalistic_adaptation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("NATURALISTIC ADAPTATION EFFICIENCY SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write("Testing PB-trained models on naturalistic images\n")
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
                
                for target_acc in [50.0, 55.0, 60.0, 65.0, 70.0]:
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
    
    print(f"📁 Naturalistic adaptation results saved to {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Test Adaptation Efficiency on Naturalistic Dataset')
    
    # Data and model paths
    parser.add_argument('--naturalistic_data_path', type=str, 
                       default='/scratch/gpfs/mg7411/samedifferent/data/naturalistic/test.h5',
                       help='Path to naturalistic test.h5 file')
    parser.add_argument('--model_base_path', type=str, 
                       default='/scratch/gpfs/mg7411/samedifferent/results/meta_baselines/conv6',
                       help='Base path to pretrained model directories')
    parser.add_argument('--save_dir', type=str, default='results/naturalistic_adaptation_efficiency',
                       help='Directory to save results')
    
    # Test parameters
    parser.add_argument('--inner_lr', type=float, default=0.01,
                       help='Inner loop learning rate for adaptation')
    parser.add_argument('--max_test_episodes', type=int, default=100,
                       help='Number of test episodes per seed')
    parser.add_argument('--max_episodes_total', type=int, default=500,
                       help='Maximum episodes to load from H5 file')
    parser.add_argument('--max_adaptation_steps', type=int, default=30,
                       help='Maximum adaptation steps to test (higher for harder task)')
    parser.add_argument('--target_accuracies', nargs='+', type=float, 
                       default=[50.0, 55.0, 60.0, 65.0, 70.0],
                       help='Target accuracies to measure steps needed')
    
    args = parser.parse_args()
    
    print("🔬 NATURALISTIC ADAPTATION EFFICIENCY TEST")
    print("=" * 80)
    print(f"Testing PB-trained models on naturalistic images")
    print(f"Hypothesis: Second-Order MAML adapts faster to novel domains")
    print(f"Using pretrained models from: {args.model_base_path}")
    print(f"Naturalistic data: {args.naturalistic_data_path}")
    print(f"Max adaptation steps: {args.max_adaptation_steps}")
    print(f"Test episodes per seed: {args.max_test_episodes}")
    print("=" * 80)
    
    # Run comparison
    results = run_naturalistic_adaptation_comparison(args)
    
    # Create visualizations
    create_naturalistic_adaptation_plots(results, args.save_dir)
    
    # Save results
    save_naturalistic_results(results, args.save_dir)
    
    print("\n🎉 NATURALISTIC ADAPTATION EFFICIENCY TEST COMPLETED!")
    print("=" * 80)
    
    # Print quick summary
    for method in results.keys():
        method_label = method.replace('_', '-').upper()
        if results[method]['metrics']:
            final_accs = [m['final_acc_mean'] for m in results[method]['metrics']]
            step_times = [m['avg_step_time'] for m in results[method]['metrics']]
            
            print(f"{method_label}:")
            print(f"  Final accuracy: {np.mean(final_accs):.1f}% ± {np.std(final_accs):.1f}%")
            print(f"  Avg step time: {np.mean(step_times):.4f}s")
            
            # Check if reaches 60%
            reaches_60 = []
            for metrics in results[method]['metrics']:
                steps = metrics['time_to_target'].get(60.0, None)
                if steps is not None:
                    reaches_60.append(steps)
            
            if reaches_60:
                print(f"  Steps to 60%: {np.mean(reaches_60):.1f} ± {np.std(reaches_60):.1f}")
            else:
                print(f"  Steps to 60%: Not reached")
        print()
    
    print(f"📊 Full results and plots saved to: {args.save_dir}")

if __name__ == '__main__':
    main()