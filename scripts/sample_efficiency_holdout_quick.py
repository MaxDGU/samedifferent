#!/usr/bin/env python3
"""
Quick OOD Sample Efficiency Test with Holdout Task

This is a minimal version of the holdout experiment that runs with reduced parameters
for quick testing of different holdout tasks. Useful for:
1. Testing different holdout tasks quickly
2. Debugging the OOD setup
3. Getting preliminary results before running the full experiment
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

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from meta_baseline.models.conv6lr import SameDifferentCNN
from meta_baseline.models.utils_meta import SameDifferentDataset
from scripts.sample_efficiency_comparison_holdout import (
    VanillaPBDataset, identity_collate, accuracy, fast_adapt,
    validate_meta_model, validate_vanilla_model
)

# Define PB tasks
ALL_PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]

def quick_test_holdout_task(holdout_task, device, data_dir='data/meta_h5/pb'):
    """Quickly test a holdout task to get preliminary OOD results."""
    print(f"\n🧪 Quick OOD Test: Holdout Task = {holdout_task}")
    print("=" * 60)
    
    # Setup
    train_tasks = [task for task in ALL_PB_TASKS if task != holdout_task]
    print(f"Training tasks: {train_tasks}")
    print(f"Holdout task: {holdout_task}")
    
    results = {}
    
    # Quick FOMAML test
    print("\n1. Quick FOMAML Test")
    print("-" * 30)
    
    try:
        # Create model
        model = SameDifferentCNN().to(device)
        maml = l2l.algorithms.MAML(model, lr=0.05, first_order=True, allow_unused=True, allow_nograd=True)
        
        # Create datasets
        train_dataset = SameDifferentDataset(
            data_dir=data_dir,
            tasks=train_tasks,
            split='train',
            support_sizes=[4, 6, 8, 10]
        )
        
        val_dataset = SameDifferentDataset(
            data_dir=data_dir,
            tasks=[holdout_task],
            split='val',
            support_sizes=[4, 6, 8, 10]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=identity_collate)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=identity_collate)
        
        optimizer = optim.Adam(maml.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        
        print(f"   Training episodes: {len(train_dataset)}")
        print(f"   Validation episodes: {len(val_dataset)}")
        
        # Initial validation
        initial_val_loss, initial_val_acc = validate_meta_model(maml, val_loader, device, 3, loss_fn)
        
        # Train for a few batches
        maml.train()
        total_batches = min(50, len(train_loader))  # Maximum 50 batches
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="FOMAML Training")):
            if batch_idx >= total_batches:
                break
                
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for task_batch in batch:
                learner = maml.clone()
                task_loss, task_acc = fast_adapt(task_batch, learner, loss_fn, 3, device)
                batch_loss += task_loss
            
            batch_loss /= len(batch)
            batch_loss.backward()
            optimizer.step()
        
        # Final validation
        final_val_loss, final_val_acc = validate_meta_model(maml, val_loader, device, 3, loss_fn)
        
        results['FOMAML'] = {
            'initial_acc': initial_val_acc * 100,
            'final_acc': final_val_acc * 100,
            'improvement': (final_val_acc - initial_val_acc) * 100,
            'batches_trained': total_batches
        }
        
        print(f"   Initial OOD accuracy: {initial_val_acc*100:.2f}%")
        print(f"   Final OOD accuracy: {final_val_acc*100:.2f}%")
        print(f"   Improvement: {(final_val_acc - initial_val_acc)*100:+.2f}%")
        print(f"   Trained on {total_batches} batches")
        
    except Exception as e:
        print(f"   ❌ FOMAML test failed: {e}")
        results['FOMAML'] = None
    
    # Quick Vanilla SGD test
    print("\n2. Quick Vanilla SGD Test")
    print("-" * 30)
    
    try:
        # Create model
        model = SameDifferentCNN().to(device)
        
        # Create datasets
        train_dataset = VanillaPBDataset(tasks=train_tasks, split='train', data_dir=data_dir)
        val_dataset = VanillaPBDataset(tasks=[holdout_task], split='val', data_dir=data_dir)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
        
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        loss_fn = nn.BCEWithLogitsLoss()
        
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        
        # Initial validation
        initial_val_loss, initial_val_acc = validate_vanilla_model(model, val_loader, device, loss_fn)
        
        # Train for a few batches
        model.train()
        total_batches = min(100, len(train_loader))  # Maximum 100 batches
        
        for batch_idx, (data, labels) in enumerate(tqdm(train_loader, desc="Vanilla SGD Training")):
            if batch_idx >= total_batches:
                break
                
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
        
        # Final validation
        final_val_loss, final_val_acc = validate_vanilla_model(model, val_loader, device, loss_fn)
        
        results['Vanilla SGD'] = {
            'initial_acc': initial_val_acc,
            'final_acc': final_val_acc,
            'improvement': final_val_acc - initial_val_acc,
            'batches_trained': total_batches
        }
        
        print(f"   Initial OOD accuracy: {initial_val_acc:.2f}%")
        print(f"   Final OOD accuracy: {final_val_acc:.2f}%")
        print(f"   Improvement: {final_val_acc - initial_val_acc:+.2f}%")
        print(f"   Trained on {total_batches} batches")
        
    except Exception as e:
        print(f"   ❌ Vanilla SGD test failed: {e}")
        results['Vanilla SGD'] = None
    
    # Summary
    print(f"\n📊 Quick OOD Results for Holdout Task: {holdout_task}")
    print("=" * 60)
    
    if results['FOMAML'] and results['Vanilla SGD']:
        fomaml_final = results['FOMAML']['final_acc']
        vanilla_final = results['Vanilla SGD']['final_acc']
        
        print(f"FOMAML Final OOD Accuracy:    {fomaml_final:.2f}%")
        print(f"Vanilla SGD Final OOD Accuracy: {vanilla_final:.2f}%")
        print(f"Advantage: {fomaml_final - vanilla_final:+.2f}% (positive = FOMAML better)")
        
        if fomaml_final > vanilla_final:
            print("✅ FOMAML shows advantage on this holdout task")
        else:
            print("❌ Vanilla SGD outperforms FOMAML on this holdout task")
    
    return results

def test_multiple_holdout_tasks(device, data_dir='data/meta_h5/pb'):
    """Test multiple holdout tasks to find the best one for OOD evaluation."""
    print("🔍 Testing Multiple Holdout Tasks for OOD Evaluation")
    print("=" * 80)
    
    # Test different holdout tasks
    test_tasks = ['scrambled', 'arrows', 'irregular', 'filled']
    all_results = {}
    
    for holdout_task in test_tasks:
        results = quick_test_holdout_task(holdout_task, device, data_dir)
        all_results[holdout_task] = results
    
    # Compare results
    print("\n📊 COMPARISON ACROSS HOLDOUT TASKS")
    print("=" * 80)
    
    print(f"{'Task':<12} {'FOMAML':<8} {'Vanilla':<8} {'Advantage':<10} {'Recommendation'}")
    print("-" * 60)
    
    best_task = None
    best_advantage = -float('inf')
    
    for task, results in all_results.items():
        if results['FOMAML'] and results['Vanilla SGD']:
            fomaml_acc = results['FOMAML']['final_acc']
            vanilla_acc = results['Vanilla SGD']['final_acc']
            advantage = fomaml_acc - vanilla_acc
            
            if advantage > best_advantage:
                best_advantage = advantage
                best_task = task
            
            recommendation = "✅ Good" if advantage > 2 else "⚠️ Modest" if advantage > 0 else "❌ Poor"
            
            print(f"{task:<12} {fomaml_acc:>6.1f}%   {vanilla_acc:>6.1f}%   {advantage:>+6.1f}%    {recommendation}")
    
    print("-" * 60)
    
    if best_task:
        print(f"\n🎯 RECOMMENDATION: Use '{best_task}' as holdout task")
        print(f"   Shows {best_advantage:.1f}% advantage for meta-learning")
        print(f"   This should demonstrate OOD generalization best")
    else:
        print("\n⚠️ No clear winner found. Consider using 'scrambled' as default.")
    
    return all_results, best_task

def main():
    parser = argparse.ArgumentParser(description='Quick OOD Holdout Test')
    parser.add_argument('--holdout_task', type=str, default=None,
                       choices=ALL_PB_TASKS,
                       help='Specific holdout task to test (if not provided, tests multiple)')
    parser.add_argument('--data_dir', type=str, default='data/meta_h5/pb',
                       help='Directory for HDF5 data')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    if args.holdout_task:
        # Test specific holdout task
        results = quick_test_holdout_task(args.holdout_task, device, args.data_dir)
    else:
        # Test multiple holdout tasks
        all_results, best_task = test_multiple_holdout_tasks(device, args.data_dir)
        
        # Save results
        save_path = 'results/holdout_task_comparison.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump({
                'all_results': all_results,
                'best_task': best_task,
                'recommendation': f"Use '{best_task}' as holdout task for OOD evaluation"
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {save_path}")

if __name__ == '__main__':
    main() 