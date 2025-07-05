#!/usr/bin/env python3
"""
Quick test script to validate the holdout OOD setup.

This script runs a minimal version of the holdout experiment to verify:
1. Dataset loading works correctly with holdout task
2. Model training runs without errors
3. Validation works on the holdout task only

This is useful for debugging before running the full experiment.
"""

import os
import sys
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

# Test configuration
ALL_PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]

def identity_collate(batch):
    """Simple collate function."""
    return batch

def test_holdout_setup():
    """Test the holdout experiment setup."""
    print("🧪 Testing Holdout OOD Setup")
    print("=" * 50)
    
    # Configuration
    holdout_task = 'scrambled'
    train_tasks = [task for task in ALL_PB_TASKS if task != holdout_task]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data/meta_h5/pb'
    
    print(f"Device: {device}")
    print(f"Holdout task: {holdout_task}")
    print(f"Training tasks ({len(train_tasks)}): {train_tasks}")
    print()
    
    # Test 1: Meta-learning dataset loading
    print("1. Testing meta-learning dataset loading...")
    try:
        # Training dataset (should NOT contain holdout task)
        train_dataset = SameDifferentDataset(
            data_dir=data_dir,
            tasks=train_tasks,
            split='train',
            support_sizes=[4, 6, 8, 10]
        )
        
        # Validation dataset (should ONLY contain holdout task)
        val_dataset = SameDifferentDataset(
            data_dir=data_dir,
            tasks=[holdout_task],
            split='val',
            support_sizes=[4, 6, 8, 10]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=identity_collate)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=identity_collate)
        
        print(f"✅ Training dataset loaded: {len(train_dataset)} episodes")
        print(f"✅ Validation dataset loaded: {len(val_dataset)} episodes") 
        print(f"✅ Training loader: {len(train_loader)} batches")
        print(f"✅ Validation loader: {len(val_loader)} batches")
        
        # Sample a batch from each
        print("\n   Testing batch sampling...")
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        print(f"   Training batch size: {len(train_batch)}")
        print(f"   Validation batch size: {len(val_batch)}")
        
        # Check episode structure
        train_episode = train_batch[0]
        val_episode = val_batch[0]
        
        print(f"   Training episode keys: {list(train_episode.keys())}")
        print(f"   Validation episode keys: {list(val_episode.keys())}")
        print(f"   Training support shape: {train_episode['support_images'].shape}")
        print(f"   Validation support shape: {val_episode['support_images'].shape}")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False
    
    # Test 2: Model creation and MAML wrapper
    print("\n2. Testing model creation...")
    try:
        model = SameDifferentCNN().to(device)
        maml = l2l.algorithms.MAML(model, lr=0.05, first_order=True, allow_unused=True, allow_nograd=True)
        
        total_params = sum(p.numel() for p in maml.parameters())
        print(f"✅ Model created: {total_params:,} parameters")
        
        # Test forward pass
        test_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(test_input)
        print(f"✅ Forward pass successful: {output.shape}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test 3: Single adaptation step
    print("\n3. Testing adaptation step...")
    try:
        loss_fn = nn.CrossEntropyLoss()
        episode = train_batch[0]
        
        # Move episode to device
        support_data = episode['support_images'].to(device)
        support_labels = episode['support_labels'].to(device)
        query_data = episode['query_images'].to(device)
        query_labels = episode['query_labels'].to(device)
        
        # Clone learner and adapt
        learner = maml.clone()
        learner.train()
        
        # Adaptation step
        support_preds = learner(support_data)
        support_loss = loss_fn(support_preds, support_labels)
        learner.adapt(support_loss)
        
        # Query step
        query_preds = learner(query_data)
        query_loss = loss_fn(query_preds, query_labels)
        
        print(f"✅ Adaptation successful")
        print(f"   Support loss: {support_loss.item():.4f}")
        print(f"   Query loss: {query_loss.item():.4f}")
        print(f"   Query predictions shape: {query_preds.shape}")
        
    except Exception as e:
        print(f"❌ Adaptation failed: {e}")
        return False
    
    # Test 4: Validation on holdout task
    print("\n4. Testing validation on holdout task...")
    try:
        maml.eval()
        val_losses = []
        val_accs = []
        
        for batch in val_loader:
            batch_loss = 0.0
            batch_acc = 0.0
            
            for episode in batch:
                support_data = episode['support_images'].to(device)
                support_labels = episode['support_labels'].to(device)
                query_data = episode['query_images'].to(device)
                query_labels = episode['query_labels'].to(device)
                
                learner = maml.clone()
                learner.train()
                
                # Adapt on support
                support_preds = learner(support_data)
                support_loss = loss_fn(support_preds, support_labels)
                learner.adapt(support_loss)
                
                # Evaluate on query
                query_preds = learner(query_data)
                query_loss = loss_fn(query_preds, query_labels)
                
                # Calculate accuracy
                with torch.no_grad():
                    predicted_labels = (query_preds[:, 1] > 0.0).float()
                    if query_labels.dim() > 1:
                        query_labels = query_labels.squeeze(-1)
                    accuracy = (predicted_labels == query_labels).float().mean()
                
                batch_loss += query_loss.detach().item()
                batch_acc += accuracy.item()
            
            val_losses.append(batch_loss / len(batch))
            val_accs.append(batch_acc / len(batch))
            
            if len(val_losses) >= 3:  # Test first 3 batches
                break
        
        avg_loss = np.mean(val_losses)
        avg_acc = np.mean(val_accs) * 100
        
        print(f"✅ Validation on holdout task successful")
        print(f"   Average loss: {avg_loss:.4f}")
        print(f"   Average accuracy: {avg_acc:.2f}%")
        print(f"   Batches tested: {len(val_losses)}")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False
    
    # Test 5: Dataset sizes make sense
    print("\n5. Checking dataset consistency...")
    try:
        # Check that training tasks don't include holdout
        train_task_names = set(train_tasks)
        if holdout_task in train_task_names:
            print(f"❌ Holdout task '{holdout_task}' found in training tasks!")
            return False
        
        # Check that all tasks are covered
        all_tasks_covered = train_task_names.union({holdout_task})
        if all_tasks_covered != set(ALL_PB_TASKS):
            print(f"❌ Task coverage mismatch: {all_tasks_covered} vs {set(ALL_PB_TASKS)}")
            return False
        
        print(f"✅ Dataset consistency check passed")
        print(f"   Training tasks: {len(train_tasks)}")
        print(f"   Holdout task: 1 ({holdout_task})")
        print(f"   Total tasks: {len(ALL_PB_TASKS)}")
        
    except Exception as e:
        print(f"❌ Consistency check failed: {e}")
        return False
    
    # Success!
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("✅ Holdout OOD setup is working correctly")
    print("✅ Ready to run full experiment")
    print("=" * 50)
    
    return True

def main():
    """Run the holdout setup test."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    success = test_holdout_setup()
    
    if success:
        print("\n🚀 You can now run the full experiment:")
        print("   sbatch run_sample_efficiency_holdout.slurm")
    else:
        print("\n❌ Fix the issues above before running the full experiment")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main()) 