import os
import torch
import json
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import learn2learn as l2l
import torch.nn.functional as F
import copy
import gc
from conv6lr import SameDifferentCNN, SameDifferentDataset, collate_episodes
from torch.utils.data import DataLoader
import sys

# Define PB tasks
PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]

def accuracy(predictions, targets):
    """Calculate binary classification accuracy."""
    predicted_labels = (predictions[:, 1] > 0.0).float()
    return (predicted_labels == targets).float().mean()

def test_model(maml, test_loader, device, test_adaptation_steps):
    """Test model using per-layer learning rates."""
    try:
        maml.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        print(f"\nTesting with {test_adaptation_steps} adaptation steps")
        print(f"Dataset size: {len(test_loader.dataset)} episodes")
        
        for episodes in tqdm(test_loader, desc="Testing episodes"):
            batch_loss = 0
            batch_acc = 0
            
            for episode in episodes:
                try:
                    # Get support and query sets
                    support_images = episode['support_images'].to(device, non_blocking=True)
                    support_labels = episode['support_labels'].to(device, non_blocking=True)
                    query_images = episode['query_images'].to(device, non_blocking=True)
                    query_labels = episode['query_labels'].to(device, non_blocking=True)
                    
                    # Clone model for adaptation
                    learner = maml.clone()
                    
                    # Get per-layer learning rates
                    layer_lrs = learner.module.get_layer_lrs()
                    
                    # Adapt on support set
                    for _ in range(test_adaptation_steps):
                        support_preds = learner(support_images)
                        support_loss = F.binary_cross_entropy_with_logits(
                            support_preds[:, 1], support_labels.float())
                        
                        # Calculate gradients
                        grads = torch.autograd.grad(support_loss, learner.parameters(),
                                                  create_graph=True, allow_unused=True)
                        
                        # Apply per-layer learning rates
                        for (name, param), grad in zip(learner.named_parameters(), grads):
                            if grad is not None:
                                lr = layer_lrs.get(name, torch.tensor(0.01).to(device))
                                param.data = param.data - lr.abs() * grad
                    
                    # Evaluate on query set
                    with torch.no_grad():
                        query_preds = learner(query_images)
                        query_loss = F.binary_cross_entropy_with_logits(
                            query_preds[:, 1], query_labels.float())
                        query_acc = accuracy(query_preds, query_labels)
                        
                        batch_loss += query_loss.item()
                        batch_acc += query_acc.item()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"WARNING: GPU OOM error. Trying to recover...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            gc.collect()
                        continue
                    else:
                        print(f"ERROR: Runtime error in episode: {str(e)}")
                        continue
                except Exception as e:
                    print(f"ERROR: Unexpected error in episode: {str(e)}")
                    continue
            
            # Update totals if we processed any episodes successfully
            if batch_loss > 0:
                total_loss += batch_loss
                total_acc += batch_acc
                num_batches += 1
                print(f"Batch {num_batches} - Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}")
            
            # Clear GPU memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if num_batches == 0:
            raise RuntimeError("No batches were processed successfully")
        
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        return avg_loss, avg_acc
    
    except Exception as e:
        print(f"ERROR in test_model: {str(e)}")
        return float('inf'), 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True,
                      help='Directory containing experiment results')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing PB dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save test results')
    parser.add_argument('--test_adaptation_steps', type=int, default=15,
                      help='Number of adaptation steps during testing')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for testing')
    args = parser.parse_args()
    
    try:
        # Check for CUDA
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Running on CPU, but this will be slow and may not work correctly.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
        print(f"Using device: {device}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Find all task directories in the experiment directory
        task_dirs = [d for d in os.listdir(args.exp_dir) if os.path.isdir(os.path.join(args.exp_dir, d))]
        
        # Filter for holdout task directories
        holdout_dirs = [d for d in task_dirs if d.startswith('holdout_')]
        
        if not holdout_dirs:
            raise ValueError(f"No holdout task directories found in {args.exp_dir}")
        
        print(f"Found {len(holdout_dirs)} holdout task directories")
        
        all_results = {}
        
        # Process each holdout task
        for holdout_dir in holdout_dirs:
            holdout_task = holdout_dir.replace('holdout_', '')
            print(f"\nProcessing holdout task: {holdout_task}")
            
            task_path = os.path.join(args.exp_dir, holdout_dir)
            seed_dirs = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d)) and d.startswith('seed')]
            
            if not seed_dirs:
                print(f"No seed directories found for {holdout_task}, skipping")
                continue
            
            print(f"Found {len(seed_dirs)} seed directories")
            
            # Process each seed
            task_results = {}
            for seed_dir in seed_dirs:
                seed = seed_dir.replace('seed', '')
                print(f"\nProcessing seed: {seed}")
                
                # Find best model checkpoint
                model_path = os.path.join(task_path, seed_dir, 'best_model.pt')
                if not os.path.exists(model_path):
                    print(f"No best model found at {model_path}, skipping")
                    continue
                
                # Create model
                model = SameDifferentCNN().to(device)
                
                # Load checkpoint
                try:
                    checkpoint = torch.load(model_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model from {model_path}")
                except Exception as e:
                    print(f"Error loading model: {str(e)}")
                    continue
                
                # Create MAML wrapper
                maml = l2l.algorithms.MAML(
                    model,
                    lr=None,  # Use per-layer learning rates
                    first_order=False,
                    allow_unused=True,
                    allow_nograd=True
                )
                
                # Test on all PB tasks
                seed_results = {}
                for test_task in PB_TASKS:
                    print(f"Testing on task: {test_task}")
                    
                    # Create test dataset
                    test_dataset = SameDifferentDataset(
                        args.data_dir,
                        [test_task],
                        'test',
                        support_sizes=[10]  # Use fixed support size
                    )
                    
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                        collate_fn=collate_episodes
                    )
                    
                    # Test model
                    test_loss, test_acc = test_model(
                        maml,
                        test_loader,
                        device,
                        args.test_adaptation_steps
                    )
                    
                    seed_results[test_task] = {
                        'loss': test_loss,
                        'accuracy': test_acc
                    }
                    
                    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
                
                # Save seed results
                task_results[f"seed{seed}"] = seed_results
            
            # Calculate average results across seeds
            if task_results:
                avg_results = {}
                for test_task in PB_TASKS:
                    task_accs = [seed_results[test_task]['accuracy'] 
                                for seed_name, seed_results in task_results.items() 
                                if test_task in seed_results]
                    
                    if task_accs:
                        avg_acc = np.mean(task_accs)
                        std_acc = np.std(task_accs)
                        avg_results[test_task] = {
                            'mean_accuracy': float(avg_acc),
                            'std_accuracy': float(std_acc),
                            'num_seeds': len(task_accs)
                        }
                
                all_results[holdout_task] = {
                    'seed_results': task_results,
                    'average_results': avg_results
                }
                
                # Save task results
                task_output_path = os.path.join(args.output_dir, f"{holdout_task}_results.json")
                with open(task_output_path, 'w') as f:
                    json.dump(all_results[holdout_task], f, indent=4)
                print(f"Saved results to {task_output_path}")
        
        # Save all results
        all_output_path = os.path.join(args.output_dir, "all_results.json")
        with open(all_output_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"\nSaved all results to {all_output_path}")
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 