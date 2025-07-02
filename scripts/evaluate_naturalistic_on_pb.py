import os
import torch
import numpy as np
import argparse
import json
import pandas as pd
import sys
import time
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Path Setup ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from baselines.models.conv6 import SameDifferentCNN
from baselines.models.utils import SameDifferentDataset

def evaluate_model(model, data_loader, device, verbose=True, max_batches=None):
    """Evaluates the model on a given test dataset."""
    model.eval()
    model.to(device)
    
    total_correct = 0
    total_samples = 0
    
    num_batches_to_process = len(data_loader)
    if max_batches is not None and max_batches < num_batches_to_process:
        num_batches_to_process = max_batches
        if verbose:
            print(f"    Starting evaluation on a subset of {max_batches}/{len(data_loader)} batches...")
    elif verbose:
        print(f"    Starting evaluation with {len(data_loader)} batches...")
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="    Evaluating batches", total=num_batches_to_process, leave=False) if verbose else data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                if verbose:
                    print(f"    Stopping evaluation after {max_batches} batches as requested.")
                break

            # The SameDifferentDataset returns a dictionary with 'image' and 'label' keys
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            # The dataset loader might wrap batches in an extra dimension
            if images.dim() == 5: # (batch, num_images, channels, H, W)
                bs, n_imgs, c, h, w = images.shape
                images = images.view(bs * n_imgs, c, h, w)
                labels = labels.view(bs * n_imgs)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            
            # Print progress every 50 batches
            if verbose and batch_idx > 0 and batch_idx % 50 == 0:
                current_acc = (total_correct / total_samples) * 100
                elapsed = time.time() - start_time
                print(f"    Batch {batch_idx}/{num_batches_to_process}: Current accuracy: {current_acc:.2f}% ({total_correct}/{total_samples}) - {elapsed:.1f}s elapsed")
    
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"    Evaluation completed in {elapsed:.1f}s: {total_correct}/{total_samples} correct ({accuracy:.2f}%)")
    
    return accuracy

def main(args):
    """Main function to run the cross-domain evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Define Model and Data Paths ---
    naturalistic_seeds = [0, 1, 2, 3, 4] # Use the new seeds
    pb_tasks = [
        'original', 'filled', 'irregular', 'arrows', 'random_color', 
        'scrambled', 'wider_line', 'open', 'lines', 'regular'
    ]
    
    # Updated template for the new, nested path structure
    model_path_template = os.path.join(args.model_dir, 'seed_{seed}', 'conv6lr', 'seed_{seed}', 'conv6lr_best.pth')

    print(f"\nStarting cross-domain evaluation:")
    print(f"  - {len(naturalistic_seeds)} naturalistic models (seeds: {naturalistic_seeds})")
    print(f"  - {len(pb_tasks)} PB tasks: {pb_tasks}")
    print(f"  - Model path template: {model_path_template}")
    print(f"  - Data directory: {args.data_dir}")
    print(f"  - Batch size: {args.batch_size}")
    if args.max_batches:
        print(f"  - Evaluating on a subset of {args.max_batches} batches per task.")

    # --- Evaluation Loop ---
    all_results = {}
    total_evaluations = len(naturalistic_seeds) * len(pb_tasks)
    completed_evaluations = 0

    for seed_idx, seed in enumerate(naturalistic_seeds):
        print(f"\n{'='*60}")
        print(f"PROCESSING SEED {seed} ({seed_idx+1}/{len(naturalistic_seeds)})")
        print(f"{'='*60}")
        
        # --- Load Model ---
        model_path = model_path_template.format(seed=seed)
        print(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"WARNING: Model for seed {seed} not found at {model_path}. Skipping.")
            continue
            
        model = SameDifferentCNN()
        try:
            # Load checkpoint, handling potential DataParallel 'module.' prefix
            print("  Loading model state dict...")
            state_dict = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            print("  Model loaded successfully!")
        except Exception as e:
            print(f"ERROR loading model for seed {seed}: {e}. Skipping.")
            continue

        seed_results = {}
        for task_idx, task in enumerate(pb_tasks):
            print(f"\n  Task {task} ({task_idx+1}/{len(pb_tasks)}):")
            
            # --- Load PB Task Data ---
            try:
                print(f"    Loading dataset for task '{task}'...")
                # Correctly set the support sizes to match the available test data
                test_dataset = SameDifferentDataset(args.data_dir, task_names=[task], split='test', support_sizes=[4, 6, 8, 10])
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
                print(f"    Dataset loaded: {len(test_dataset)} samples, {len(test_loader)} batches")
            except ValueError as e:
                print(f"    WARNING: Could not load data for task '{task}'. Skipping. Error: {e}")
                continue

            # --- Evaluate and Store Accuracy ---
            print(f"    Evaluating model on task '{task}'...")
            start_time = time.time()
            accuracy = evaluate_model(model, test_loader, device, verbose=args.verbose, max_batches=args.max_batches)
            elapsed = time.time() - start_time
            
            seed_results[task] = accuracy
            completed_evaluations += 1
            
            print(f"  ✓ Seed {seed} | Task: {task:12} | Accuracy: {accuracy:.2f}% | Time: {elapsed:.1f}s")
            print(f"    Progress: {completed_evaluations}/{total_evaluations} evaluations completed ({100*completed_evaluations/total_evaluations:.1f}%)")

        all_results[f'seed_{seed}'] = seed_results

    # --- Save and Summarize Results ---
    if not all_results:
        print("\nNo models were evaluated. Exiting.")
        return

    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    # Save detailed results to JSON
    results_path = os.path.join(args.output_dir, 'naturalistic_on_pb_accuracies.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"Detailed results saved to {results_path}")

    # Create and print a summary DataFrame
    df = pd.DataFrame(all_results).T # Transpose to have seeds as rows
    
    # Calculate mean and std deviation for each task
    summary = df.agg(['mean', 'std'])
    
    # Calculate overall mean across all tasks and seeds
    overall_mean = df.to_numpy().mean()
    summary['Overall_Avg'] = [overall_mean, df.to_numpy().std()]
    
    print("\n--- Summary of Accuracies (%) ---")
    print(summary.round(2))
    
    # Save summary to CSV
    summary_path = os.path.join(args.output_dir, 'naturalistic_on_pb_summary.csv')
    summary.to_csv(summary_path)
    print(f"Summary saved to {summary_path}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate meta-trained naturalistic models on PB test sets.')
    parser.add_argument('--model_dir', type=str, 
                        default='/scratch/gpfs/mg7411/samedifferent/naturalistic/results_meta_della/conv6lr', 
                        help='Directory containing the saved naturalistic models (seed folders).')
    parser.add_argument('--data_dir', type=str, 
                        default='/scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb', 
                        help='Directory containing the PB task HDF5 files.')
    parser.add_argument('--output_dir', type=str, default='results/cross_domain_evaluation/', 
                        help='Directory to save the results JSON file.')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for evaluation.')
    parser.add_argument('--max_batches', type=int, default=30,
                        help='Maximum number of batches to evaluate per task. Set to 0 or None to evaluate all.')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Enable verbose output with detailed progress tracking.')
    
    args = parser.parse_args()
    if args.max_batches == 0:
        args.max_batches = None
    main(args) 