import os
import torch
import numpy as np
import argparse
import json
import pandas as pd
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Path Setup ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from baselines.models.conv6 import SameDifferentCNN
from baselines.models.utils import SameDifferentDataset

def evaluate_model(model, data_loader, device):
    """Evaluates the model on a given test dataset."""
    model.eval()
    model.to(device)
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
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
            
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
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

    # --- Evaluation Loop ---
    all_results = {}

    for seed in tqdm(naturalistic_seeds, desc="Processing Seeds"):
        # --- Load Model ---
        model_path = model_path_template.format(seed=seed)
        if not os.path.exists(model_path):
            print(f"Warning: Model for seed {seed} not found at {model_path}. Skipping.")
            continue
            
        model = SameDifferentCNN()
        try:
            # Load checkpoint, handling potential DataParallel 'module.' prefix
            state_dict = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading model for seed {seed}: {e}. Skipping.")
            continue

        seed_results = {}
        for task in tqdm(pb_tasks, desc=f"Evaluating Seed {seed}", leave=False):
            # --- Load PB Task Data ---
            try:
                # Correctly set the support sizes to match the available test data
                test_dataset = SameDifferentDataset(args.data_dir, task_names=[task], split='test', support_sizes=[4, 6, 8, 10])
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            except ValueError as e:
                print(f"Warning: Could not load data for task '{task}'. Skipping. Error: {e}")
                continue

            # --- Evaluate and Store Accuracy ---
            accuracy = evaluate_model(model, test_loader, device)
            seed_results[task] = accuracy
            print(f"  Seed {seed} | Task: {task:12} | Accuracy: {accuracy:.2f}%")

        all_results[f'seed_{seed}'] = seed_results

    # --- Save and Summarize Results ---
    if not all_results:
        print("\nNo models were evaluated. Exiting.")
        return

    # Save detailed results to JSON
    results_path = os.path.join(args.output_dir, 'naturalistic_on_pb_accuracies.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nDetailed results saved to {results_path}")

    # Create and print a summary DataFrame
    df = pd.DataFrame(all_results).T # Transpose to have seeds as rows
    
    # Calculate mean and std deviation for each task
    summary = df.agg(['mean', 'std'])
    
    # Calculate overall mean across all tasks and seeds
    overall_mean = df.to_numpy().mean()
    summary['Overall_Avg'] = [overall_mean, df.to_numpy().std()]
    
    print("\n--- Summary of Accuracies (%) ---")
    print(summary.round(2))
    

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
    
    args = parser.parse_args()
    main(args) 