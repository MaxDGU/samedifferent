import os
import json
import pandas as pd
from pathlib import Path
import numpy as np

# Define paths
NEW_RESULTS_DIR = Path('logs_naturalistic_vanilla')
OLD_RESULTS_FILE = Path('test_results_vanilla/vanilla_test_summary.json')

def parse_new_results(base_dir):
    """
    Parses the new experimental results from the logs_naturalistic_vanilla directory.
    Extracts the best validation accuracy from each run.
    """
    results = {}
    architectures = [d.name for d in base_dir.iterdir() if d.is_dir()]
    
    for arch in architectures:
        arch_dir = base_dir / arch
        accuracies = []
        
        seed_dirs = [d for d in arch_dir.iterdir() if d.is_dir() and d.name.startswith('seed_')]
        for seed_dir in seed_dirs:
            metrics_file = seed_dir / 'training_metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    # The best validation accuracy is the max value in the 'val_acc' list
                    best_val_acc = max(data.get('val_acc', [0]))
                    if best_val_acc > 0:
                        accuracies.append(best_val_acc)
            else:
                print(f"Warning: Missing training_metrics.json in {seed_dir}")
        
        if accuracies:
            results[arch] = {
                'accuracies': accuracies,
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'num_seeds_successful': len(accuracies)
            }
            
    return results

def load_old_results(file_path):
    """Loads the previously compiled results from the summary JSON file."""
    if not file_path.exists():
        print(f"Error: Old results file not found at {file_path}")
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    """
    Main function to load, process, and compare the old and new results.
    """
    print("--- Comparing Vanilla Naturalistic Runs ---")
    
    # Load and process results
    new_results = parse_new_results(NEW_RESULTS_DIR)
    old_results = load_old_results(OLD_RESULTS_FILE)
    
    if old_results is None:
        return
        
    # Prepare data for comparison table
    comparison_data = []
    all_archs = sorted(list(set(new_results.keys()) | set(old_results.keys())))

    for arch in all_archs:
        old_mean = old_results.get(arch, {}).get('mean_accuracy', 'N/A')
        old_std = old_results.get(arch, {}).get('std_accuracy', 'N/A')
        
        new_mean = new_results.get(arch, {}).get('mean_accuracy', 'N/A')
        new_std = new_results.get(arch, {}).get('std_accuracy', 'N/A')

        if isinstance(old_mean, float): old_mean = f"{old_mean:.4f}"
        if isinstance(old_std, float): old_std = f"{old_std:.4f}"
        if isinstance(new_mean, float): new_mean = f"{new_mean:.4f}"
        if isinstance(new_std, float): new_std = f"{new_std:.4f}"
        
        comparison_data.append({
            'Architecture': arch,
            'Old Mean Accuracy (Test)': old_mean,
            'Old Std Dev (Test)': old_std,
            'New Mean Accuracy (Val)': new_mean,
            'New Std Dev (Val)': new_std
        })
        
    # Create and print DataFrame
    df = pd.DataFrame(comparison_data)
    df.set_index('Architecture', inplace=True)
    
    print("\n--- Results Summary ---")
    print("NOTE: 'Old' results are from final TEST set evaluations.")
    print("NOTE: 'New' results are from best VALIDATION set performance during training.")
    print(df.to_string())
    
    # Print detailed new results
    print("\n--- Detailed New Run Accuracies (Validation) ---")
    for arch, data in new_results.items():
        accs_str = ", ".join([f"{acc:.4f}" for acc in data['accuracies']])
        print(f"  {arch}:")
        print(f"    - Mean Acc: {data['mean_accuracy']:.4f}")
        print(f"    - Std Dev:  {data['std_accuracy']:.4f}")
        print(f"    - Raw Accs: [{accs_str}]")


if __name__ == '__main__':
    main()
