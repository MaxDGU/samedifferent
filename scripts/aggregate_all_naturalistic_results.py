import os
import re
import json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
VANILLA_LOG_DIR = Path('logs_naturalistic_vanilla')
META_LOG_DIR = Path('logs_naturalistic_meta')
SLURM_LOG_DIR = Path('slurm_logs') # Assumes you run this script from the project root on Della

# We need to replicate the SLURM array logic to find the correct log files
ARCHITECTURES = ["conv2lr", "conv4lr", "conv6lr"]
ALL_SEEDS = [42, 123, 555, 789, 999, 111, 222, 333]
NEW_SEEDS = [111, 222, 333]

def find_meta_slurm_output_file(arch, seed):
    """
    Finds the SLURM output file for a given meta-learning run by doing a robust glob search.
    """
    if not SLURM_LOG_DIR.exists():
        print(f"Warning: SLURM log directory not found at {SLURM_LOG_DIR}")
        return None
    
    # Robustly search for any file containing the arch and seed. This is less brittle.
    search_pattern = f"*_{arch}_*_{seed}_*.out"
    found_files = list(SLURM_LOG_DIR.glob(search_pattern))
    
    if not found_files:
        # Try another common ordering
        search_pattern = f"*_{seed}_*_{arch}_*.out"
        found_files = list(SLURM_LOG_DIR.glob(search_pattern))

    if not found_files:
        # Try to find based on old naming conventions as a final fallback
        if seed in [111, 222, 333]:
             search_pattern = f"meta_nat_more_seeds_*{seed}*.out"
        else:
             search_pattern = f"meta_nat_exp_add_seeds_*{seed}*.out"
        found_files = list(SLURM_LOG_DIR.glob(search_pattern))

    if not found_files:
        return None
    
    if len(found_files) > 1:
        print(f"Warning: Found multiple log files for {arch}/{seed}. Using most recent.")
        found_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
    return found_files[0]


def scrape_accuracy_from_log(log_file, experiment_type='vanilla'):
    """
    Scrapes the last accuracy value from a SLURM output file.
    - For 'vanilla' runs, it looks for the last 'Val Acc:'.
    - For 'meta' runs, it looks for the final 'Test Results: ... Avg Acc:'.
    """
    if not log_file or not log_file.exists():
        return None
    
    last_acc = None
    
    if experiment_type == 'meta':
        # "Test Results: Avg Loss: 0.3142, Avg Acc: 0.928 across 3600 episodes."
        acc_regex = re.compile(r"Test Results:.*Avg Acc:\s*([0-9\.]+)")
    else: # Default to vanilla
        # "Val Acc: 0.5418" or "val_acc: 0.5418"
        acc_regex = re.compile(r"val acc:\s*([0-9\.]+)", re.IGNORECASE)
    
    with open(log_file, 'r', errors='ignore') as f:
        for line in f:
            match = acc_regex.search(line)
            if match:
                try:
                    last_acc = float(match.group(1))
                except ValueError:
                    continue # Ignore if conversion fails
    
    return last_acc

def parse_results(base_dir, experiment_type='vanilla'):
    """
    Parses results, falling back to specific SLURM log scraping for meta runs.
    """
    results = {}
    if not base_dir.exists():
        print(f"Warning: Directory not found: {base_dir}")
        return results

    for arch in ARCHITECTURES:
        arch_dir = base_dir / arch
        if not arch_dir.exists():
            continue
            
        accuracies = []
        for seed in ALL_SEEDS:
            # Vanilla runs only used the original 5 seeds
            if experiment_type == 'vanilla' and seed in NEW_SEEDS:
                continue

            seed_dir = arch_dir / f"seed_{seed}"
            if not seed_dir.exists():
                continue

            accuracy = None
            metrics_file = seed_dir / 'training_metrics.json'
            
            # --- Strategy 1: Try to parse JSON ---
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        val_accs = data.get('val_acc', [])
                        if val_accs:
                            accuracy = max(val_accs)
                except (json.JSONDecodeError, TypeError):
                    print(f"Warning: Could not parse JSON in {metrics_file}")

            # --- Strategy 2: Fallback to scraping SLURM .out file ---
            if accuracy is None:
                log_file = None
                if experiment_type == 'meta':
                    log_file = find_meta_slurm_output_file(arch, seed)
                
                if log_file:
                    scraped_acc = scrape_accuracy_from_log(log_file, experiment_type=experiment_type)
                    if scraped_acc is not None:
                        accuracy = scraped_acc
                    else:
                        print(f"Warning: Found log {log_file.name} but could not scrape accuracy.")
                else:
                    if experiment_type == 'meta':
                         print(f"Warning: No JSON and could not find SLURM log for meta run {arch}/seed_{seed}")

            if accuracy is not None:
                accuracies.append(accuracy)
        
        if accuracies:
            results[arch] = {
                'accuracies': accuracies,
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'num_seeds_successful': len(accuracies)
            }
            
    return results

def print_summary(title, results_data, experiment_type):
    """Prints a formatted summary of the results."""
    print(f"\n--- {title} ---")
    if not results_data:
        print("No results found.")
        return

    summary_list = []
    for arch, data in sorted(results_data.items()):
        summary_list.append({
            'Architecture': arch,
            'Mean Accuracy': f"{data['mean_accuracy']:.4f}",
            'Std Dev': f"{data['std_accuracy']:.4f}",
            'Successful Seeds': f"{data['num_seeds_successful']}/{len(ALL_SEEDS)}",
            'Raw Accuracies': ", ".join([f"{acc:.4f}" for acc in data['accuracies']])
        })
    
    df = pd.DataFrame(summary_list)
    if not df.empty:
        df.set_index('Architecture', inplace=True)
        # Add a note about the accuracy type
        if experiment_type == 'meta':
            print("NOTE: Accuracies are final TEST set results from meta-training runs.")
        else:
            print("NOTE: Accuracies are best VALIDATION set results from vanilla training runs.")
        print(df.to_string())
    else:
        print("No successful runs found to summarize.")

def plot_results(vanilla_data, meta_data):
    """Generates and saves a bar chart comparing vanilla and meta results."""
    
    plot_data = []
    for arch in ARCHITECTURES:
        if arch in vanilla_data:
            plot_data.append({
                'Architecture': arch.upper(),
                'Training Type': 'Vanilla',
                'Mean Accuracy': vanilla_data[arch]['mean_accuracy'],
                'Std Dev': vanilla_data[arch]['std_accuracy']
            })
        if arch in meta_data:
            plot_data.append({
                'Architecture': arch.upper(),
                'Training Type': 'Meta-Trained',
                'Mean Accuracy': meta_data[arch]['mean_accuracy'],
                'Std Dev': meta_data[arch]['std_accuracy']
            })

    if not plot_data:
        print("\nNo data to plot.")
        return
        
    df = pd.DataFrame(plot_data)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # --- ROBUST PLOTTING LOGIC ---
    # This avoids the seaborn IndexError by plotting each group manually.
    bar_width = 0.35
    index = np.arange(len(ARCHITECTURES))
    
    vanilla_means = df[df['Training Type'] == 'Vanilla']['Mean Accuracy']
    vanilla_std = df[df['Training Type'] == 'Vanilla']['Std Dev']
    
    meta_means = df[df['Training Type'] == 'Meta-Trained']['Mean Accuracy']
    meta_std = df[df['Training Type'] == 'Meta-Trained']['Std Dev']
    
    # Plot bars
    ax.bar(index - bar_width/2, vanilla_means, bar_width, yerr=vanilla_std,
           capsize=5, label='Vanilla', color='#1f77b4')
    ax.bar(index + bar_width/2, meta_means, bar_width, yerr=meta_std,
           capsize=5, label='Meta-Trained', color='#ff7f0e')

    ax.set_title('Comparison of Vanilla vs. Meta-Trained Models on Naturalistic Data', fontsize=16)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_xlabel('Model Architecture', fontsize=12)
    ax.set_xticks(index)
    ax.set_xticklabels([arch.upper() for arch in ARCHITECTURES])
    ax.set_ylim(bottom=0.5)
    ax.legend(title='Training Type')
    ax.grid(axis='x') # Remove vertical grid lines for clarity
    
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'naturalistic_vanilla_vs_meta_comparison.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Successfully saved plot to {output_path}")

def main():
    """Main function to orchestrate the aggregation and printing."""
    print("--- Aggregating All Naturalistic Experiment Results ---")
    print(f"Searching for vanilla results in: {VANILLA_LOG_DIR}")
    print(f"Searching for meta results in:    {META_LOG_DIR}")
    print(f"Searching for SLURM logs in:    {SLURM_LOG_DIR}")
    
    vanilla_results = parse_results(VANILLA_LOG_DIR, experiment_type='vanilla')
    meta_results = parse_results(META_LOG_DIR, experiment_type='meta')
    
    print_summary("Vanilla Model Results", vanilla_results, experiment_type='vanilla')
    print_summary("Meta-Trained Model Results", meta_results, experiment_type='meta')
    
    plot_results(vanilla_results, meta_results)


if __name__ == '__main__':
    main()
