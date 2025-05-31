#!/usr/bin/env python
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import collections

# Define constants (should match your directory structure and experiment setup)
PB_TASKS = ['regular', 'lines', 'open', 'wider_line', 'scrambled',
            'random_color', 'arrows', 'irregular', 'filled', 'original']
ARCHITECTURES = ['conv2', 'conv4', 'conv6']
SEEDS = list(range(10)) # Seeds 0 through 9

def parse_results(base_results_dir):
    """
    Parses metrics.json files to extract validation accuracies.

    Args:
        base_results_dir (Path): The base directory containing the results.
                                 Expected structure: base_dir/task/arch/seed_X/metrics.json

    Returns:
        dict: A nested dictionary storing lists of validation accuracies:
              results[task][architecture] = [val_acc_seed0, val_acc_seed1, ...]
    """
    parsed_accuracies = collections.defaultdict(lambda: collections.defaultdict(list))
    
    print(f"Parsing results from: {base_results_dir}")

    for task_name in PB_TASKS:
        for arch_name in ARCHITECTURES:
            for seed in SEEDS:
                metrics_file = base_results_dir / task_name / arch_name / f'seed_{seed}' / 'metrics.json'
                
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            data = json.load(f)
                        
                        # Try to get 'val_acc' from training_history first, then top-level 'best_val_acc'
                        val_acc = None
                        if 'training_history' in data and 'val_acc' in data['training_history'] and data['training_history']['val_acc']:
                            # Use the last recorded validation accuracy from the history if available and not empty
                            val_acc = data['training_history']['val_acc'][-1]
                        elif 'best_val_acc' in data: # Fallback to best_val_acc if history is not as expected
                            val_acc = data['best_val_acc']
                        elif 'val_acc' in data: # Fallback for older metric files maybe
                             val_acc = data['val_acc']


                        if val_acc is not None:
                            parsed_accuracies[task_name][arch_name].append(float(val_acc))
                        else:
                            print(f"    Warning: 'val_acc' or 'best_val_acc' not found or empty in {metrics_file}")
                            
                    except json.JSONDecodeError:
                        print(f"    Warning: Could not decode JSON from {metrics_file}")
                    except Exception as e:
                        print(f"    Warning: Error processing {metrics_file}: {e}")
                else:
                    print(f"    Warning: Metrics file not found: {metrics_file}")
            
            if not parsed_accuracies[task_name][arch_name]:
                print(f"  Info: No validation accuracies found for task '{task_name}', architecture '{arch_name}'.")

    return parsed_accuracies

def calculate_stats(parsed_accuracies):
    """
    Calculates mean and standard deviation for validation accuracies.

    Args:
        parsed_accuracies (dict): Dict from parse_results.
                                  results[task][architecture] = [val_acc_seed0, ...]

    Returns:
        dict: stats[task][architecture] = {'mean': ..., 'std': ..., 'count': ...}
    """
    stats = collections.defaultdict(lambda: collections.defaultdict(dict))
    for task, arch_data in parsed_accuracies.items():
        for arch, acc_list in arch_data.items():
            if acc_list:
                stats[task][arch]['mean'] = np.mean(acc_list)
                stats[task][arch]['std'] = np.std(acc_list)
                stats[task][arch]['count'] = len(acc_list)
            else:
                stats[task][arch]['mean'] = np.nan # Or 0, depending on how you want to treat missing data
                stats[task][arch]['std'] = np.nan  # Or 0
                stats[task][arch]['count'] = 0
    return stats

def plot_accuracies(stats, output_plot_path):
    """
    Generates and saves a grouped bar chart of mean validation accuracies.

    Args:
        stats (dict): Dict from calculate_stats.
                      stats[task][architecture] = {'mean': ..., 'std': ...}
        output_plot_path (Path): Path to save the generated plot.
    """
    n_tasks = len(PB_TASKS)
    n_archs = len(ARCHITECTURES)

    means = np.zeros((n_tasks, n_archs))
    stds = np.zeros((n_tasks, n_archs))
    counts = np.zeros((n_tasks, n_archs), dtype=int)

    for i, task in enumerate(PB_TASKS):
        for j, arch in enumerate(ARCHITECTURES):
            if task in stats and arch in stats[task]:
                means[i, j] = stats[task][arch].get('mean', np.nan)
                stds[i, j] = stats[task][arch].get('std', np.nan)
                counts[i,j] = stats[task][arch].get('count',0)
            else:
                means[i, j] = np.nan
                stds[i, j] = np.nan
                counts[i,j] = 0
    
    x = np.arange(n_tasks)  # the label locations
    width = 0.25  # the width of the bars. If n_archs is different, adjust this.
    
    fig, ax = plt.subplots(figsize=(18, 10)) # Increased figure size

    for j, arch in enumerate(ARCHITECTURES):
        # Calculate offset for each group of bars
        offset = width * (j - (n_archs - 1) / 2.0) 
        
        # Filter out NaNs for plotting (bar doesn't handle NaNs well for position)
        valid_indices = ~np.isnan(means[:, j])
        if np.any(valid_indices):
             rects = ax.bar(x[valid_indices] + offset, means[valid_indices, j], width, 
                            label=f"{arch} (N={counts[valid_indices,j][0] if np.any(valid_indices) else 0 })", # Show N for first valid entry as example
                            yerr=stds[valid_indices, j], capsize=5, alpha=0.8)
        else: # Plot a dummy bar if all NaNs, or handle differently
            ax.bar(x + offset, np.zeros(n_tasks), width, label=f"{arch} (N=0)", alpha=0.8)


    ax.set_ylabel('Mean Validation Accuracy', fontsize=14)
    ax.set_title(f'Mean Validation Accuracy by Task and Architecture (across up to {len(SEEDS)} seeds)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(PB_TASKS, rotation=45, ha="right", fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., title="Architecture", fontsize=10, title_fontsize=12)
    ax.set_ylim([0, 1.05]) # Accuracy is between 0 and 1
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for legend outside
    
    try:
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_plot_path)
        print(f"\nPlot saved to {output_plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Parse metrics files and plot validation accuracies.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Base directory containing the single-task results (e.g., /scratch/gpfs/user/results/pb_single_tasks_fixed)')
    parser.add_argument('--output_plot', type=str, required=True,
                        help='Full path to save the generated plot image (e.g., /scratch/gpfs/user/plots/val_acc_summary.png)')
    
    args = parser.parse_args()

    base_results_dir = Path(args.results_dir)
    output_plot_file = Path(args.output_plot)

    if not base_results_dir.is_dir():
        print(f"Error: Results directory not found: {base_results_dir}")
        return

    parsed_accs = parse_results(base_results_dir)
    
    if not any(parsed_accs.values()): # Check if any data was parsed
        print("No data parsed. Exiting.")
        return
        
    stats = calculate_stats(parsed_accs)
    plot_accuracies(stats, output_plot_file)

if __name__ == '__main__':
    main()