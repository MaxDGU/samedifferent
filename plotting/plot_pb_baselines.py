import os
import json
import numpy as np
from pathlib import Path
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define constants
PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]
ARCHITECTURES = ['conv2', 'conv4', 'conv6']
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

def main():
    try:
        # Read the compiled results
        with open('results/pb_baselines/compiled_stats/all_stats.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Error: Could not find compiled results file. Please run compile_pb_baselines.py first.")
        return
    except json.JSONDecodeError:
        print("Error: Could not parse results file. The JSON file may be corrupted.")
        return

    # Base results directory
    results_dir = 'results/pb_baselines'
    
    print(f"\nLooking for results in: {results_dir}")
    
    # Dictionary to store all results
    all_results = {
        'per_architecture': {},
        'per_task': {},
        'overall': {}
    }
    
    # Track if we found any results
    found_any_results = False
    
    # Collect results for each task
    for task in PB_TASKS:
        print(f"\nProcessing task: {task}")
        all_results['per_task'][task] = {
            'per_architecture': {},
            'overall': {
                'accuracies': [],
                'losses': []
            }
        }
        
        for arch in ARCHITECTURES:
            task_accuracies = []
            task_losses = []
            
            # Collect results across seeds
            for seed in SEEDS:
                seed_str = "seed_" + str(seed)  
                # Path format: results/pb_baselines/{task_name}/{architecture}/{seed_num}/metrics.json
                result_path = os.path.join(results_dir, task, arch, seed_str, 'metrics.json')
                
                if os.path.exists(result_path):
                    try:
                        with open(result_path, 'r') as f:
                            metrics = json.load(f)
                            
                        # Get final metrics
                        if 'test_acc' in metrics:
                            task_accuracies.append(metrics['test_acc'])
                            found_any_results = True
                        if 'test_loss' in metrics:
                            task_losses.append(metrics['test_loss'])
                    except Exception as e:
                        print(f"Error reading {result_path}: {str(e)}")
                else:
                    print(f"No results found at: {result_path}")
            
            # Calculate statistics for this task-architecture combination
            if task_accuracies:
                task_arch_stats = {
                    'mean_accuracy': float(np.mean(task_accuracies)),
                    'std_accuracy': float(np.std(task_accuracies)),
                    'mean_loss': float(np.mean(task_losses)) if task_losses else None,
                    'std_loss': float(np.std(task_losses)) if task_losses else None,
                    'num_seeds': len(task_accuracies)
                }
                
                # Store in both per-task and per-architecture results
                all_results['per_task'][task]['per_architecture'][arch] = task_arch_stats
                
                if arch not in all_results['per_architecture']:
                    all_results['per_architecture'][arch] = {
                        'per_task': {},
                        'overall': {
                            'accuracies': [],
                            'losses': []
                        }
                    }
                
                all_results['per_architecture'][arch]['per_task'][task] = task_arch_stats
                all_results['per_architecture'][arch]['overall']['accuracies'].extend(task_accuracies)
                all_results['per_architecture'][arch]['overall']['losses'].extend(task_losses)
    
    if not found_any_results:
        print("\nERROR: No results found in any of the expected locations!")
        print("Expected path format: {results_dir}/{task}/{arch}/{seed}/metrics.json")
        sys.exit(1)
    
    # Calculate overall statistics for each architecture
    for arch in ARCHITECTURES:
        if arch in all_results['per_architecture']:
            stats = all_results['per_architecture'][arch]['overall']
            if stats['accuracies']:
                stats.update({
                    'mean_accuracy': float(np.mean(stats['accuracies'])),
                    'std_accuracy': float(np.std(stats['accuracies'])),
                    'mean_loss': float(np.mean(stats['losses'])) if stats['losses'] else None,
                    'std_loss': float(np.std(stats['losses'])) if stats['losses'] else None,
                    'num_samples': len(stats['accuracies'])
                })
    
    # Calculate overall statistics across all architectures
    all_accuracies = []
    all_losses = []
    
    for arch in ARCHITECTURES:
        if arch in all_results['per_architecture']:
            stats = all_results['per_architecture'][arch]['overall']
            all_accuracies.extend(stats['accuracies'])
            all_losses.extend(stats['losses'])
    
    if all_accuracies:
        all_results['overall'] = {
            'mean_accuracy': float(np.mean(all_accuracies)),
            'std_accuracy': float(np.std(all_accuracies)),
            'mean_loss': float(np.mean(all_losses)) if all_losses else None,
            'std_loss': float(np.std(all_losses)) if all_losses else None,
            'num_samples': len(all_accuracies)
        }
    
    # Save compiled results
    output_dir = os.path.join(results_dir, 'compiled_stats')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'all_stats.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\nResults Summary:")
    print("="*50)
    
    if 'mean_accuracy' in all_results.get('overall', {}):
        print("\nOverall Statistics:")
        print(f"Mean accuracy across all architectures and tasks: {all_results['overall']['mean_accuracy']*100:.2f}% ± {all_results['overall']['std_accuracy']*100:.2f}%")
        if all_results['overall'].get('mean_loss') is not None:
            print(f"Mean loss: {all_results['overall']['mean_loss']:.4f} ± {all_results['overall']['std_loss']:.4f}")
    
        print("\nPer-Architecture Statistics:")
        for arch in ARCHITECTURES:
            if arch in all_results['per_architecture']:
                stats = all_results['per_architecture'][arch]['overall']
                if 'mean_accuracy' in stats:
                    print(f"\n{arch}:")
                    print(f"Mean accuracy: {stats['mean_accuracy']*100:.2f}% ± {stats['std_accuracy']*100:.2f}%")
                    if stats.get('mean_loss'):
                        print(f"Mean loss: {stats['mean_loss']:.4f} ± {stats['std_loss']:.4f}")
    
        print("\nPer-Task Statistics:")
        for task in PB_TASKS:
            print(f"\n{task}:")
            for arch in ARCHITECTURES:
                if arch in all_results['per_task'][task]['per_architecture']:
                    stats = all_results['per_task'][task]['per_architecture'][arch]
                    print(f"{arch}: {stats['mean_accuracy']*100:.2f}% ± {stats['std_accuracy']*100:.2f}%")
    else:
        print("\nNo valid results found to display!")
    
    print(f"\nResults saved to: {output_dir}")

    # Extract data for plotting
    plot_data = []
    for task in results['per_task']:
        for arch in results['per_task'][task]['per_architecture']:
            stats = results['per_task'][task]['per_architecture'][arch]
            plot_data.append({
                'Task': task,
                'Architecture': arch,
                'Accuracy': stats['mean_accuracy'] * 100,  # Convert to percentage
                'Std': stats['std_accuracy'] * 100  # Convert to percentage
            })
    
    # Create DataFrame
    df = pd.DataFrame(plot_data)
    
    # Set the style
    sns.set_theme(style="ticks")
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.size'] = 20 # Increased from 16

    # Create figure with three subplots stacked vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 24))
    
    # Create a color palette for tasks using viridis
    n_tasks = len(PB_TASKS)
    task_colors = dict(zip(PB_TASKS, sns.color_palette("viridis", n_tasks)))
    
    # Plot for each architecture
    architectures = ['conv2', 'conv4', 'conv6']
    axes = [ax1, ax2, ax3]
    
    for ax, arch in zip(axes, architectures):
        # Filter data for this architecture
        arch_data = df[df['Architecture'] == arch].copy()
        # Reset index to get proper x positions for bars
        arch_data = arch_data.reset_index(drop=True)
        
        # Create bar plot with task-specific colors
        bars = ax.bar(
            range(len(arch_data)),
            arch_data['Accuracy'],
            color=[task_colors[task] for task in arch_data['Task']]
        )
        
        # Add error bars
        for idx, row in arch_data.iterrows():
            height = row['Accuracy']
            std = row['Std']
            ax.vlines(idx, height - std, height + std, color='gray', alpha=0.8, linewidth=1)
            ax.hlines(height + std, idx - 0.05, idx + 0.05, color='gray', alpha=0.8, linewidth=1)
            ax.hlines(height - std, idx - 0.05, idx + 0.05, color='gray', alpha=0.8, linewidth=1)
            
            # Adjust spacing based on architecture
            if arch == 'conv4':
                spacing = 0.015  # Less space for conv4
            else:
                spacing = 0.03   # Keep original spacing for conv2 and conv6
                
            ax.text(
                idx,
                height + std + spacing,
                f'{height:.1f}%',
                ha='center',
                va='bottom',
                fontsize=18
            )
        
        # Customize each subplot
        layer_count = {'conv2': '2', 'conv4': '4', 'conv6': '6'}
        ax.set_title(f'{layer_count[arch]} Layer CNN Test Accuracy', pad=20, fontsize=24)  # Increased from 20
        ax.set_ylabel('Accuracy (%)', fontsize=22)  # Increased from 18
        ax.set_xticks(range(len(arch_data)))
        ax.set_xticklabels(arch_data['Task'], rotation=45, ha='right', fontsize=20)  # Increased from 16
        ax.tick_params(axis='both', labelsize=18)  # Increased from 16
        ax.set_ylim(0, 110)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('results/pb_baselines/accuracy_by_arch_stacked.png', 
                dpi=300, 
                bbox_inches='tight')
    
    print("Plots saved as:")
    print("1. accuracy_by_arch_stacked.png")

if __name__ == '__main__':
    main() 