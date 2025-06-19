import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def parse_slurm_output(file_path):
    """
    Parses a SLURM output file to extract experiment parameters and results.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    try:
        num_tasks_match = re.search(r'Number of Tasks: (\d+)', content)
        num_tasks = int(num_tasks_match.group(1)) if num_tasks_match else None

        seed_match = re.search(r'Seed: (\d+)', content)
        seed = int(seed_match.group(1)) if seed_match else None

        accuracies = re.findall(r'New best validation accuracy: (\d+\.\d+)', content)
        best_acc = float(accuracies[-1]) if accuracies else None

        if num_tasks is None or seed is None or best_acc is None:
            print(f"Warning: Could not parse all required information from {file_path}. Skipping.")
            # Try to find at least the validation accuracy from the end if other info is missing
            final_acc_match = re.findall(r'Val Acc: (\d\.\d+)', content)
            if best_acc is None and final_acc_match:
                 best_acc = float(final_acc_match[-1])

            if num_tasks is None or seed is None or best_acc is None:
                return None


        return {'num_tasks': num_tasks, 'seed': seed, 'accuracy': best_acc}
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
        return None

def main(results_dir, output_dir):
    """
    Main function to aggregate results and plot them.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for f in results_dir.glob('slurm_*.out'):
        data = parse_slurm_output(f)
        if data:
            results.append(data)

    if not results:
        print("No results found. Exiting.")
        return

    df = pd.DataFrame(results)
    
    # Save raw extracted data
    df.to_csv(output_dir / 'variable_task_raw_results.csv', index=False)
    print(f"Saved raw results to {output_dir / 'variable_task_raw_results.csv'}")

    # Aggregate results
    agg_df = df.groupby('num_tasks')['accuracy'].agg(['mean', 'std', lambda x: x.sem()]).rename(columns={'<lambda_0>': 'sem'}).reset_index()
    agg_df.to_csv(output_dir / 'variable_task_aggregated_results.csv', index=False)
    print(f"Saved aggregated results to {output_dir / 'variable_task_aggregated_results.csv'}")


    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(agg_df['num_tasks'], agg_df['mean'], yerr=agg_df['sem'], fmt='-o', capsize=5, label='Mean Accuracy with SEM')
    
    ax.set_xlabel('Number of Meta-Training Tasks', fontsize=12)
    ax.set_ylabel('Final Validation Accuracy', fontsize=12)
    ax.set_title('Model Performance vs. Number of Meta-Training Tasks (conv6)', fontsize=14)
    ax.set_xticks(range(1, agg_df['num_tasks'].max() + 1))
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plot_path = output_dir / 'variable_task_accuracy_vs_num_tasks.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate and plot results from variable task experiments.')
    parser.add_argument('--results_dir', type=str, 
                        default='/scratch/gpfs/mg7411/samedifferent/results/variable_task_experiments/slurm_logs/var_tasks_conv6',
                        help='Directory containing the slurm output files.')
    parser.add_argument('--output_dir', type=str, default='variable_task/results',
                        help='Directory to save plots and aggregated data.')
    
    args = parser.parse_args()
    main(args.results_dir, args.output_dir) 