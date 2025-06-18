import os
import json
import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def aggregate_results(results_dir, output_dir):
    """
    Aggregates results from variable task experiments, calculates mean accuracies on held-out tasks,
    and generates a summary plot.

    Args:
        results_dir (str): The base directory where experimental results are stored.
        output_dir (str): The directory to save the aggregated summary and plot.
    """
    print(f"Aggregating results from: {results_dir}")
    # Regex to parse architecture, seed, and task count from the directory name
    pattern = re.compile(r"arch_(?P<arch>[\w\d]+)_seed_(?P<seed>\d+)_(\d+)tasks")
    
    all_results = []
    
    # Discover all PB tasks from a sample results file to identify held-out tasks
    all_pb_tasks = set()
    sample_files = glob.glob(os.path.join(results_dir, "**/results.json"), recursive=True)
    if not sample_files:
        print("No result files found. Exiting.")
        return
        
    with open(sample_files[0], 'r') as f:
        sample_data = json.load(f)
        all_pb_tasks = set(sample_data['test_results'].keys())
    print(f"Discovered all PB tasks for evaluation: {sorted(list(all_pb_tasks))}")

    # Iterate over all result files
    for filepath in sample_files:
        match = pattern.search(filepath)
        if not match:
            print(f"Warning: Could not parse metadata from path: {filepath}")
            continue
            
        data = match.groupdict()
        num_tasks = int(os.path.basename(os.path.dirname(filepath)).split('_')[-1].replace('tasks', ''))
        
        with open(filepath, 'r') as f:
            results_data = json.load(f)
        
        training_tasks = set(results_data['args']['tasks'])
        held_out_tasks = all_pb_tasks - training_tasks

        # Calculate average accuracy on held-out tasks
        held_out_accuracies = []
        for task in held_out_tasks:
            if task in results_data['test_results']:
                held_out_accuracies.append(results_data['test_results'][task]['post_adaptation_accuracy'])
        
        if held_out_accuracies:
            avg_held_out_accuracy = sum(held_out_accuracies) / len(held_out_accuracies)
        else:
            avg_held_out_accuracy = None # Handle case with 10 tasks where there are no held-out tasks

        all_results.append({
            'arch': data['arch'],
            'seed': int(data['seed']),
            'num_tasks': num_tasks,
            'avg_held_out_accuracy': avg_held_out_accuracy
        })

    if not all_results:
        print("No valid results were aggregated.")
        return

    # Create a pandas DataFrame for easy analysis
    df = pd.DataFrame(all_results)
    
    # Save the raw aggregated data
    raw_csv_path = os.path.join(output_dir, 'variable_task_raw_results.csv')
    df.to_csv(raw_csv_path, index=False)
    print(f"Saved raw aggregated data to {raw_csv_path}")

    # Calculate mean and std dev across seeds
    summary_df = df.groupby('num_tasks')['avg_held_out_accuracy'].agg(['mean', 'std']).reset_index()
    
    # Save the summary data
    summary_json_path = os.path.join(output_dir, 'variable_task_summary.json')
    summary_df.to_json(summary_json_path, orient='records', indent=4)
    print(f"Saved summary data to {summary_json_path}")
    
    summary_csv_path = os.path.join(output_dir, 'variable_task_summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary data to {summary_csv_path}")

    # Plotting the results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.lineplot(data=summary_df, x='num_tasks', y='mean', marker='o', ax=ax, label='Mean Accuracy')
    ax.fill_between(summary_df['num_tasks'], 
                    summary_df['mean'] - summary_df['std'], 
                    summary_df['mean'] + summary_df['std'], 
                    alpha=0.2, label='Standard Deviation')

    ax.set_title('Impact of Number of Meta-Training Tasks on Held-Out Task Performance', fontsize=16, pad=20)
    ax.set_xlabel('Number of Training Tasks', fontsize=12)
    ax.set_ylabel('Mean Post-Adaptation Accuracy on Held-Out Tasks', fontsize=12)
    ax.set_xticks(range(1, 11))
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plot_path = os.path.join(output_dir, 'variable_task_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate results from the variable task experiment.")
    parser.add_argument('--results_dir', type=str, required=True, 
                        help='Directory containing the experimental results.')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save aggregated results and plots.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    aggregate_results(args.results_dir, args.output_dir) 