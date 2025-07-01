import os
import json
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_results(results_dir):
    """
    Aggregates results from all FOMAML experiment JSON files.
    """
    json_files = glob.glob(os.path.join(results_dir, 'arch_*/exp_*/final_summary_results.json'))
    
    if not json_files:
        raise FileNotFoundError(f"No 'final_summary_results.json' files found in subdirectories of {results_dir}")

    all_data = []
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            args = data.get('args', {})
            arch = args.get('architecture')
            seed = args.get('seed')
            task_results = data.get('individual_task_test_results', {})

            if not all([arch, seed, task_results]):
                print(f"Warning: Skipping file with missing data: {file_path}")
                continue

            for task, metrics in task_results.items():
                all_data.append({
                    'architecture': arch,
                    'seed': seed,
                    'task': task,
                    'accuracy': metrics.get('accuracy')
                })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}")
            continue
            
    return pd.DataFrame(all_data)

def plot_results(df, output_dir):
    """
    Generates and saves a grouped bar chart of the aggregated results.
    """
    if df.empty:
        print("Cannot generate plot from empty dataframe.")
        return

    # Define the desired order for tasks
    task_order = [
        'regular', 'lines', 'open', 'wider_line', 'scrambled',
        'random_color', 'arrows', 'irregular', 'filled', 'original'
    ]
    df['task'] = pd.Categorical(df['task'], categories=task_order, ordered=True)
    df = df.sort_values('task')

    # Calculate mean and standard deviation
    summary_df = df.groupby(['architecture', 'task'])['accuracy'].agg(['mean', 'std']).reset_index()

    # Set up the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 10))

    # Create the bar plot
    sns.barplot(
        data=summary_df,
        x='task',
        y='mean',
        hue='architecture',
        hue_order=['conv2', 'conv4', 'conv6'],
        ax=ax
    )

    # Add error bars manually
    num_archs = len(summary_df['architecture'].unique())
    bar_width = 0.8 / num_archs
    
    for i, task in enumerate(task_order):
        for j, arch in enumerate(['conv2', 'conv4', 'conv6']):
            task_arch_data = summary_df[(summary_df['task'] == task) & (summary_df['architecture'] == arch)]
            if not task_arch_data.empty:
                mean = task_arch_data['mean'].iloc[0]
                std = task_arch_data['std'].iloc[0]
                
                # Position of the bar
                bar_pos = i - 0.4 + bar_width * (j + 0.5)
                ax.errorbar(bar_pos, mean, yerr=std, fmt='none', c='black', capsize=5)

    # Formatting
    ax.set_title('FOMAML Test Accuracy by Architecture and Task', fontsize=20, pad=20)
    ax.set_xlabel('PB Task', fontsize=16, labelpad=15)
    ax.set_ylabel('Mean Test Accuracy', fontsize=16, labelpad=15)
    ax.tick_params(axis='x', rotation=45, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(title='Architecture', fontsize=14, title_fontsize=16)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls='--', color='gray', label='Chance Accuracy (0.5)')
    
    plt.legend()
    plt.tight_layout()

    # Save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, 'fomaml_task_accuracies.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Aggregate and plot results from FOMAML experiments.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Base directory containing the experiment results (e.g., results/fomaml_experiments).')
    parser.add_argument('--output_dir', type=str, default='./visualizations/fomaml_summary',
                        help='Directory to save the final plot.')
    args = parser.parse_args()

    try:
        results_df = aggregate_results(args.results_dir)
        plot_results(results_df, args.output_dir)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main() 