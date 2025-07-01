import os
import json
import glob
import argparse
import pandas as pd
import numpy as np

def aggregate_results(results_dir):
    """
    Aggregates results from all FOMAML experiment JSON files.
    """
    json_files = glob.glob(os.path.join(results_dir, '**/final_summary_results.json'), recursive=True)
    
    if not json_files:
        raise FileNotFoundError(f"No 'final_summary_results.json' files found recursively in {results_dir}")

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

def generate_latex_table(df):
    """
    Generates and prints a LaTeX table from the aggregated results.
    """
    if df.empty:
        print("Cannot generate table from empty dataframe.")
        return

    # --- Create the main results table ---
    # Pivot table to get Tasks as rows and Arch as columns
    pivot_df = df.pivot_table(
        index='task',
        columns='architecture',
        values='accuracy',
        aggfunc=[np.mean, np.std]
    )
    
    # --- Calculate the average across tasks for each architecture ---
    avg_across_tasks = df.groupby('architecture')['accuracy'].agg(['mean', 'std'])
    
    # Define the desired order for tasks
    task_order = [
        'regular', 'lines', 'open', 'wider_line', 'scrambled',
        'random_color', 'arrows', 'irregular', 'filled', 'original'
    ]
    pivot_df = pivot_df.reindex(task_order)

    # --- Format the table for LaTeX ---
    # Format each cell as "mean ± std"
    formatted_df = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns.levels[1])
    for arch in pivot_df.columns.levels[1]:
        mean_series = pivot_df[('mean', arch)]
        std_series = pivot_df[('std', arch)]
        formatted_df[arch] = mean_series.map('{:.2f}'.format) + ' $\\pm$ ' + std_series.map('{:.2f}'.format)
    
    # --- Add the 'Average' row ---
    avg_row = {}
    for arch in avg_across_tasks.index:
        mean_val = avg_across_tasks.loc[arch, 'mean']
        std_val = avg_across_tasks.loc[arch, 'std']
        avg_row[arch] = f"{mean_val:.2f} $\\pm$ {std_val:.2f}"
    
    formatted_df.loc['Average'] = avg_row
    
    # --- Convert to LaTeX ---
    # Rename columns for clarity in the table
    formatted_df.columns = [col.upper() for col in formatted_df.columns]
    formatted_df.index.name = 'Task'
    
    latex_string = formatted_df.to_latex(
        column_format='l|ccc',
        escape=False,
        caption='FOMAML Test Accuracy by Architecture and Task.',
        label='tab:fomaml_results',
        header=True,
        bold_rows=True
    )
    
    # --- Print the final LaTeX code ---
    print("\\begin{table}[h]")
    print("\\centering")
    print(latex_string)
    print("\\end{table}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate FOMAML experiment results into a LaTeX table.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Base directory containing the experiment results (e.g., results/fomaml_experiments).')
    args = parser.parse_args()

    try:
        results_df = aggregate_results(args.results_dir)
        generate_latex_table(results_df)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main() 