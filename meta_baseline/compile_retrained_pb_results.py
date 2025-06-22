import json
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
import re

def parse_log_file(file_path):
    """Parses a single log file to extract test accuracies for each task."""
    task_accuracies = {}
    # Regex to find the task name and the test accuracy
    task_regex = re.compile(r"Testing on task: ([\w_]+)")
    acc_regex = re.compile(r"Test Acc: ([\d\.]+)")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        task_match = task_regex.search(line)
        if task_match:
            task_name = task_match.group(1)
            
            if task_name in task_accuracies:
                continue

            # Search in the subsequent lines for the accuracy for this task
            # A window of 50 lines should be more than enough.
            for j in range(i + 1, min(i + 50, len(lines))):
                # We simplify the regex here to be more robust
                acc_match = re.search(r"Test Acc: ([\d\.]+)", lines[j])
                if acc_match:
                    accuracy = float(acc_match.group(1))
                    task_accuracies[task_name] = accuracy
                    break  # Found acc for this task, break inner loop
    return task_accuracies

def main():
    parser = argparse.ArgumentParser(description="Aggregate results from retrained PB model log files.")
    parser.add_argument(
        "--logs_dir",
        type=str,
        required=True,
        help="Directory containing the log files (e.g., 'logs/')",
    )
    parser.add_argument(
        "--log_pattern",
        type=str,
        default="pb_retrain_conv6_*.out",
        help="Glob pattern for the log files to parse.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the compiled results JSON file (e.g., 'results/pb_retrained_conv6lr/conv6')",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="compiled_results_from_logs.json",
        help="Name for the output JSON file with aggregated results.",
    )
    args = parser.parse_args()

    logs_path = Path(args.logs_dir)
    if not logs_path.is_dir():
        print(f"Error: Log directory not found at {logs_path}")
        return
        
    output_dir_path = Path(args.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    log_files = list(logs_path.glob(args.log_pattern))
    if not log_files:
        print(f"No log files found in {logs_path} matching pattern '{args.log_pattern}'")
        return

    print(f"Found {len(log_files)} log files. Parsing...")

    # This will be a dict of lists: {task: [acc1, acc2, ...]}
    all_task_accuracies = {}

    for f in log_files:
        parsed_data = parse_log_file(f)
        for task, acc in parsed_data.items():
            if task not in all_task_accuracies:
                all_task_accuracies[task] = []
            all_task_accuracies[task].append(acc)

    if not all_task_accuracies:
        print("No test results could be parsed from the log files.")
        return

    aggregated_results = {}
    summary_data = []

    print("\n--- Aggregated Test Accuracies (Mean ± Std Dev) ---")
    for task, accs in sorted(all_task_accuracies.items()):
        if not accs:
            continue
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        aggregated_results[task] = {
            'accuracies': accs,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'num_seeds': len(accs)
        }
        summary_data.append({
            "Task": task,
            "Mean Accuracy": f"{mean_acc:.4f}",
            "Std Dev": f"{std_acc:.4f}",
            "N": len(accs)
        })
    
    if not summary_data:
        print("Could not generate summary. No valid data found.")
        return
        
    df = pd.DataFrame(summary_data)
    print(df.to_string(index=False))
    
    # Calculate and print overall average
    overall_mean = np.mean([res['mean_accuracy'] for res in aggregated_results.values()])
    print(f"\nOverall Mean Accuracy across all tasks and seeds: {overall_mean:.4f}")

    output_path = output_dir_path / args.output_file
    with open(output_path, 'w') as f:
        json.dump(aggregated_results, f, indent=4)

    print(f"\nAggregated results saved to {output_path}")

if __name__ == "__main__":
    main() 