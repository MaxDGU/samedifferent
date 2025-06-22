import json
import numpy as np
import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Aggregate results from retrained PB models.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing the seed-level results (e.g., 'results/pb_retrained_conv6lr/conv6')",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="compiled_results.json",
        help="Name for the output JSON file with aggregated results.",
    )
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.is_dir():
        print(f"Error: Directory not found at {results_path}")
        return

    json_files = list(results_path.glob("seed_*/results.json"))
    if not json_files:
        print(f"No 'results.json' files found in subdirectories of {results_path}")
        return

    print(f"Found {len(json_files)} result files. Aggregating...")

    task_accuracies = {}

    for f in json_files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            test_results = data.get('test_results', {})
            for task, metrics in test_results.items():
                if task not in task_accuracies:
                    task_accuracies[task] = []
                # Handle case where testing a task failed and no 'accuracy' key exists
                if 'accuracy' in metrics:
                    task_accuracies[task].append(metrics.get('accuracy', 0.0))

    if not task_accuracies:
        print("No test results with accuracies found in the JSON files.")
        return

    aggregated_results = {}
    summary_data = []

    print("\n--- Aggregated Test Accuracies (Mean ± Std Dev) ---")
    for task, accs in sorted(task_accuracies.items()):
        # Ensure we only process tasks that had successful runs
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

    # Calculate and print overall average, ensuring we don't divide by zero
    valid_means = [res['mean_accuracy'] for res in aggregated_results.values() if res['num_seeds'] > 0]
    if valid_means:
        overall_mean = np.mean(valid_means)
        print(f"\nOverall Mean Accuracy across all tasks and seeds: {overall_mean:.4f}")

    output_path = results_path / args.output_file
    with open(output_path, 'w') as f:
        json.dump(aggregated_results, f, indent=4)

    print(f"\nAggregated results saved to {output_path}")

if __name__ == "__main__":
    main() 