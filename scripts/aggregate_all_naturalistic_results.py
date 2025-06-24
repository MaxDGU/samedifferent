import os
import re
import json
import pandas as pd
from pathlib import Path
import numpy as np

# --- Configuration ---
VANILLA_LOG_DIR = Path('logs_naturalistic_vanilla')
META_LOG_DIR = Path('logs_naturalistic_meta') # As found in the SLURM script
SLURM_LOG_DIR = Path('slurm_logs') # Assuming this is where the .out files are

def find_slurm_output_file(job_id_str):
    """Finds the SLURM output file based on a job ID string like 'slurm-654321'."""
    if not SLURM_LOG_DIR.exists():
        return None
    
    # Simple search for now, can be made more robust if needed
    for f in SLURM_LOG_DIR.glob(f"*{job_id_str}*.out"):
        return f
    return None

def scrape_accuracy_from_log(log_file):
    """
    Scrapes the last 'Val Acc' value from a SLURM output file.
    This is a fallback for when JSON is missing.
    """
    if not log_file or not log_file.exists():
        return None
    
    last_val_acc = None
    # Regex to find lines like "Val Acc: 0.5418"
    acc_regex = re.compile(r"Val Acc:\s*([0-9\.]+)")
    
    with open(log_file, 'r') as f:
        for line in f:
            match = acc_regex.search(line)
            if match:
                last_val_acc = float(match.group(1))
    
    return last_val_acc

def parse_results(base_dir, slurm_job_name_pattern):
    """
    Parses results from a base directory, checking for JSON first,
    then falling back to scraping SLURM logs.
    """
    results = {}
    if not base_dir.exists():
        print(f"Warning: Directory not found: {base_dir}")
        return results

    architectures = [d.name for d in base_dir.iterdir() if d.is_dir()]
    
    for arch in architectures:
        arch_dir = base_dir / arch
        accuracies = []
        
        seed_dirs = [d for d in arch_dir.iterdir() if d.is_dir() and d.name.startswith('seed_')]
        for seed_dir in seed_dirs:
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
                # This part is a placeholder - finding the exact .out file requires more info.
                # For now, we'll assume a naming convention or that you'll add it.
                # Let's try to find a job_id file if it exists
                job_id_file = seed_dir / 'slurm_job_id.txt'
                if job_id_file.exists():
                     with open(job_id_file, 'r') as f:
                         job_id = f.read().strip()
                         slurm_file = find_slurm_output_file(job_id)
                         if slurm_file:
                             scraped_acc = scrape_accuracy_from_log(slurm_file)
                             if scraped_acc:
                                 accuracy = scraped_acc
                else:
                    # Generic fallback if no job id is logged
                    # NOTE: This part is highly dependent on your SLURM output naming scheme
                    # You may need to manually find the right .out files if this fails.
                    print(f"Warning: No JSON and no job_id file for {seed_dir}. Cannot scrape SLURM log.")


            if accuracy is not None:
                accuracies.append(accuracy)
            else:
                print(f"Warning: Could not find any result for {seed_dir}")
        
        if accuracies:
            results[arch] = {
                'accuracies': accuracies,
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'num_seeds_successful': len(accuracies)
            }
            
    return results

def print_summary(title, results_data):
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
            'Successful Seeds': f"{data['num_seeds_successful']}",
            'Raw Accuracies': ", ".join([f"{acc:.4f}" for acc in data['accuracies']])
        })
    
    df = pd.DataFrame(summary_list)
    df.set_index('Architecture', inplace=True)
    print(df.to_string())


def main():
    """Main function to orchestrate the aggregation and printing."""
    print("--- Aggregating All Naturalistic Experiment Results ---")
    print(f"Searching for vanilla results in: {VANILLA_LOG_DIR}")
    print(f"Searching for meta results in:    {META_LOG_DIR}")
    
    vanilla_results = parse_results(VANILLA_LOG_DIR, "vanilla_nat_exp")
    meta_results = parse_results(META_LOG_DIR, "meta_nat_exp")
    
    print_summary("Vanilla Model Results (Validation Accuracy)", vanilla_results)
    print_summary("Meta-Trained Model Results (Validation Accuracy)", meta_results)


if __name__ == '__main__':
    main()
