import os
import json
import numpy as np
from datetime import datetime
import subprocess
import argparse
import random

def create_slurm_script(seed, output_dir):
    """Create a Slurm script for a specific seed"""
    # Get absolute paths
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(workspace_dir, 'data')
    
    return f"""#!/bin/bash
#SBATCH --job-name=exp1_s{seed}
#SBATCH --output={output_dir}/slurm_%j.out
#SBATCH --error={output_dir}/slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=8
#SBATCH --partition=pli
#SBATCH --account=nam

# Load necessary modules
module purge
module load anaconda3
module load cuda/11.7.0

# Activate conda environment
conda activate tensorflow

# Run the training script
python conv6lr.py \\
    --seed {seed} \\
    --output_dir {output_dir} \\
    --pb_data_dir {data_dir}/pb/pb \\
    --svrt_data_dir {data_dir}/svrt_fixed
"""

def submit_jobs(num_seeds=20):
    """Submit jobs for all seeds"""
    # Create timestamp for unique experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"experiment1_results_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Generate seeds
    seeds = list(range(num_seeds))
    
    for seed in seeds:
        # Create seed-specific output directory
        seed_dir = os.path.join(base_output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        
        # Create slurm script for this seed
        script_content = create_slurm_script(seed, seed_dir)
        script_path = os.path.join(seed_dir, f"run_seed_{seed}.sh")
        
        # Write script to file
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        # Submit the job
        try:
            subprocess.run(['sbatch', script_path], check=True)
            print(f"Submitted job for seed {seed}")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job for seed {seed}: {e}")
            continue

def compile_results(experiment_dir):
    """Compile and average results across seeds"""
    results = {
        'test_acc_mean': {},
        'test_acc_std': {},
        'val_acc': []
    }
    
    # Collect results from all seeds
    seed_results = []
    failed_seeds = []
    for seed in range(20):
        result_path = os.path.join(experiment_dir, f"seed_{seed}", "results.json")
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r') as f:
                    seed_results.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Warning: Corrupted results file for seed {seed}")
                failed_seeds.append(seed)
        else:
            failed_seeds.append(seed)
    
    if failed_seeds:
        print(f"\nWarning: Missing or corrupted results for seeds: {failed_seeds}")
    
    if not seed_results:
        print("No valid results found")
        return None
    
    # Calculate means and stds for SVRT test accuracies
    svrt_tasks = [key for key in seed_results[0]['test_metrics'].keys() if key.startswith('svrt_')]
    for task in svrt_tasks:
        task_accs = [r['test_metrics'][task]['acc'] for r in seed_results]
        results['test_acc_mean'][task] = float(np.mean(task_accs))
        results['test_acc_std'][task] = float(np.std(task_accs))
    
    # Calculate average validation accuracy
    val_accs = [max(r['val_acc']) for r in seed_results if 'val_acc' in r and r['val_acc']]
    if val_accs:
        results['val_acc_mean'] = float(np.mean(val_accs))
        results['val_acc_std'] = float(np.std(val_accs))
    
    # Save compiled results
    output_path = os.path.join(experiment_dir, "compiled_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\nExperiment Results Summary:")
    print("=" * 50)
    if 'val_acc_mean' in results:
        print(f"\nValidation Accuracy: {results['val_acc_mean']:.3f} ± {results['val_acc_std']:.3f}")
    print("\nSVRT Test Accuracies:")
    for task, acc in results['test_acc_mean'].items():
        std = results['test_acc_std'][task]
        task_num = task.split('_')[1]  # Extract task number from 'svrt_X'
        print(f"SVRT Problem {task_num}: {acc:.3f} ± {std:.3f}")
    
    return results

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', action='store_true', help='Compile results from a previous run')
    parser.add_argument('--experiment_dir', type=str, help='Directory containing experiment results to compile')
    parser.add_argument('--num_seeds', type=int, default=20, help='Number of seeds to run (default: 20)')
    args = parser.parse_args()
    
    if args.compile:
        if args.experiment_dir:
            compile_results(args.experiment_dir)
        else:
            print("Error: Please provide the experiment directory to compile")
            print("Usage: python run_experiment1.py --compile --experiment_dir <experiment_dir>")
    else:
        # Submit jobs
        print(f"Submitting jobs for {args.num_seeds} seeds...")
        submit_jobs(num_seeds=args.num_seeds)
        
        # Create timestamp once
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = os.path.abspath(f"experiment1_results_{timestamp}")
        
        print(f"\nAll jobs submitted. Experiment directory: {exp_dir}")
        print("\nTo compile results after all jobs complete, run:")
        print(f"python {__file__} --compile --experiment_dir {exp_dir}") 