import os
import json
import numpy as np
from datetime import datetime
import subprocess
import argparse

def create_slurm_script(seed, output_dir):
    """Create a Slurm script for a specific seed"""
    # Get absolute paths
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(workspace_dir, 'data')
    
    return f"""#!/bin/bash
#SBATCH --job-name=exp2_conv6_s{seed}
#SBATCH --output={output_dir}/slurm_logs/conv6_seed{seed}_%j.out
#SBATCH --error={output_dir}/slurm_logs/conv6_seed{seed}_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=mg7411@princeton.edu

# Load necessary modules
module purge
module load cuda/11.7
module load anaconda3

# Activate conda environment (update this to your PyTorch environment name)
conda activate tensorflow

# Set up data directory paths
export DATA_ROOT={data_dir}
export SVRT_DATA_DIR=$DATA_ROOT/svrt_fixed
export PB_DATA_DIR=$DATA_ROOT/pb/pb

# Run the training script
python conv6lr_svrt.py \\
    --seed {seed} \\
    --output_dir {output_dir}/seed{seed} \\
    --svrt_data_dir $SVRT_DATA_DIR \\
    --pb_data_dir $PB_DATA_DIR
"""

def submit_jobs(num_seeds=20):
    """Submit jobs for all seeds"""
    # Create timestamp for unique experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"experiment2_results_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "slurm_logs"), exist_ok=True)
    
    # Generate seeds
    seeds = list(range(num_seeds))
    
    for seed in seeds:
        # Create slurm script for this seed
        script_content = create_slurm_script(seed, base_output_dir)
        script_path = os.path.join(base_output_dir, "slurm_logs", f"run_seed_{seed}.sh")
        
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
    
    return base_output_dir

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
    
    # Calculate means and stds for PB test accuracies
    pb_tasks = [key for key in seed_results[0]['test_metrics'].keys() if key.startswith('pb_')]
    for task in pb_tasks:
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
    print("\nPB Test Accuracies:")
    for task, acc in results['test_acc_mean'].items():
        std = results['test_acc_std'][task]
        task_name = task.split('_')[1]  # Extract task name from 'pb_X'
        print(f"PB Task {task_name}: {acc:.3f} ± {std:.3f}")
    
    return results

if __name__ == "__main__":
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
            print("Usage: python run_experiment2.py --compile --experiment_dir <experiment_dir>")
    else:
        # Submit jobs
        print(f"Submitting jobs for {args.num_seeds} seeds...")
        experiment_dir = submit_jobs(num_seeds=args.num_seeds)
        
        print(f"\nAll jobs submitted. Experiment directory: {experiment_dir}")
        print("\nTo compile results after all jobs complete, run:")
        print(f"python {__file__} --compile --experiment_dir {experiment_dir}") 