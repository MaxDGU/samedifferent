import os
import json
import numpy as np
from datetime import datetime
import subprocess
import argparse

def create_slurm_script(task, seed, output_dir, models=None):
    """Create a Slurm script that runs specified architectures for a given task and seed"""
    # Get absolute paths
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(workspace_dir, 'svrt', 'data')
    
    # Use specified models or default to all
    models_to_run = models if models else ['conv2', 'conv4', 'conv6']
    
    return f"""#!/bin/bash
#SBATCH --job-name=base_task{task}_s{seed}
#SBATCH --output={output_dir}/slurm_logs/task{task}_seed{seed}_%j.out
#SBATCH --error={output_dir}/slurm_logs/task{task}_seed{seed}_%j.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --mem=10G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --mail-user=mg7411@princeton.edu

# Load necessary modules
module purge
module load cuda/11.7
module load anaconda3

# Activate conda environment
source activate pytorch_env

# Run specified architectures for this task
for model in {' '.join(models_to_run)}; do
    echo "Running $model on task {task} with seed {seed}"
    python $model.py \\
        --task {task} \\
        --seed {seed} \\
        --output_dir {output_dir}/$model/task{task}/seed{seed} \\
        --data_dir {data_dir}
done
"""

def submit_jobs(num_seeds=10, specific_tasks=None, specific_seeds=None, specific_models=None):
    """Submit jobs for specified tasks, seeds, and architectures"""
    # Create timestamp for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(f"baseline_runs_{timestamp}")
    
    # Create necessary directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/slurm_logs", exist_ok=True)
    
    # Verify data directory exists
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(workspace_dir, 'svrt', 'data')
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory not found at {data_dir}")
        print("Please ensure your SVRT data is in the correct location before running jobs")
        return None, None
    
    # Create directories for specified models
    models_to_run = specific_models if specific_models else ['conv2', 'conv4', 'conv6']
    for model in models_to_run:
        os.makedirs(f"{base_dir}/{model}", exist_ok=True)
    
    # Dictionary to store job IDs
    job_ids = {}
    
    # SVRT tasks to run
    tasks = specific_tasks if specific_tasks else ['1', '7', '5', '15', '16', '19', '20', '21', '22']
    seeds = specific_seeds if specific_seeds else range(num_seeds)
    
    # Submit one job per task and seed
    for task in tasks:
        job_ids[task] = []
        for seed in seeds:
            # Create directories for this task
            for model in models_to_run:
                os.makedirs(f"{base_dir}/{model}/task{task}", exist_ok=True)
            
            # Create Slurm script
            script_content = create_slurm_script(task, seed, base_dir, models_to_run)
            script_path = f"{base_dir}/slurm_logs/task{task}_seed{seed}.sh"
            
            # Write script to file
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # Submit job and get job ID
            try:
                result = subprocess.run(['sbatch', script_path], 
                                     capture_output=True, text=True,
                                     check=True)
                job_id = result.stdout.strip().split()[-1]
                job_ids[task].append(job_id)
                print(f"Submitted task {task} seed {seed} job: {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting task {task} seed {seed} job: {e}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
    
    # Save job IDs for reference
    with open(f"{base_dir}/job_ids.json", 'w') as f:
        json.dump(job_ids, f, indent=4)
    
    return base_dir, job_ids

def compile_results(experiment_dir):
    """Compile and average results across seeds for each model and task"""
    results = {}
    tasks = ['1', '7', '5', '15', '16', '19', '20', '21', '22']
    
    for model_type in ['conv2', 'conv4', 'conv6']:
        model_results = {
            'train_acc_mean': {},
            'train_acc_std': {},
            'val_acc_mean': {},
            'val_acc_std': {},
            'test_acc_mean': {},
            'test_acc_std': {}
        }
        
        # Process each task
        for task in tasks:
            task_train_accs = []
            task_val_accs = []
            task_test_accs = []
            failed_seeds = []
            
            # Collect results from all seeds
            for seed in range(10):
                result_path = f"{experiment_dir}/{model_type}/task{task}/seed{seed}/results.json"
                if os.path.exists(result_path):
                    try:
                        with open(result_path, 'r') as f:
                            seed_results = json.load(f)
                            task_train_accs.append(seed_results['train_acc'][-1])
                            task_val_accs.append(seed_results['val_acc'][-1])
                            task_test_accs.append(seed_results['test_acc'])
                    except json.JSONDecodeError:
                        print(f"Warning: Corrupted results file for {model_type} task {task} seed {seed}")
                        failed_seeds.append(seed)
                else:
                    failed_seeds.append(seed)
            
            if failed_seeds:
                print(f"\nWarning: Missing or corrupted results for {model_type} task {task} seeds: {failed_seeds}")
            
            if task_train_accs:
                model_results['train_acc_mean'][task] = float(np.mean(task_train_accs))
                model_results['train_acc_std'][task] = float(np.std(task_train_accs))
                model_results['val_acc_mean'][task] = float(np.mean(task_val_accs))
                model_results['val_acc_std'][task] = float(np.std(task_val_accs))
                model_results['test_acc_mean'][task] = float(np.mean(task_test_accs))
                model_results['test_acc_std'][task] = float(np.std(task_test_accs))
        
        results[model_type] = model_results
    
    # Save compiled results
    output_path = f"{experiment_dir}/compiled_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\nBaseline Results Summary:")
    print("=" * 50)
    for model_type, model_results in results.items():
        print(f"\n{model_type.upper()}:")
        print("\nTest Accuracies:")
        for task in tasks:
            mean = model_results['test_acc_mean'][task]
            std = model_results['test_acc_std'][task]
            print(f"Task {task}: {mean:.3f} ± {std:.3f}")
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', action='store_true', help='Compile results from a previous run')
    parser.add_argument('--experiment_dir', type=str, help='Directory containing experiment results to compile')
    parser.add_argument('--num_seeds', type=int, default=10, help='Number of seeds to run (default: 10)')
    parser.add_argument('--specific_tasks', type=str, nargs='+', help='Specific tasks to run (e.g., "1 7")')
    parser.add_argument('--specific_seeds', type=int, nargs='+', help='Specific seeds to run (e.g., 1 3 4 6)')
    parser.add_argument('--specific_models', type=str, nargs='+', help='Specific models to run (e.g., "conv2")')
    args = parser.parse_args()
    
    if args.compile:
        if args.experiment_dir:
            compile_results(args.experiment_dir)
        else:
            print("Error: Please provide the experiment directory to compile")
            print("Usage: python run_baselines.py --compile --experiment_dir <experiment_dir>")
    else:
        # Submit jobs
        if args.specific_seeds or args.specific_tasks or args.specific_models:
            print(f"Running specific combinations:")
            if args.specific_tasks:
                print(f"Tasks: {args.specific_tasks}")
            if args.specific_seeds:
                print(f"Seeds: {args.specific_seeds}")
            if args.specific_models:
                print(f"Models: {args.specific_models}")
        else:
            print(f"Submitting jobs for {args.num_seeds} seeds...")
            
        experiment_dir, job_ids = submit_jobs(
            num_seeds=args.num_seeds,
            specific_tasks=args.specific_tasks,
            specific_seeds=args.specific_seeds,
            specific_models=args.specific_models
        )
        
        if experiment_dir:
            print(f"\nAll jobs submitted. Experiment directory: {experiment_dir}")
            print("Job IDs have been saved to job_ids.json in the experiment directory")
            print("\nTo compile results after all jobs complete, run:")
            print(f"python {__file__} --compile --experiment_dir {experiment_dir}") 