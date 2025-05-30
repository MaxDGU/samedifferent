import os
import subprocess
import json
import numpy as np
from itertools import product
from pathlib import Path

# Define tasks, architectures, and seeds
PB_TASKS = ['regular', 'lines', 'open', 'wider_line', 'scrambled',
            'random_color', 'arrows', 'irregular', 'filled', 'original']
ARCHITECTURES = ['conv2', 'conv4', 'conv6']
# Use 10 seeds
SEEDS = list(range(10)) # Seeds 0 through 9

# --- SLURM Script Template ---
def create_array_script(output_base_dir, base_code_dir, data_dir):
    """Create a Slurm array script template for single-task jobs."""

    num_tasks = len(PB_TASKS)
    num_archs = len(ARCHITECTURES)
    num_seeds = len(SEEDS)
    total_jobs = num_tasks * num_archs * num_seeds # 10 * 3 * 10 = 300
    array_indices = f"0-{total_jobs - 1}" # 0-299

    # Ensure base output dir exists
    output_path = Path(output_base_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    slurm_log_dir = output_path / "slurm_logs"
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    # Escape curly braces intended for bash by doubling them {{ }}
    # Use f-string interpolation for Python variables
    # Ensure bash arrays are correctly formatted within the f-string
    tasks_str = " ".join(PB_TASKS)
    archs_str = " ".join(ARCHITECTURES)
    original_seeds_str = " ".join(map(str, SEEDS)) # String of "0 1 2 ... 9"

    script = f"""#!/bin/bash
#SBATCH --job-name=pb_single_task_drift
#SBATCH --output={slurm_log_dir}/slurm_%A_%a.out
#SBATCH --error={slurm_log_dir}/slurm_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array={array_indices}

declare -a TASKS=({tasks_str})
declare -a ARCHS=({archs_str})
declare -a ORIGINAL_SEEDS=({original_seeds_str}) # Represents the 0-9 run index for each task

NUM_ARCHS={num_archs}
NUM_ORIGINAL_SEEDS={num_seeds} # This is 10

# Calculate indices for task, architecture, and the original seed run (0-9)
task_idx=$((SLURM_ARRAY_TASK_ID / (NUM_ARCHS * NUM_ORIGINAL_SEEDS)))
arch_idx=$(((SLURM_ARRAY_TASK_ID % (NUM_ARCHS * NUM_ORIGINAL_SEEDS)) / NUM_ORIGINAL_SEEDS))
original_seed_run_idx=$((SLURM_ARRAY_TASK_ID % NUM_ORIGINAL_SEEDS))

TASK=${{TASKS[$task_idx]}}
ARCH=${{ARCHS[$arch_idx]}}
ORIGINAL_SEED_VAL=${{ORIGINAL_SEEDS[$original_seed_run_idx]}} # This is the 0, 1, ..., 9 value

# Calculate a globally unique seed to pass to train_single_task_pb.py
# This ensures different initial random weights for each task/arch/original_seed_run combination
let "globally_unique_seed=(task_idx * NUM_ORIGINAL_SEEDS) + ORIGINAL_SEED_VAL"

echo "Loading modules..."
module load anaconda3/2024.2
module load cuda/11.7.0

echo "Activating conda environment..."
conda activate tensorflow

BASE_CODE_DIR="{base_code_dir}"
DATA_DIR="{data_dir}"
OUTPUT_DIR="{output_base_dir}"

# The --seed argument to train_single_task_pb.py now uses globally_unique_seed
# The output directory structure will still use the ORIGINAL_SEED_VAL for the seed_X folder name
# This is because train_single_task_pb.py constructs its output path using args.seed
# which will now be the globally_unique_seed. We need to adjust this if we want seed_0, seed_1... folders
# For now, let's pass globally_unique_seed and the folders will be like seed_0, seed_1, ... seed_10, seed_11 ...

echo "Running job for task=$TASK arch=$ARCH original_seed_run_idx=$ORIGINAL_SEED_VAL (passed as --seed $globally_unique_seed to script)"
python ${{BASE_CODE_DIR}}/train_single_task_pb.py \\
    --task $TASK \\
    --architecture $ARCH \\
    --seed $globally_unique_seed \\
    --data_dir $DATA_DIR \\
    --output_dir $OUTPUT_DIR \\
    --epochs 100 \\
    --patience 10 \\
    --val_freq 5 \\
    --improvement_threshold 0.005

echo "Job finished for task=$TASK arch=$ARCH original_seed_run_idx=$ORIGINAL_SEED_VAL (passed as --seed $globally_unique_seed)"
"""

    script_path = Path(output_base_dir) / 'run_single_task_array_drift.sh' # New script name
    with open(script_path, 'w') as f:
        f.write(script)

    # Make the script executable
    os.chmod(script_path, 0o755)

    return script_path

# --- Job Submission --- (Simplified for direct array submission)
def submit_array_job(script_path):
    """Submits the generated Slurm array script."""
    try:
        cmd = ['sbatch', str(script_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        array_job_id = result.stdout.strip().split()[-1]
        print(f'Submitted array job {array_job_id}')
        return array_job_id
    except subprocess.CalledProcessError as e:
        print(f'Error submitting array job: {e}')
        print(f'Stderr: {e.stderr}')
        return None
    except FileNotFoundError:
         print(f"Error: 'sbatch' command not found. Are you on a Slurm system?")
         return None

# --- Results Compilation ---
def compile_results(output_dir, save_path=None):
    """Compile results from single-task training runs."""
    results = {}
    missing_results = False
    output_dir = Path(output_dir)

    print(f"Compiling results from: {output_dir}")

    for task in PB_TASKS:
        results[task] = {}
        for arch in ARCHITECTURES:
            accuracies = []
            for seed in SEEDS:
                # Path structure: output_dir / task / arch / seed_X / metrics.json
                metrics_path = output_dir / task / arch / f'seed_{seed}' / 'metrics.json'
                try:
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        # Use test accuracy from the metrics file
                        accuracies.append(metrics['test_acc'])
                except FileNotFoundError:
                    # Don't print warning if seeds haven't run yet
                    # print(f'Warning: No results found for {task} {arch} seed {seed} at {metrics_path}')
                    missing_results = True
                except KeyError:
                     print(f"Warning: 'test_acc' key not found in {metrics_path}")
                     missing_results = True
                except Exception as e:
                     print(f"Error reading {metrics_path}: {e}")
                     missing_results = True

            if accuracies:
                results[task][arch] = {
                    'mean': float(np.mean(accuracies)),
                    'std': float(np.std(accuracies)),
                    'count': len(accuracies),
                    'individual_accuracies': accuracies
                }
            else:
                 # Indicate no results found for this combo yet
                 results[task][arch] = {'mean': 0, 'std': 0, 'count': 0, 'individual_accuracies': []}

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Compiled results saved to {save_path}")
        except Exception as e:
             print(f"Error saving compiled results: {e}")

    return results, not missing_results # Return status indicating if all expected results were found

# --- Main Execution ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Define directories relative to the script location or use absolute paths
    script_dir = Path(__file__).parent.resolve()
    # Default output dir relative to script location
    default_output = script_dir / 'results' / 'pb_single_task'
    # Default code dir assumes script is in remote/baselines
    default_code_dir = script_dir.parent.parent
    # Example default data path - USER MUST VERIFY/CHANGE THIS FOR CLUSTER
    default_data_dir = default_code_dir / 'data' / 'pb' / 'pb'

    parser.add_argument('--output_dir', type=str, default=str(default_output),
                        help="Base directory to save results (task/model/seed folders will be created within)")
    parser.add_argument('--base_code_dir', type=str, default=str(default_code_dir),
                         help="Path to the root of your code repository (e.g., /scratch/gpfs/user/)")
    parser.add_argument('--data_dir', type=str, default=str(default_data_dir),
                         help="Path to the PB HDF5 data files")
    parser.add_argument('--compile_only', action='store_true',
                      help='Only compile results without submitting new jobs')
    parser.add_argument('--force_compile', action='store_true',
                        help='Compile results even if some files are missing')

    args = parser.parse_args()

    # **Important**: Replace with your actual base code and data dirs on the cluster if defaults are wrong
    # These paths are used when GENERATING the sbatch script
    # Ensure they are correct absolute paths on the cluster file system
    cluster_base_code_dir = "/scratch/gpfs/mg7411" # CHANGE IF NEEDED
    cluster_data_dir = "/scratch/gpfs/mg7411/data/pb/pb"             # CHANGE IF NEEDED

    # The output directory used for storing results and for compilation
    # This should also be an absolute path on the cluster
    cluster_output_dir = args.output_dir
    print(f"Using Cluster Code Directory: {cluster_base_code_dir}")
    print(f"Using Cluster Data Directory: {cluster_data_dir}")
    print(f"Using Cluster Output Directory: {cluster_output_dir}")

    if not args.compile_only:
        print("Generating Slurm array script for Conv2...")
        script_path = create_array_script(cluster_output_dir, cluster_base_code_dir, cluster_data_dir)
        print(f"Script created at: {script_path}")
        job_id = submit_array_job(script_path)
        if job_id:
            print(f"Run 'squeue -u $USER' to check job status.")
            print(f"Run 'python {Path(__file__).name} --compile_only --output_dir {cluster_output_dir}' later to compile results.")
        else:
            print("Job submission failed.")
    else:
        # Compile results
        print("Compiling results...")
        results, complete = compile_results(
            output_dir=cluster_output_dir,
            save_path=Path(cluster_output_dir) / 'compiled_single_task_results.json'
        )

        if not complete and not args.force_compile:
             print("\nWarning: Some results are missing. Run with --force_compile to compile anyway.")

        # Print summary (optional)
        if results:
            print('\nResults Summary:')
            for task in sorted(results.keys()):
                print(f'\nTask: {task}')
                for arch in sorted(results[task].keys()):
                    res = results[task][arch]
                    if res['count'] > 0:
                         mean = res['mean'] * 100
                         std = res['std'] * 100
                         print(f"  {arch} ({res['count']} seeds): {mean:.2f}% ± {std:.2f}%")
                    else:
                         # Only print if compiling forced or all results are expected
                         if complete or args.force_compile:
                             print(f"  {arch}: No results found.")
        else:
             print("No results found to compile.")