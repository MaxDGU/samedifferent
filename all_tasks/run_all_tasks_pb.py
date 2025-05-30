import os
import subprocess
import json
import numpy as np
from itertools import product
from pathlib import Path

# Define architectures and seeds
ARCHITECTURES = ['conv2', 'conv4', 'conv6']
SEEDS = list(range(10)) # Seeds 0 through 9, 10 unique seeds in total
# FIXED_QUERY_SIZE will be handled by MAML script internally

# --- SLURM Script Template for MAML ---
def create_array_script_maml(output_base_dir, base_code_dir_on_cluster, data_dir_on_cluster,
                             epochs=100,
                             patience=20,      # Default updated patience
                             val_freq=5,       # Default updated val_freq
                             slurm_time="03:00:00", # Default updated slurm_time
                             num_workers=4,
                             first_order=True,
                             inner_lr=0.01,
                             outer_lr=0.001,
                             meta_batch_size=4,
                             use_amp=True, # New parameter for AMP
                             test_job_index=None, # New parameter
                             adaptation_steps_train=5, # New default
                             adaptation_steps_test=15,  # New default
                             weight_decay=0.01 # New parameter for weight_decay
                            ):
    """Create a Slurm array script template for MAML (all tasks) jobs with variable S."""

    num_archs = len(ARCHITECTURES)
    num_seeds = len(SEEDS)
    
    if test_job_index is not None:
        array_indices = str(test_job_index)
        total_jobs_for_message = 1
        job_name_suffix = f"_test{test_job_index}"
    else:
        total_jobs = num_archs * num_seeds
        array_indices = f"0-{total_jobs - 1}"
        total_jobs_for_message = total_jobs
        job_name_suffix = "_full_array"

    output_path = Path(output_base_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Ensure slurm log dir name is also unique if testing a single job vs full array
    slurm_log_dir_name = f"slurm_logs_maml_Svar{job_name_suffix.replace('_', '-')}" 
    slurm_log_dir = output_path / slurm_log_dir_name
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    archs_str = " ".join(ARCHITECTURES)
    seeds_str = " ".join(map(str, SEEDS))

    maml_script_args = [
        f"--data_dir {data_dir_on_cluster}",
        f"--output_base_dir {output_base_dir}",
        f"--epochs {epochs}",
        f"--patience {patience}",
        f"--val_freq {val_freq}",
        f"--num_workers {num_workers}",
        f"--inner_lr {inner_lr}",
        f"--outer_lr {outer_lr}",
        f"--meta_batch_size {meta_batch_size}",
        f"--adaptation_steps_train {adaptation_steps_train}",
        f"--adaptation_steps_test {adaptation_steps_test}",
        f"--weight_decay {weight_decay}"
    ]
    if not first_order:
        pass
    else:
        maml_script_args.append("--first_order")
    
    if use_amp:
        maml_script_args.append("--use_amp")

    maml_script_args_str = " \\\n    ".join(maml_script_args)

    script = f"""#!/bin/bash
#SBATCH --job-name=pb_maml_Svar{job_name_suffix}
#SBATCH --output={slurm_log_dir}/slurm_%A_%a.out
#SBATCH --error={slurm_log_dir}/slurm_%A_%a.err
#SBATCH --time={slurm_time} # Use passed slurm_time
#SBATCH --mem=16G
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={num_workers}
#SBATCH --array={array_indices}

declare -a ARCHS=({archs_str})
declare -a SEEDS=({seeds_str})

NUM_SEEDS={num_seeds}

arch_idx=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
seed_idx=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

ARCH=${{ARCHS[$arch_idx]}}
SEED_VAL=${{SEEDS[$seed_idx]}}

echo "Loading modules..."
module load anaconda3/2024.2
module load cuda/11.7.0

echo "Activating conda environment..."
conda activate tensorflow

BASE_CODE_DIR="{base_code_dir_on_cluster}"  
MAML_SCRIPT_PATH="${{BASE_CODE_DIR}}/experiment_all_tasks_fomaml.py"

echo "Running MAML job (variable S) for arch=$ARCH seed=$SEED_VAL (Array Job ID: $SLURM_ARRAY_TASK_ID)"
python ${{MAML_SCRIPT_PATH}} \
    --architecture $ARCH \
    --seed $SEED_VAL \
    {maml_script_args_str}

echo "MAML job (variable S) finished for arch=$ARCH seed=$SEED_VAL (Array Job ID: $SLURM_ARRAY_TASK_ID)"
"""

    script_name = f'run_maml_Svar{job_name_suffix}_array.sh'
    script_path = output_path / script_name
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path, total_jobs_for_message # Return number of jobs for message

# --- Job Submission ---
def submit_array_job(script_path):
    """Submits the generated Slurm array script."""
    try:
        cmd = ['sbatch', str(script_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        array_job_id = result.stdout.strip().split()[-1]
        # print(f'Submitted MAML array job {array_job_id}') # Message handled by main
        return array_job_id
    except subprocess.CalledProcessError as e:
        print(f'Error submitting MAML array job: {e}')
        print(f'Stderr: {e.stderr}')
        return None
    except FileNotFoundError:
         print(f"Error: 'sbatch' command not found. Are you on a Slurm system?")
         return None

# --- Main Execution ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run MAML training (variable S) on all PB tasks for multiple architectures and seeds via Slurm.")
    
    script_dir = Path(__file__).parent.resolve()
    default_code_dir = script_dir.parent
    default_output_dir = default_code_dir / 'results' / 'pb_maml_Svar_Q3_runs' # Updated default output dir name
    default_data_dir = default_code_dir / 'data' / 'pb' / 'pb'

    parser.add_argument('--output_dir', type=str, default=str(default_output_dir),
                        help="Base directory where MAML experiment folders will be created.")
    parser.add_argument('--base_code_dir_cluster', type=str, default=str(default_code_dir),
                         help="Absolute path to the root of your code repository on the Slurm cluster.")
    parser.add_argument('--data_dir_cluster', type=str, default=str(default_data_dir),
                         help="Absolute path to the PB HDF5 data files on the Slurm cluster.")
    parser.add_argument('--test_job_index', type=int, default=None,
                        help="If specified, runs only the Slurm array job with this index (e.g., 0 for the first job). Otherwise, runs all jobs.")
    parser.add_argument('--patience', type=int, default=20,
                        help="Patience for early stopping in MAML script. Default: 20.")
    parser.add_argument('--val_freq', type=int, default=5,
                        help="Validation frequency in MAML script (epochs). Default: 5.")
    parser.add_argument('--slurm_time', type=str, default="03:00:00",
                        help="Slurm time limit (e.g., HH:MM:SS). Default: 03:00:00.")
    parser.add_argument('--use_amp', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Use Automatic Mixed Precision (AMP) in MAML script. Default: True.")
    parser.add_argument('--first_order', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help="Use First-Order MAML (FOMAML). Default: False (uses 2nd order MAML).")
    parser.add_argument('--inner_lr', type=float, default=0.001, help="Inner loop learning rate for MAML. Default: 0.001")
    parser.add_argument('--outer_lr', type=float, default=0.0001, help="Outer loop learning rate for MAML. Default: 0.0001")
    parser.add_argument('--meta_batch_size', type=int, default=16, help="Meta batch size for MAML. Default: 16")
    parser.add_argument('--adaptation_steps_train', type=int, default=5, help="Number of adaptation steps during meta-training. Default: 5")
    parser.add_argument('--adaptation_steps_test', type=int, default=15, help="Number of adaptation steps during meta-testing. Default: 15")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay for the optimizer. Default: 0.01")
    
    args = parser.parse_args()

    cluster_output_dir_for_maml_experiments = Path(args.output_dir).resolve()
    base_code_dir_on_cluster = Path(args.base_code_dir_cluster).resolve()
    data_dir_on_cluster = Path(args.data_dir_cluster).resolve()

    print(f"Output directory for MAML experiments (on cluster): {cluster_output_dir_for_maml_experiments}")
    print(f"Base code directory (on cluster): {base_code_dir_on_cluster}")
    print(f"Data directory (on cluster): {data_dir_on_cluster}")
    
    if args.test_job_index is not None:
        print(f"Generating Slurm script for a SINGLE MAML TEST JOB (index: {args.test_job_index})...")
    else:
        print("Generating Slurm array script for FULL MAML (variable S) run...")
    
    print(f"MAML runs will use: Patience={args.patience}, Val Freq={args.val_freq}, Slurm Time={args.slurm_time}, Use AMP={args.use_amp}, First Order={args.first_order}")
    print("MAML runs will internally sample from support sizes [4,6,8,10] and use fixed query size 3.")

    script_path, num_jobs_to_run = create_array_script_maml(
        output_base_dir=str(cluster_output_dir_for_maml_experiments),
        base_code_dir_on_cluster=str(base_code_dir_on_cluster),
        data_dir_on_cluster=str(data_dir_on_cluster),
        patience=args.patience,
        val_freq=args.val_freq,
        slurm_time=args.slurm_time,
        use_amp=args.use_amp,
        first_order=args.first_order,
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        meta_batch_size=args.meta_batch_size,
        test_job_index=args.test_job_index,
        adaptation_steps_train=args.adaptation_steps_train,
        adaptation_steps_test=args.adaptation_steps_test,
        weight_decay=args.weight_decay
    )
    print(f"MAML Slurm script created at: {script_path}")
    
    job_id = submit_array_job(script_path)
    if job_id:
        if args.test_job_index is not None:
            print(f"Submitted SINGLE MAML test job {job_id} (Slurm Array Task ID: {args.test_job_index}).")
        else:
            print(f"Submitted MAML array job {job_id} for {num_jobs_to_run} tasks.")
        print(f"Run 'squeue -u $USER' to check job status.")
        print(f"Individual MAML experiment folders will be created in: {cluster_output_dir_for_maml_experiments}")
    else:
        print("MAML job submission failed.") 