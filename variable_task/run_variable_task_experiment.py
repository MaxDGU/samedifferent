import os
import subprocess
from pathlib import Path
import json

# --- Experiment Configuration ---
# The specific architecture to use for this experiment
ARCHITECTURE = 'conv6'
# All available PB tasks to select from
ALL_PB_TASKS = [
    'regular', 'lines', 'open', 'wider_line', 'scrambled',
    'random_color', 'arrows', 'irregular', 'filled', 'original'
]
# Seeds to run for each task subset
SEEDS = [123, 42, 555, 789, 999] # 5 seeds
# Number of tasks to use in each meta-training run
NUM_TASKS_TO_RUN = list(range(1, len(ALL_PB_TASKS) + 1)) # 1 to 10 tasks

# --- SLURM Script Template ---
def create_slurm_script(output_base_dir, base_code_dir_on_cluster, data_dir_on_cluster,
                          num_tasks, arch, seed,
                          epochs=100,
                          patience=20,
                          val_freq=5,
                          slurm_time="04:00:00",
                          num_workers=4,
                          first_order=True,
                          inner_lr=0.01,
                          outer_lr=0.001,
                          meta_batch_size=4,
                          use_amp=True,
                          adaptation_steps_train=5,
                          adaptation_steps_test=15,
                          weight_decay=0.01):
    """Creates a single Slurm job script for one experiment run."""
    
    # Select a subset of tasks for this run
    # To ensure consistency, we'll use a fixed ordering of tasks
    tasks_for_run = ALL_PB_TASKS[:num_tasks]
    tasks_str = " ".join(tasks_for_run)

    job_name = f"var_tasks_{arch}_s{seed}_{num_tasks}t"
    
    # Define a unique output directory for this specific job's Slurm logs
    slurm_log_dir = Path(output_base_dir) / "slurm_logs" / job_name
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    # The actual training script to be called
    training_script_path = Path(base_code_dir_on_cluster) / "variable_task" / "train_variable_task.py"

    script_args = [
        f"--data_dir {data_dir_on_cluster}",
        f"--output_base_dir {output_base_dir}",
        f"--architecture {arch}",
        f"--seed {seed}",
        f"--tasks {tasks_str}",
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
    if first_order:
        script_args.append("--first_order")
    if use_amp:
        script_args.append("--use_amp")

    script_args_str = " \\\n    ".join(script_args)

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={slurm_log_dir}/slurm_%j.out
#SBATCH --error={slurm_log_dir}/slurm_%j.err
#SBATCH --time={slurm_time}
#SBATCH --mem=16G
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={num_workers}

echo "Loading modules..."
module load anaconda3/2024.2
module load cuda/11.7.0

echo "Activating conda environment..."
conda activate tensorflow

echo "Running variable task training job for: Arch={arch}, Seed={seed}, NumTasks={num_tasks}"
python {training_script_path} \\
    {script_args_str}

echo "Job finished."
"""
    
    # Save the script to a file
    script_path = slurm_log_dir / f"{job_name}.sh"
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path

# --- Job Submission ---
def submit_job(script_path):
    """Submits the generated Slurm script."""
    try:
        cmd = ['sbatch', str(script_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted job {job_id} from script: {script_path}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job from script {script_path}: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: 'sbatch' command not found. Are you on a Slurm login node?")
        return None

# --- Main Execution ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run variable-task meta-training experiments.")
    
    script_dir = Path(__file__).parent.resolve()
    default_code_dir = script_dir.parent
    default_output_dir = default_code_dir / 'results' / 'variable_task_experiments'
    default_data_dir = default_code_dir / 'data' / 'meta_h5' / 'pb'

    parser.add_argument('--output_dir', type=str, default=str(default_output_dir),
                        help="Base directory for experiment outputs.")
    parser.add_argument('--base_code_dir_cluster', type=str, default=str(default_code_dir),
                        help="Absolute path to the code repository on the Slurm cluster.")
    parser.add_argument('--data_dir_cluster', type=str, default=str(default_data_dir),
                        help="Absolute path to the PB HDF5 data on the Slurm cluster.")
    parser.add_argument('--dry_run', action='store_true',
                        help="Generate scripts but do not submit them.")

    args = parser.parse_args()

    output_base = Path(args.output_dir).resolve()
    code_base_cluster = Path(args.base_code_dir_cluster).resolve()
    data_dir_cluster = Path(args.data_dir_cluster).resolve()

    print(f"Output directory: {output_base}")
    print(f"Code directory on cluster: {code_base_cluster}")
    print(f"Data directory on cluster: {data_dir_cluster}")

    submitted_jobs = []
    for num_tasks in NUM_TASKS_TO_RUN:
        for seed in SEEDS:
            print(f"\n--- Preparing job: {ARCHITECTURE}, {seed=}, {num_tasks=} tasks ---")
            
            script_path = create_slurm_script(
                output_base_dir=output_base,
                base_code_dir_on_cluster=code_base_cluster,
                data_dir_on_cluster=data_dir_cluster,
                num_tasks=num_tasks,
                arch=ARCHITECTURE,
                seed=seed
            )

            print(f"Generated Slurm script: {script_path}")

            if not args.dry_run:
                job_id = submit_job(script_path)
                if job_id:
                    submitted_jobs.append(job_id)
            else:
                print("DRY RUN: Skipping job submission.")

    print("\n--- Summary ---")
    if not args.dry_run:
        print(f"Total jobs submitted: {len(submitted_jobs)}")
        if submitted_jobs:
            print("Submitted job IDs:", " ".join(submitted_jobs))
        print(f"Check job status with: squeue -u $USER")
    else:
        print("DRY RUN complete. No jobs were submitted.")
    print(f"All outputs will be located under: {output_base}") 