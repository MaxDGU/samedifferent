import os
import subprocess
from pathlib import Path
import itertools

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
NUM_TASKS_LEVELS = list(range(1, len(ALL_PB_TASKS) + 1)) # 1 to 10 tasks

# --- SLURM Array Script ---
def create_array_script(output_base_dir, base_code_dir_on_cluster, data_dir_on_cluster,
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
    """Creates a single Slurm array script for all experiment runs."""
    
    num_seeds = len(SEEDS)
    num_task_levels = len(NUM_TASKS_LEVELS)
    total_jobs = num_seeds * num_task_levels

    job_name = f"var_tasks_{ARCHITECTURE}"
    slurm_log_dir = Path(output_base_dir) / "slurm_logs" / job_name
    slurm_log_dir.mkdir(parents=True, exist_ok=True)

    training_script_path = Path(base_code_dir_on_cluster) / "variable_task" / "train_variable_task.py"

    # Convert Python lists to bash array strings
    seeds_str = " ".join(map(str, SEEDS))
    num_tasks_levels_str = " ".join(map(str, NUM_TASKS_LEVELS))
    all_pb_tasks_str = " ".join(ALL_PB_TASKS)

    script_args = [
        f"--data_dir {data_dir_on_cluster}",
        f"--output_base_dir {output_base_dir}",
        f"--architecture {ARCHITECTURE}",
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
#SBATCH --array=0-{total_jobs - 1}
#SBATCH --output={slurm_log_dir}/slurm_%A_%a.out
#SBATCH --error={slurm_log_dir}/slurm_%A_%a.err
#SBATCH --time={slurm_time}
#SBATCH --mem=16G
#SBATCH --partition=pli
#SBATCH --account=nam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={num_workers}

# --- Bash Arrays for Seeds and Task Counts ---
declare -a SEEDS=({seeds_str})
declare -a NUM_TASKS_LEVELS=({num_tasks_levels_str})
declare -a ALL_PB_TASKS=({all_pb_tasks_str})

# --- Task Indexing ---
NUM_SEEDS={num_seeds}
seed_idx=$((SLURM_ARRAY_TASK_ID / {num_task_levels}))
task_level_idx=$((SLURM_ARRAY_TASK_ID % {num_task_levels}))

SEED=${{SEEDS[$seed_idx]}}
NUM_TASKS=${{NUM_TASKS_LEVELS[$task_level_idx]}}

# --- Select Tasks ---
TASKS_FOR_RUN=("${{ALL_PB_TASKS[@]:0:$NUM_TASKS}}")

echo "--- Job Details ---"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Architecture: {ARCHITECTURE}"
echo "Seed: $SEED"
echo "Number of Tasks: $NUM_TASKS"
echo "Tasks: ${{TASKS_FOR_RUN[*]}}"
echo "-------------------"

# --- Environment Setup ---
echo "Loading modules..."
module load anaconda3/2024.2
module load cuda/11.7.0

echo "Activating conda environment..."
conda activate tensorflow

# Fix for ModuleNotFoundError by adding project root to PYTHONPATH
echo "Setting PYTHONPATH..."
export PYTHONPATH={base_code_dir_on_cluster}:$PYTHONPATH

# --- Run Training Script ---
echo "Running variable task training job..."
python {training_script_path} \\
    --seed $SEED \\
    --tasks ${{TASKS_FOR_RUN[*]}} \\
    {script_args_str}

echo "Job finished."
"""
    
    script_path = slurm_log_dir / f"{job_name}_array.sh"
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)
    return script_path, total_jobs

# --- Job Submission ---
def submit_job(script_path):
    """Submits the generated Slurm script."""
    try:
        cmd = ['sbatch', str(script_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split()[-1]
        print(f"Submitted array job {job_id} from script: {script_path}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job from script {script_path}: {e}\nStderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: 'sbatch' command not found. Are you on a Slurm login node?")
        return None

# --- Main Execution ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run variable-task meta-training experiments using a Slurm array job.")
    
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
    parser.add_argument('--test', action='store_true',
                        help="Run only the first job in the array for testing purposes.")

    args = parser.parse_args()

    output_base = Path(args.output_dir).resolve()
    code_base_cluster = Path(args.base_code_dir_cluster).resolve()
    data_dir_cluster = Path(args.data_dir_cluster).resolve()

    print(f"Generating Slurm array script...")
    print(f"Output directory: {output_base}")
    print(f"Code directory on cluster: {code_base_cluster}")
    print(f"Data directory on cluster: {data_dir_cluster}")

    script_path, total_jobs = create_array_script(
        output_base_dir=output_base,
        base_code_dir_on_cluster=code_base_cluster,
        data_dir_on_cluster=data_dir_cluster
    )

    print(f"Generated Slurm array script: {script_path}")
    
    if args.test:
        print("--- TEST MODE ---")
        print("Modifying script to run only job #0.")
        # Read the generated script
        with open(script_path, 'r') as f:
            lines = f.readlines()
        # Find and replace the #SBATCH --array line
        for i, line in enumerate(lines):
            if line.strip().startswith("#SBATCH --array"):
                lines[i] = "#SBATCH --array=0\n"
                break
        # Write the modified script back
        with open(script_path, 'w') as f:
            f.writelines(lines)
        print(f"Modified {script_path} for a single test job.")
        total_jobs = 1

    print(f"This will submit a single array job with {total_jobs} tasks.")

    if not args.dry_run:
        job_id = submit_job(script_path)
        if job_id:
            print(f"\nSuccessfully submitted array job {job_id}.")
            print(f"Check job status with: squeue -j {job_id}")
    else:
        print("\nDRY RUN complete. No job was submitted.")
    
    print(f"All outputs will be located under: {output_base}") 