#!/bin/bash
#SBATCH --job-name=meta_nat_exp  # Job name
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Number of tasks (MPI processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=32G                # Memory per node
#SBATCH --time=24:00:00          # Walltime limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # Number of GPUs per node
#SBATCH --output=slurm_logs/meta_nat_exp_%A_%a.out # Standard output and error log, %A is job ID, %a is array task ID
#SBATCH --array=0-8              # Array job: 3 seeds * 3 architectures = 9 jobs (0-8)

# --- Environment Setup ---
echo "Setting up environment..."
source /usr/share/Modules/init/bash # Initialize environment modules
module purge                     # Remove any inherited modules
module load anaconda3/2023.9      # Load Anaconda
source activate samediff_env       # Activate your conda environment
echo "Conda environment activated: $CONDA_DEFAULT_ENV"
echo "Python executable: $(which python)"

# --- Project Setup ---
PROJECT_DIR="/scratch/gpfs/mg7411/samedifferent" # CHANGE THIS TO YOUR PROJECT DIRECTORY ON DELLA
SCRIPT_PATH="${PROJECT_DIR}/naturalistic/meta_naturalistic_train.py"
DATA_DIR="${PROJECT_DIR}/data/naturalistic"
LOG_BASE_DIR="${PROJECT_DIR}/logs_naturalistic_meta" # Base directory for logs from these experiments

# --- Parameters ---
SEEDS=(42 123 789)
ARCHITECTURES=("conv2lr" "conv4lr" "conv6lr")

# Calculate seed and architecture for this array task
NUM_SEEDS=${#SEEDS[@]}
NUM_ARCHS=${#ARCHITECTURES[@]}

SEED_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_ARCHS))
ARCH_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_ARCHS))

SEED=${SEEDS[$SEED_INDEX]}
ARCH=${ARCHITECTURES[$ARCH_INDEX]}

echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running with Seed: ${SEED}, Architecture: ${ARCH}"

# Create a unique log directory for this specific run
RUN_LOG_DIR="${LOG_BASE_DIR}/${ARCH}/seed_${SEED}"
mkdir -p "${RUN_LOG_DIR}"
echo "Logging to: ${RUN_LOG_DIR}"

# --- Run Training Script ---
echo "Starting meta-training script..."
python "${SCRIPT_PATH}" \
    --model "${ARCH}" \
    --seed "${SEED}" \
    --data_dir "${DATA_DIR}" \
    --log_dir "${RUN_LOG_DIR}" \
    --device "cuda" \
    --epochs 60 \
    --episodes_per_epoch 1000 \
    --meta_lr 1e-5 \
    --inner_lr 1e-5 \
    --patience 20 \
    --grad_clip_norm 1.0 \
    --val_interval 1 \
    --checkpoint_interval 5 \
    --log_interval 100 \
    # Add any other parameters you want to control from here
    # --first_order \ # Uncomment to use first-order MAML

echo "Script finished."
date

# --- Example: How to submit this script ---
# sbatch run_naturalistic_meta_experiments_della.slurm
#
# To check job status:
# squeue -u $USER
#
# To cancel a job:
# scancel <JOB_ID>
#
# To cancel all jobs in an array:
# scancel <ARRAY_JOB_ID>
#
# To get info about a running/finished job:
# sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,ExitCode,MaxRSS,AllocTRES%30
#
# To view output:
# Check the slurm_logs directory specified in #SBATCH --output 