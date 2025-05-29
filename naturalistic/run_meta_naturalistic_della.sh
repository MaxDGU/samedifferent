#!/bin/bash
#SBATCH --job-name=meta_nat_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # Number of CPUs, can be num_workers for DataLoader + 1
#SBATCH --mem=16G
#SBATCH --time=08:00:00      # Adjust as needed, e.g., 8 hours
#SBATCH --gres=gpu:1         # Request 1 GPU
#SBATCH --output=slurm_logs/naturalistic/meta_train_%A_%a.out
#SBATCH --error=slurm_logs/naturalistic/meta_train_%A_%a.err

# --- User Configuration ---
NETID="mg7411" # !!! IMPORTANT: REPLACE YOUR_NETID !!!

# --- Directory Setup ---
# Assuming this script is in remote/naturalistic/ and run from project root or remote/
# Or, provide absolute paths for CLUSTER_CODE_BASE and PYTHON_SCRIPT_REL_PATH
PROJECT_ROOT_ON_CLUSTER="/scratch/gpfs/${NETID}" # Base for code and output
PYTHON_SCRIPT_NAME="meta_naturalistic_train.py" # Name of the training script
# Location of the training script relative to PROJECT_ROOT_ON_CLUSTER
# This assumes meta_naturalistic_train.py and convX.py models are directly in PROJECT_ROOT_ON_CLUSTER
PYTHON_SCRIPT_PATH="${PROJECT_ROOT_ON_CLUSTER}/${PYTHON_SCRIPT_NAME}"

DATA_DIR_ON_CLUSTER="${PROJECT_ROOT_ON_CLUSTER}/data/naturalistic/meta"
BASE_OUTPUT_DIR_ON_CLUSTER="${PROJECT_ROOT_ON_CLUSTER}/results/naturalistic/528_run"

SLURM_LOG_BASE_DIR="${PROJECT_ROOT_ON_CLUSTER}/slurm_logs/naturalistic"
mkdir -p "${SLURM_LOG_BASE_DIR}" # Ensure Slurm log directory exists

# --- Parameters for meta_naturalistic_train.py ---
EPOCHS=100
EPISODES_PER_EPOCH=500
META_LR=0.001
INNER_LR=0.05       # As per meta_naturalistic_train.py default
INNER_STEPS=5       # As per meta_naturalistic_train.py default
WEIGHT_DECAY=0.01   # Consistent with PB runs
NUM_WORKERS=2       # For DataLoader on cluster
PATIENCE=15         # Early stopping patience

# --- Array Job Setup ---
ARCHITECTURES=("conv2lr" "conv4lr" "conv6lr")
SEEDS=(0 1 2 3 4 5 6 7 8 9)

NUM_ARCHS=${#ARCHITECTURES[@]}
NUM_SEEDS=${#SEEDS[@]}

# Total jobs = NUM_ARCHS * NUM_SEEDS. Array index from 0 to (total_jobs - 1)
TOTAL_JOBS=$((NUM_ARCHS * NUM_SEEDS))
#SBATCH --array=0-$((TOTAL_JOBS - 1))

# Calculate architecture and seed for the current Slurm array task ID
ARCH_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

CURRENT_ARCH=${ARCHITECTURES[$ARCH_INDEX]}
CURRENT_SEED=${SEEDS[$SEED_INDEX]}

# --- Construct Output Directory for this specific job ---
# This will be passed as --log_dir to the python script
JOB_OUTPUT_DIR="${BASE_OUTPUT_DIR_ON_CLUSTER}/${CURRENT_ARCH}/seed_${CURRENT_SEED}"
mkdir -p "${JOB_OUTPUT_DIR}"

# --- Environment Setup ---
echo "Loading modules..."
module load anaconda3/2023.9 # Or your preferred Anaconda module on Della
module load cuda/11.8      # Or your preferred CUDA module

# Activate your conda environment
CONDA_ENV_NAME="tensorflow" # Replace with your environment name
echo "Activating conda environment: ${CONDA_ENV_NAME}..."
source activate "${CONDA_ENV_NAME}"

# --- Run the Python Training Script ---
echo "Starting Python script..."
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running Architecture: ${CURRENT_ARCH}, Seed: ${CURRENT_SEED}"
echo "Outputting to: ${JOB_OUTPUT_DIR}"

cd "${PROJECT_ROOT_ON_CLUSTER}" # Change to the directory containing the script and models

python "${PYTHON_SCRIPT_NAME}" \
    --model "${CURRENT_ARCH}" \
    --seed "${CURRENT_SEED}" \
    --data_dir "${DATA_DIR_ON_CLUSTER}" \
    --log_dir "${JOB_OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --episodes_per_epoch "${EPISODES_PER_EPOCH}" \
    --meta_lr "${META_LR}" \
    --inner_lr "${INNER_LR}" \
    --inner_steps "${INNER_STEPS}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --num_workers "${NUM_WORKERS}" \
    --patience "${PATIENCE}" \
    --device cuda

EXIT_CODE=$?
echo "Python script finished with exit code $EXIT_CODE for Arch: ${CURRENT_ARCH}, Seed: ${CURRENT_SEED}."

conda deactivate
echo "Job finished." 