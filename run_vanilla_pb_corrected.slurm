#!/bin/bash
#SBATCH --job-name=train_vanilla_pb_corrected
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00 
#SBATCH --gres=gpu:1
#SBATCH --output=logs/vanilla_pb_corrected_train_%A_%a.out
#SBATCH --array=0-5 # (2 test tasks * 3 seeds) - 1

# --- Environment Setup ---
echo "Setting up the environment..."
source /usr/share/Modules/init/bash
module load anaconda3/2023.9
conda activate tensorflow
echo "Conda environment activated."

# --- Project and Data Paths ---
export GIT_REPO_PATH="/scratch/gpfs/mg7411/samedifferent"
export PYTHONPATH="${PYTHONPATH}:${GIT_REPO_PATH}"
cd "${GIT_REPO_PATH}"
echo "Current working directory: $(pwd)"

PB_DATA_DIR="/scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb"
OUTPUT_DIR="/scratch/gpfs/mg7411/samedifferent/results/pb_baselines_stable" # Parent directory

# --- Job Array Parameters ---
TEST_TASKS=("regular" "lines")
SEEDS=(42 123 555)

# Calculate parameters for this job
TASK_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
CURRENT_TEST_TASK=${TEST_TASKS[$TASK_INDEX]}
CURRENT_SEED=${SEEDS[$SEED_INDEX]}

echo "Running job for test_task=${CURRENT_TEST_TASK}, seed=${CURRENT_SEED}"

# --- Execution ---
echo "Starting corrected vanilla PB training script..."
python -u baselines/train_vanilla_pb_corrected.py \
    --data_dir "${PB_DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --test_task "${CURRENT_TEST_TASK}" \
    --seed "${CURRENT_SEED}" \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --patience 3 \
    --val_freq 10 \
    --improvement_threshold 0.02

echo "Job finished with exit code $? at $(date)" 