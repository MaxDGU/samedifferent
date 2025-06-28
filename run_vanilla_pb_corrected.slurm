#!/bin/bash
#SBATCH --job-name=train_vanilla_pb_corrected
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00 
#SBATCH --gres=gpu:1
#SBATCH --output=logs/vanilla_pb_corrected_train_%A_%a.out
#SBATCH --array=0-2

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
OUTPUT_DIR="/scratch/gpfs/mg7411/samedifferent/results/pb_baselines_stable/conv6lr"

# --- Seed Array ---
SEEDS=(42 123 555)
CURRENT_SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "Running job for SEED=${CURRENT_SEED}"

# --- Execution ---
echo "Starting corrected vanilla PB training script..."
python -u baselines/train_vanilla_pb_corrected.py \
    --data_dir "${PB_DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed "${CURRENT_SEED}" \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.0001

echo "Job finished with exit code $? at $(date)" 