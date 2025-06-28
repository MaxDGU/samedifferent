#!/bin/bash
#SBATCH --job-name=vanilla-pb-retrain
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/vanilla_pb_retrain_%A_%a.out

# --- Environment Setup ---
echo "Setting up the environment..."
source /usr/share/Modules/init/bash
module load anaconda3/2023.9
conda activate tensorflow
echo "Conda environment activated."

# --- Array Job Setup ---
SEEDS=(42 123 555 789 999)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Running job for SEED=${SEED}"

# --- Set Project Path ---
export GIT_REPO_PATH="/scratch/gpfs/mg7411/samedifferent"
export PYTHONPATH="${PYTHONPATH}:${GIT_REPO_PATH}"
cd "${GIT_REPO_PATH}"
echo "Current working directory: $(pwd)"

# --- Execution ---
echo "Starting Vanilla PB re-training with new conv6lr architecture for seed ${SEED}..."
python -u baselines/train_pb_models_new_arch.py \
    --test_task regular \
    --architecture conv6 \
    --seed ${SEED} \
    --batch_size 32 \
    --lr 0.0001 \
    --epochs 150 \
    --patience 10 \
    --val_freq 5 \
    --data_dir /scratch/gpfs/mg7411/data/pb/pb \
    --output_dir /scratch/gpfs/mg7411/samedifferent/results/pb_baselines_retrained

echo "Job finished with exit code $? at $(date)" 