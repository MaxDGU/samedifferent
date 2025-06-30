#!/bin/bash
#SBATCH --job-name=vanilla_pb_corrected
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/vanilla_pb_final_%A_%a.out

# Activate conda environment
source ~/.bashrc
conda activate tensorflow

# Navigate to the project directory
cd /scratch/gpfs/mg7411/samedifferent

echo "Setting up the environment..."
conda activate tensorflow
echo "Conda environment activated."
echo "Current working directory: $(pwd)"

# Define test tasks and seeds
TEST_TASKS=(original filled lines arrows scrambled wider_line open regular random_color irregular)
SEEDS=(123 42 555 789 999)

# Get the task and seed for the current job array index
CURRENT_TASK_INDEX=$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))
CURRENT_SEED_INDEX=$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))
TEST_TASK=${TEST_TASKS[$CURRENT_TASK_INDEX]}
SEED=${SEEDS[$CURRENT_SEED_INDEX]}

echo "Running job for test_task=${TEST_TASK}, seed=${SEED}"

echo "Starting corrected vanilla PB training script..."
python baselines/train_vanilla_pb_final.py \
    --test_task ${TEST_TASK} \
    --seed ${SEED}

echo "Job finished with exit code $? at $(date)" 