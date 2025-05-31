#!/bin/bash
#SBATCH --job-name=rerun_conv24
#SBATCH --output=slurm_logs/rerun_conv24_%A_%a.out
#SBATCH --error=slurm_logs/rerun_conv24_%A_%a.err
#SBATCH --array=0-9  # 2 architectures * 5 seeds = 10 jobs
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --partition=pli
#SBATCH --account=nam

# Create logs directory
mkdir -p slurm_logs

# Load required modules
module load anaconda3
module load cuda/11.7.0

# Activate conda environment
source activate torch_env

# Define architectures and seeds
declare -a ARCHS=("conv2" "conv4")
declare -a SEEDS=(42 43 44 45 46)  # Using 5 seeds

# Calculate architecture and seed from array index
arch_idx=$((SLURM_ARRAY_TASK_ID / 5))
seed_idx=$((SLURM_ARRAY_TASK_ID % 5))

ARCH=${ARCHS[$arch_idx]}
SEED=${SEEDS[$seed_idx]}

echo "Running architecture: $ARCH with seed: $SEED"

# Run the training and testing script
python -m train_and_test_meta_baselines \
    --data_dir data/pb/pb \
    --output_dir results/meta_baselines/rerun_conv24 \
    --architecture $ARCH \
    --seed $SEED \
    --batch_size 32 \
    --epochs 100 \
    --support_size 10 \
    --adaptation_steps 5 \
    --test_adaptation_steps 15 \
    --inner_lr 0.05 \
    --outer_lr 0.001 