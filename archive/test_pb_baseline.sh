#!/bin/bash
#SBATCH --job-name=test_pb_baseline
#SBATCH --output=slurm_logs/test_pb_baseline_%j.out
#SBATCH --error=slurm_logs/test_pb_baseline_%j.err
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

# Run the test script
python test_pb_baseline.py \
    --test_task regular \
    --architecture conv6 \
    --seed 42 \
    --output_dir results/test_pb_baseline/regular_conv6 \
    --support_size 10 \
    --adaptation_steps 5 \
    --test_adaptation_steps 15 \
    --inner_lr 0.05 \
    --outer_lr 0.001 