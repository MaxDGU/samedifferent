#!/bin/bash
#SBATCH --job-name=sample_efficiency_comparison
#SBATCH --output=logs/sample_efficiency_%j.out
#SBATCH --error=logs/sample_efficiency_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load required modules
module purge
module load anaconda3/2023.9
module load cudatoolkit/11.7
module load cudnn/cuda-11.x/8.5.0

# Activate conda environment (adjust path as needed)
source activate /home/mg7411/.conda/envs/metasamedifferent

# Set working directory
cd /scratch/gpfs/mg7411/samedifferent/metasamedifferent

# Create logs directory
mkdir -p logs

# Run the sample efficiency comparison
python scripts/sample_efficiency_comparison.py \
    --epochs 50 \
    --seed 42 \
    --meta_batch_size 8 \
    --vanilla_batch_size 32 \
    --inner_lr 0.001 \
    --outer_lr 0.0001 \
    --vanilla_lr 0.0001 \
    --adaptation_steps 5 \
    --val_frequency 500 \
    --save_dir results/sample_efficiency_comparison \
    --methods fomaml second_order vanilla \
    --data_dir /scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb

# After completion, generate plots
python scripts/plot_sample_efficiency_results.py \
    --results_dir results/sample_efficiency_comparison/seed_42 \
    --individual_plots

echo "Sample efficiency comparison completed!" 