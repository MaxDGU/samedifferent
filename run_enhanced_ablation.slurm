#!/bin/bash
#SBATCH --job-name=enhanced_ablation
#SBATCH --output=enhanced_ablation_%j.out
#SBATCH --error=enhanced_ablation_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load necessary modules
module load anaconda3/2021.11
module load cudatoolkit/11.3

# Activate conda environment (adjust path as needed)
conda activate /scratch/gpfs/mg7411/conda_envs/samedifferent

# Navigate to project directory
cd /scratch/gpfs/mg7411/samedifferent

# Install learn2learn if not already installed
pip install learn2learn

# Run the enhanced ablation experiment
echo "Starting enhanced ablation experiment..."
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Set data directory for della
export DATA_DIR="/scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb"

# Run the script
python scripts/run_ablation_experiment.py

echo "Experiment completed at: $(date)"
echo "Check results in: results/local_ablation_experiment/" 