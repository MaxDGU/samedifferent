#!/bin/bash
#SBATCH --job-name=viz-mt-pca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz_mt_pca_%A.out
#SBATCH --error=logs/viz_mt_pca_%A.err

# --- Start of commands ---
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Starting job at: $(date)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules
module load anaconda3/2023.9
conda activate tensorflow

# Set PYTHONPATH to include the project root
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Run the python script
echo "Running PCA visualization script..."
python scripts/visualize_multi_task_adaptation_pca.py --output_dir 'visualizations/multi_task_adaptation_pca'

echo "Job finished at: $(date)" 