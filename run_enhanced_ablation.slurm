#!/bin/bash
#SBATCH --job-name=enhanced_ablation
#SBATCH --output=enhanced_ablation_%j.out
#SBATCH --error=enhanced_ablation_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load necessary modules
module load anaconda3/2021.11
module load cudatoolkit/11.3

# Activate conda environment
conda activate /scratch/gpfs/mg7411/conda_envs/samedifferent

# Navigate to project directory
cd /scratch/gpfs/mg7411/samedifferent

# Run the enhanced ablation experiment
echo "Starting enhanced ablation experiment..."
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"

python scripts/run_ablation_experiment.py

echo "Experiment completed at: $(date)"
echo "Results saved to: results/ablation_experiment/enhanced_ablation_results.json" 