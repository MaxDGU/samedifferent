#!/bin/bash
#SBATCH --job-name=weight_space_viz
#SBATCH --output=/scratch/gpfs/mg7411/samedifferent/slurm/weight_viz_%A.out
#SBATCH --error=/scratch/gpfs/mg7411/samedifferent/slurm/weight_viz_%A.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --partition=pli
#SBATCH --account=nam

# Load required modules
module purge
module load anaconda3/2023.9
module load cuda/11.7.0
source activate /scratch/gpfs/mg7411/samedifferent/env/

# Ensure the visualizations directory exists
mkdir -p /scratch/gpfs/mg7411/samedifferent/visualizations

# Run the visualization script
# The script now takes multiple output arguments for pca and tsne
python -u scripts/visualize_weight_space.py \
    --base_dir /scratch/gpfs/mg7411/samedifferent \
    --vanilla_pb_dir /scratch/gpfs/mg7411/samedifferent/results/pb_baselines \
    --vanilla_nat_dir /scratch/gpfs/mg7411/samedifferent/logs_naturalistic_vanilla \
    --output_pca /scratch/gpfs/mg7411/samedifferent/visualizations/weights_pca_all.png \
    --output_tsne /scratch/gpfs/mg7411/samedifferent/visualizations/weights_tsne_all.png

echo "Visualization job finished." 