#!/bin/bash
#SBATCH --job-name=nat_pca_viz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/naturalistic_pca_viz_%j.out

# Activate conda environment
source ~/.bashrc
conda activate tensorflow

# Navigate to the project directory
cd /scratch/gpfs/mg7411/samedifferent

echo "Setting up the environment..."
conda activate tensorflow
echo "Conda environment activated."
echo "Current working directory: $(pwd)"

# Set PYTHONPATH to include the project root
export PYTHONPATH="/scratch/gpfs/mg7411/samedifferent:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# --- Model Paths and Seeds ---
META_SEED=111
VANILLA_SEED=123

META_PATH="/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_meta/conv6lr/seed_${META_SEED}/conv6lr/seed_${META_SEED}/conv6lr_best.pth"
VANILLA_PATH="/scratch/gpfs/mg7411/samedifferent/logs_naturalistic_vanilla/conv6lr/seed_${VANILLA_SEED}/best_model.pt"
SAVE_DIR="./visualizations/domain_adaptation_pca/naturalistic_to_pb"

echo "Running Naturalistic to PB Adaptation Visualization"
echo "Meta Model: ${META_PATH}"
echo "Vanilla Model: ${VANILLA_PATH}"

python scripts/visualize_naturalistic_to_pb_adaptation.py \
    --meta_model_path ${META_PATH} \
    --vanilla_model_path ${VANILLA_PATH} \
    --meta_seed ${META_SEED} \
    --vanilla_seed ${VANILLA_SEED} \
    --save_dir ${SAVE_DIR}

echo "Job finished with exit code $? at $(date)" 