#!/bin/bash
#SBATCH --job-name=pb-adapt-pca     # Job name
#SBATCH --output=logs/pb_adapt_pca_%j.out # Standard output and error log
#SBATCH --nodes=1                     # Run all processes on a single node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=64G                     # Job memory request
#SBATCH --gres=gpu:1                  # Request one GPU
#SBATCH --time=01:00:00               # Time limit hrs:min:sec

# --- Environment Setup ---
echo "Setting up the environment..."
source /usr/share/Modules/init/bash
module load anaconda3/2023.9
conda activate tensorflow
echo "Conda environment activated."

# --- Set Project Path ---
export GIT_REPO_PATH="/scratch/gpfs/mg7411/samedifferent"
export PYTHONPATH="${PYTHONPATH}:${GIT_REPO_PATH}"
cd "${GIT_REPO_PATH}"
echo "Current working directory: $(pwd)"
echo "PYTHONPATH: ${PYTHONPATH}"

# --- Execution ---
echo "Running the PB Adaptation PCA script..."
python scripts/visualize_pb_adaptation_pca.py \
    --vanilla_model_path "/scratch/gpfs/mg7411/samedifferent/single_task/results/pb_single_task/regular/conv6/seed_80/initial_model.pth" \
    --meta_model_path "/scratch/gpfs/mg7411/samedifferent/maml_pbweights_conv6/model_seed_3_pretesting.pt" \
    --data_path "/scratch/gpfs/mg7411/data/pb/pb/arrows_support6_test.h5" \
    --output_dir "/scratch/gpfs/mg7411/samedifferent/visualizations/adaptation_pca" \
    --lr 0.01 \
    --steps 3 \
    --adaptation_batch_size 64

echo "Job finished with exit code $? at $(date)" 