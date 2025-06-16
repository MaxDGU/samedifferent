#!/bin/bash
#SBATCH --job-name=h5-inspect       # Job name
#SBATCH --output=logs/h5_inspect_%j.out # Standard output and error log
#SBATCH --nodes=1                     # Run all processes on a single node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --mem=1G                      # Job memory request
#SBATCH --time=00:05:00               # Time limit hrs:min:sec

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

# --- Execution ---
echo "Running the HDF5 inspection script..."
python scripts/inspect_h5_file.py /scratch/gpfs/mg7411/data/pb/pb/arrows_support6_test.h5

echo "Job finished at $(date)" 