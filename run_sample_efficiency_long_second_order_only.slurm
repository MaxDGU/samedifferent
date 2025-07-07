#!/bin/bash
#SBATCH --job-name=sample_efficiency_iid_long_second_order_only
#SBATCH --output=logs/sample_efficiency_iid_long_second_order_only_%j.out
#SBATCH --error=logs/sample_efficiency_iid_long_second_order_only_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

# Load required modules
module load anaconda3/2022.5
module load cudatoolkit/12.6

# Activate conda environment
source /usr/licensed/anaconda3/2022.5/etc/profile.d/conda.sh
conda activate tensorflow

# Create logs directory
mkdir -p logs

# Run the LONG IID sample efficiency comparison - SECOND-ORDER MAML ONLY
echo "Starting IN-DISTRIBUTION Sample Efficiency Comparison (LONG VERSION - SECOND-ORDER MAML ONLY)"
echo "============================================================================================="
echo "This experiment completes the Second-Order MAML portion of the IID comparison:"
echo "- Training and validation on the SAME tasks (all 10 PB tasks)"
echo "- Tests learning efficiency within the training distribution"
echo "- EXTENDED TIME ALLOCATION: 48 hours to ensure completion"
echo ""
echo "LONG VERSION PARAMETERS:"
echo "- Adaptation steps: 10 (training) / 15 (testing)"
echo "- Epochs: 25 (thorough evaluation)"
echo "- Validation frequency: 1000 (frequent monitoring)"
echo "- Meta batch size: 16"
echo "- Expected runtime: 24-48 hours"
echo ""
echo "All tasks: regular, lines, open, wider_line, scrambled, random_color, arrows, irregular, filled, original"
echo "METHOD: Second-Order MAML ONLY"
echo ""

python scripts/sample_efficiency_comparison_long.py \
    --epochs 25 \
    --seed 42 \
    --meta_batch_size 16 \
    --vanilla_batch_size 64 \
    --inner_lr 0.05 \
    --outer_lr 0.001 \
    --vanilla_lr 0.0001 \
    --adaptation_steps 10 \
    --test_adaptation_steps 15 \
    --val_frequency 1000 \
    --save_dir results/sample_efficiency_comparison_long \
    --methods second_order \
    --data_dir /scratch/gpfs/mg7411/samedifferent/data/meta_h5/pb

echo ""
echo "============================================================================================="
echo "LONG IID Sample Efficiency Experiment (Second-Order MAML ONLY) Completed!"
echo "Results saved to: results/sample_efficiency_comparison_long/seed_42/"
echo ""
echo "This completes the Second-Order MAML portion of the IID comparison."
echo "Results can be combined with previously completed FOMAML results."
echo "=============================================================================================" 