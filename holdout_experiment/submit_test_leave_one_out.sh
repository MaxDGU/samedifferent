#!/bin/bash
#SBATCH --job-name=pb_test_loo_no_adapt
#SBATCH --output=slurm_logs/test_loo_no_adapt_%A_%a.out
#SBATCH --error=slurm_logs/test_loo_no_adapt_%A_%a.err
#SBATCH --array=0-49  # 10 tasks * 5 seeds = 50 jobs
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL    # Mail events
#SBATCH --mail-user=mg7411@princeton.edu

# Create logs directory
mkdir -p slurm_logs

# Array of PB tasks and seeds
PB_TASKS=(regular lines open wider_line scrambled random_color arrows irregular filled original)
SEEDS=(42 43 44 45 46)  # Using first 5 seeds

# Calculate task and seed indices
TASK_IDX=$((SLURM_ARRAY_TASK_ID % 10))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 10))

# Get task and seed for this job
TASK=${PB_TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "Testing task $TASK with seed $SEED without adaptation"

# Run the testing script
python test_leave_one_out.py \
    --pb_data_dir data/pb/pb \
    --output_dir results/leave_one_out_test_no_adapt \
    --held_out_task $TASK \
    --seed $SEED \
    --batch_size 16