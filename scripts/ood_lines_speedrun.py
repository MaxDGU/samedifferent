#!/usr/bin/env python3
"""
OOD Lines Holdout Speed Run Generator

Generates SLURM script for running speed run OOD sample efficiency with "lines" as holdout task.
"""

import os
import argparse

def generate_ood_lines_slurm():
    """Generate SLURM script for OOD lines holdout speed run."""
    
    # Speed run parameters
    epochs = 15
    meta_batch_size = 64
    vanilla_batch_size = 256
    adaptation_steps = 1
    val_frequency = 100
    inner_lr = 0.02
    outer_lr = 0.002
    vanilla_lr = 0.0005
    
    # Experiment setup
    holdout_task = 'lines'
    seeds = [42, 43, 44, 45, 46]
    methods = ['fomaml', 'second_order', 'vanilla']
    save_dir = 'results/sample_efficiency_ood_multiseed'
    
    # Create directories
    output_dir = os.path.join(save_dir, f'holdout_{holdout_task}')
    slurm_out_dir = os.path.join(output_dir, 'slurm_out')
    os.makedirs(slurm_out_dir, exist_ok=True)
    
    # Generate SLURM script
    script_path = os.path.join(output_dir, 'run_ood_lines_speedrun.slurm')
    
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=ood_lines_speedrun
#SBATCH --output={output_dir}/slurm_out/slurm_%A_%a.out
#SBATCH --error={output_dir}/slurm_out/slurm_%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-{len(seeds)-1}

# Load required modules
module load anaconda3
module load cuda/11.7.0

# Activate Conda environment
conda activate tensorflow

# Get seed for this job
SEEDS=({' '.join(map(str, seeds))})
SEED=${{SEEDS[$SLURM_ARRAY_TASK_ID]}}

echo "Running OOD Lines Holdout Speed Run for seed $SEED"

python scripts/sample_efficiency_comparison_holdout.py \\
    --epochs {epochs} \\
    --seed $SEED \\
    --holdout_task {holdout_task} \\
    --meta_batch_size {meta_batch_size} \\
    --vanilla_batch_size {vanilla_batch_size} \\
    --inner_lr {inner_lr} \\
    --outer_lr {outer_lr} \\
    --vanilla_lr {vanilla_lr} \\
    --adaptation_steps {adaptation_steps} \\
    --val_frequency {val_frequency} \\
    --save_dir {save_dir} \\
    --methods {' '.join(methods)} \\
    --data_dir data/meta_h5/pb

echo "Job for seed $SEED finished."
"""
    
    with open(script_path, 'w') as f:
        f.write(slurm_content)
    
    print("🚀 OOD LINES HOLDOUT SPEED RUN")
    print("="*50)
    print(f"Holdout task: {holdout_task}")
    print(f"Seeds: {seeds}")
    print(f"Methods: {methods}")
    print(f"Speed run parameters:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Meta batch size: {meta_batch_size}")
    print(f"  - Vanilla batch size: {vanilla_batch_size}")
    print(f"  - Adaptation steps: {adaptation_steps}")
    print(f"  - Validation frequency: {val_frequency}")
    print("="*50)
    print(f"\n✅ Generated SLURM script: {script_path}")
    print(f"\n👉 To run the experiments, submit to SLURM:")
    print(f"   sbatch {script_path}")
    print(f"\n📊 Results will be saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate OOD Lines Holdout Speed Run SLURM Script')
    args = parser.parse_args()
    
    generate_ood_lines_slurm()

if __name__ == '__main__':
    main()
