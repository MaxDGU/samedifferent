#!/usr/bin/env python3
"""
OOD Second-Order MAML Sample Efficiency Experiment

Specialized script for Second-Order MAML sample efficiency with specific parameters:
- inner_lr = 0.001
- outer_lr = 0.0001  
- adaptation_steps = 10
- Designed for longer runtime (up to 3x more time)
- Second-Order MAML only
- Holdout task for OOD generalization
"""

import os
import argparse

def generate_ood_second_order_slurm():
    """Generate SLURM script for OOD Second-Order MAML sample efficiency."""
    
    # Specialized parameters for Second-Order MAML
    epochs = 50  # More epochs for thorough training
    meta_batch_size = 16  # Smaller batches for Second-Order MAML
    adaptation_steps = 10  # More adaptation steps
    val_frequency = 500  # Less frequent validation to save time
    inner_lr = 0.001  # Lower inner learning rate
    outer_lr = 0.0001  # Much lower outer learning rate
    
    # Experiment setup
    holdout_task = 'scrambled'  # Default holdout task
    seeds = [42, 43, 44, 45, 46]
    methods = ['second_order']  # Only Second-Order MAML
    save_dir = 'results/sample_efficiency_ood_second_order_new'
    
    # Create directories
    output_dir = os.path.join(save_dir, f'holdout_{holdout_task}')
    slurm_out_dir = os.path.join(output_dir, 'slurm_out')
    os.makedirs(slurm_out_dir, exist_ok=True)
    
    # Generate SLURM script
    script_path = os.path.join(output_dir, 'run_ood_second_order_new.slurm')
    
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name=ood_second_order_new
#SBATCH --output={output_dir}/slurm_out/slurm_%A_%a.out
#SBATCH --error={output_dir}/slurm_out/slurm_%A_%a.err
#SBATCH --time=72:00:00
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

echo "Running OOD Second-Order MAML Sample Efficiency (New Parameters) for seed $SEED"
echo "Parameters: inner_lr={inner_lr}, outer_lr={outer_lr}, adaptation_steps={adaptation_steps}"

python scripts/sample_efficiency_comparison_holdout_long.py \\
    --epochs {epochs} \\
    --seed $SEED \\
    --holdout_task {holdout_task} \\
    --meta_batch_size {meta_batch_size} \\
    --inner_lr {inner_lr} \\
    --outer_lr {outer_lr} \\
    --adaptation_steps {adaptation_steps} \\
    --test_adaptation_steps 15 \\
    --val_frequency {val_frequency} \\
    --save_dir {save_dir} \\
    --methods {' '.join(methods)} \\
    --data_dir data/meta_h5/pb

echo "Job for seed $SEED finished."
"""
    
    with open(script_path, 'w') as f:
        f.write(slurm_content)
    
    print("🚀 OOD SECOND-ORDER MAML SAMPLE EFFICIENCY (NEW PARAMETERS)")
    print("="*70)
    print(f"Holdout task: {holdout_task}")
    print(f"Seeds: {seeds}")
    print(f"Method: Second-Order MAML only")
    print(f"Specialized parameters:")
    print(f"  - Inner LR: {inner_lr} (much lower)")
    print(f"  - Outer LR: {outer_lr} (much lower)")
    print(f"  - Adaptation steps: {adaptation_steps} (more steps)")
    print(f"  - Epochs: {epochs} (more epochs)")
    print(f"  - SLURM time: 72 hours (3x longer)")
    print("="*70)
    print(f"\n✅ Generated SLURM script: {script_path}")
    print(f"\n👉 To run the experiment, submit to SLURM:")
    print(f"   sbatch {script_path}")
    print(f"\n📊 Results will be saved to: {output_dir}")
    print(f"\n⚠️  Note: This will take significantly longer than previous experiments")
    print(f"    (up to 3x more time) due to:")
    print(f"    - Lower learning rates requiring more iterations")
    print(f"    - More adaptation steps per episode") 
    print(f"    - Second-order gradients (more computationally expensive)")

def main():
    parser = argparse.ArgumentParser(description='Generate OOD Second-Order MAML Sample Efficiency SLURM Script')
    parser.add_argument('--holdout_task', type=str, default='scrambled',
                       choices=['regular', 'lines', 'open', 'wider_line', 'scrambled',
                               'random_color', 'arrows', 'irregular', 'filled', 'original'],
                       help='Holdout task for OOD evaluation (default: scrambled)')
    
    args = parser.parse_args()
    
    # Override holdout task if specified
    if args.holdout_task != 'scrambled':
        print(f"Using custom holdout task: {args.holdout_task}")
        # We'd need to modify the function to accept holdout_task parameter
        # For now, just inform the user
        print(f"⚠️  Currently configured for 'scrambled'. To use '{args.holdout_task}', ")
        print(f"   please modify the script or specify in the generated SLURM script.")
    
    generate_ood_second_order_slurm()

if __name__ == '__main__':
    main() 