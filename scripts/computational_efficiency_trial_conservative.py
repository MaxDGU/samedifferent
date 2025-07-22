#!/usr/bin/env python3
"""
Quick trial run of computational efficiency comparison with conservative parameters.
This is for testing the optimization fixes before running the full SLURM job.
"""

import subprocess
import sys

def main():
    """Run a quick trial of the computational efficiency comparison."""
    
    print("🚀 COMPUTATIONAL EFFICIENCY TRIAL (Conservative Parameters)")
    print("=" * 60)
    print("Testing fixes for Second-Order MAML optimization issues")
    print("Conservative learning rates: inner_lr=0.001, outer_lr=0.0001")
    print("Added: gradient clipping, weight decay, learning rate scheduler")
    print("=" * 60)
    
    # Conservative trial parameters
    cmd = [
        "python", "scripts/computational_efficiency_comparison.py",
        "--data_dir", "data/meta_h5/pb",
        "--save_dir", "results/computational_efficiency_trial_conservative",
        "--meta_batch_size", "4",  # Smaller for quick trial
        "--inner_lr", "0.001",     # Conservative
        "--outer_lr", "0.0001",    # Conservative  
        "--max_training_batches", "5",     # Very small for trial
        "--warmup_batches", "20",          # Small warmup
        "--training_adaptation_steps", "3", # Fewer steps
        "--max_test_episodes", "15",       # Small test set
        "--max_adaptation_steps", "10",    # Fewer adaptation steps
        "--target_accuracies", "55.0", "60.0", "65.0",
        "--device", "auto"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n🎉 TRIAL COMPLETED SUCCESSFULLY!")
        print("Check results/computational_efficiency_trial_conservative/ for outputs")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ TRIAL FAILED with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ TRIAL FAILED with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 